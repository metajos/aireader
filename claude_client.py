"""
Thin async wrapper around the `claude` CLI for structured translations and streaming chat.

Uses a slim CLI invocation (custom --system-prompt, --tools "", --disable-slash-commands,
--no-session-persistence, --model haiku, --effort low) to keep cost & latency low.
"""

import asyncio
import json
import os
import shutil
from typing import AsyncIterator, Optional

import db

CLAUDE_BIN = shutil.which("claude") or "claude"
_DEFAULT_TIMEOUT = float(os.environ.get("CLAUDE_TIMEOUT", "120"))

_SLIM_ARGS = [
    "-p",
    "--model", "haiku",
    "--effort", "low",
    "--tools", "",
    "--disable-slash-commands",
    "--no-session-persistence",
]


class ClaudeError(RuntimeError):
    pass


async def call_json(prompt: str, schema: dict, system: str, use_cache: bool = True) -> dict:
    """One-shot structured call. Returns the dict matching `schema`."""
    cache_key = json.dumps({"sys": system, "p": prompt, "s": schema}, sort_keys=True, ensure_ascii=False)
    if use_cache:
        cached = db.cache_get(cache_key)
        if cached is not None:
            return cached

    args = [
        CLAUDE_BIN, *_SLIM_ARGS,
        "--system-prompt", system,
        "--output-format", "json",
        "--json-schema", json.dumps(schema, ensure_ascii=False),
        prompt,
    ]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",  # escape project dir so global git-check hooks bail
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=_DEFAULT_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise ClaudeError("claude -p timed out")

    if proc.returncode != 0:
        raise ClaudeError(
            f"claude exited {proc.returncode}: {stderr_b.decode('utf-8', errors='replace')[:500]}"
        )

    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    parsed = _extract_json_payload(stdout)

    if use_cache:
        db.cache_put(cache_key, parsed)
    return parsed


def _extract_json_payload(stdout: str) -> dict:
    """
    claude -p with --json-schema returns a wrapper. The schema-conformant object is in
    the `structured_output` field. Fall back to parsing `result` as JSON.
    """
    try:
        outer = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ClaudeError(f"could not parse claude stdout as JSON: {e}; first 200 chars: {stdout[:200]}")

    if not isinstance(outer, dict):
        raise ClaudeError(f"unexpected claude output shape: {type(outer).__name__}")

    if outer.get("is_error"):
        raise ClaudeError(f"claude error: {outer.get('result', '')[:300]}")

    so = outer.get("structured_output")
    if isinstance(so, dict):
        return so

    result = outer.get("result")
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            raise ClaudeError(f"`result` is a string but not JSON: {result[:200]}")

    raise ClaudeError(f"no structured_output or result in claude reply: {list(outer.keys())}")


async def stream_chat(messages: list[dict], system: str) -> AsyncIterator[str]:
    """
    Stream a chat response token-by-token.
    `messages` is the full conversation so far; the last message must be a user turn.
    Yields text deltas as they arrive.
    """
    prompt = _serialize_conversation(messages)

    args = [
        CLAUDE_BIN, *_SLIM_ARGS,
        "--system-prompt", system,
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",  # required when --print pairs with --output-format stream-json
        prompt,
    ]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",
    )

    assert proc.stdout is not None
    first_turn_done = False
    try:
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                evt = json.loads(text)
            except json.JSONDecodeError:
                continue
            # Stop after the first assistant message completes — anything after
            # is from a hook-triggered re-prompt, not the user's question.
            if first_turn_done:
                continue
            for chunk in _extract_text_deltas(evt):
                yield chunk
            if _is_message_end(evt):
                first_turn_done = True
    finally:
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()


def _is_message_end(evt: dict) -> bool:
    if not isinstance(evt, dict) or evt.get("type") != "stream_event":
        return False
    inner = evt.get("event") or {}
    if not isinstance(inner, dict):
        return False
    if inner.get("type") == "message_stop":
        return True
    if inner.get("type") == "message_delta":
        return (inner.get("delta") or {}).get("stop_reason") == "end_turn"
    return False


def _serialize_conversation(messages: list[dict]) -> str:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must end with a user turn")
    if len(messages) == 1:
        return messages[0]["content"]
    lines = []
    for m in messages[:-1]:
        tag = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{tag}: {m['content']}")
    lines.append(f"User: {messages[-1]['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _extract_text_deltas(evt: dict) -> list[str]:
    """
    Extract user-visible text from stream-json events. We only want text_delta —
    NOT thinking_delta or signature_delta which appear when extended thinking is on.
    """
    if not isinstance(evt, dict) or evt.get("type") != "stream_event":
        return []
    inner = evt.get("event") or {}
    if not isinstance(inner, dict) or inner.get("type") != "content_block_delta":
        return []
    delta = inner.get("delta") or {}
    if delta.get("type") != "text_delta":
        return []
    txt = delta.get("text")
    return [txt] if isinstance(txt, str) and txt else []


# ---------- Prompt templates ----------

TRANSLATE_SYSTEM = (
    "You translate French to English for a language learner. "
    "Translate naturally (not literally) and add a 1-2 sentence note on idiom, "
    "grammar, or nuance only if non-obvious. Always respect the JSON schema."
)

WORD_SYSTEM = (
    "You annotate French text word-by-word for a language learner. "
    "For each token give a direct English gloss and a SHORT (<=8 words) grammatical note. "
    "Skip punctuation. Preserve order. Always respect the JSON schema."
)

CHAT_SYSTEM = (
    "You are a helpful French tutor. The user has highlighted a passage from a French book "
    "and wants to discuss it. Be concise and clear. Use French quotes where helpful."
)

VERBS_SYSTEM = (
    "You identify every verb in a French sentence and produce full conjugation tables. "
    "For each verb, return: the infinitive, an English gloss, the form_in_text (the bare "
    "conjugated form as it appeared in the text, WITHOUT subject pronoun, but INCLUDING any "
    "auxiliary for compound tenses, e.g. 'voudrais', 'a voulu'), and tense_in_text (a short "
    "label like 'conditionnel présent, 1sg'). Then provide conjugations for six tenses: "
    "present, passé composé, imparfait, futur simple, conditionnel présent, subjonctif présent. "
    "Each conjugation array MUST have exactly 6 entries in this fixed order corresponding to "
    "je, tu, il/elle, nous, vous, ils/elles. CRITICAL: each entry is the CONJUGATED FORM ONLY "
    "— do NOT include the subject pronoun and do NOT include 'que' for the subjunctive. "
    "Examples for vouloir: "
    "present=['veux','veux','veut','voulons','voulez','veulent']; "
    "passe_compose=['ai voulu','as voulu','a voulu','avons voulu','avez voulu','ont voulu']; "
    "subjonctif=['veuille','veuilles','veuille','voulions','vouliez','veuillent']. "
    "If the highlight contains no verbs, return an empty array. Respond strictly per the JSON schema."
)


def translate_sentence_prompt(sentence: str) -> tuple[str, dict, str]:
    schema = {
        "type": "object",
        "properties": {
            "translation": {"type": "string"},
            "context": {"type": "string"},
        },
        "required": ["translation", "context"],
        "additionalProperties": False,
    }
    prompt = f"FRENCH:\n{sentence}"
    return prompt, schema, TRANSLATE_SYSTEM


def word_by_word_prompt(sentence: str, translation: str) -> tuple[str, dict, str]:
    schema = {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fr": {"type": "string"},
                        "en": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["fr", "en", "note"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["words"],
        "additionalProperties": False,
    }
    prompt = f"FRENCH: {sentence}\nFULL TRANSLATION: {translation}"
    return prompt, schema, WORD_SYSTEM


def verbs_prompt(sentence: str) -> tuple[str, dict, str]:
    conj_array = {"type": "array", "items": {"type": "string"}}
    schema = {
        "type": "object",
        "properties": {
            "verbs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "infinitive": {"type": "string"},
                        "english": {"type": "string"},
                        "form_in_text": {"type": "string"},
                        "tense_in_text": {"type": "string"},
                        "group": {"type": "string"},
                        "conjugations": {
                            "type": "object",
                            "properties": {
                                "present": conj_array,
                                "passe_compose": conj_array,
                                "imparfait": conj_array,
                                "futur": conj_array,
                                "conditionnel": conj_array,
                                "subjonctif": conj_array,
                            },
                            "required": [
                                "present", "passe_compose", "imparfait",
                                "futur", "conditionnel", "subjonctif",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "infinitive", "english", "form_in_text",
                        "tense_in_text", "group", "conjugations",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verbs"],
        "additionalProperties": False,
    }
    prompt = f"FRENCH:\n{sentence}"
    return prompt, schema, VERBS_SYSTEM


# ---------- Comprehension ----------

COMPREHENSION_SYSTEM = (
    "You generate French-language comprehension exercises for a learner reading a French book. "
    "Given the chapter text and a difficulty level, produce exactly 8 questions IN FRENCH about "
    "the content, mixing three question types in roughly equal proportion: multiple_choice, "
    "fill_gap, conjugate. "
    "For multiple_choice: provide question_fr, exactly 4 options_fr, and `answer` set to the "
    "EXACT TEXT of the correct option (must match one of options_fr exactly). "
    "For fill_gap: question_fr contains a single blank rendered as '____' (four underscores). "
    "`answer` is the missing word(s), lowercase, no surrounding punctuation. "
    "For conjugate: question_fr is a brief instruction like 'Conjuguez VERBE à la TENSE, "
    "1ère personne du singulier.', `infinitive` is the infinitive (e.g. 'vouloir'), `tense` "
    "is one of present/passe_compose/imparfait/futur/conditionnel/subjonctif, and `answer` "
    "is the bare conjugated form (no subject pronoun, no 'que'). "
    "Always provide a short explanation_fr (1-2 sentences) of why the answer is correct. "
    "Unused fields must be empty strings ('') or empty arrays — never null. "
    "Difficulty scaling: debutant uses common vocabulary + simple tenses (present, passé composé) "
    "and direct comprehension; intermediaire uses imparfait/futur/conditionnel and some inference; "
    "avance uses subjonctif/literary phrasing and deeper analysis. "
    "Respond strictly per the JSON schema."
)


def comprehension_prompt(chapter_text: str, difficulty: str) -> tuple[str, dict, str]:
    if difficulty not in ("debutant", "intermediaire", "avance"):
        raise ValueError("difficulty must be debutant|intermediaire|avance")
    snippet = chapter_text[:8000]
    question_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["multiple_choice", "fill_gap", "conjugate"]},
            "question_fr": {"type": "string"},
            "options_fr": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string"},
            "blank_index": {"type": "integer"},
            "infinitive": {"type": "string"},
            "tense": {"type": "string"},
            "explanation_fr": {"type": "string"},
        },
        "required": [
            "type", "question_fr", "options_fr", "answer",
            "blank_index", "infinitive", "tense", "explanation_fr",
        ],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "questions": {"type": "array", "items": question_schema},
        },
        "required": ["questions"],
        "additionalProperties": False,
    }
    prompt = f"DIFFICULTY: {difficulty}\n\nCHAPTER TEXT:\n{snippet}"
    return prompt, schema, COMPREHENSION_SYSTEM


# ---------- Word-in-context (tooltip translation) ----------

WORD_IN_CONTEXT_SYSTEM = (
    "You translate a single French word using its surrounding sentence as disambiguation. "
    "Return the English translation plus a short note (<= 8 words) about part of speech / "
    "grammatical features. Respond strictly per the JSON schema."
)


def word_in_context_prompt(word: str, sentence: str) -> tuple[str, dict, str]:
    schema = {
        "type": "object",
        "properties": {
            "translation": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["translation", "note"],
        "additionalProperties": False,
    }
    prompt = f"WORD: {word}\nSENTENCE: {sentence}"
    return prompt, schema, WORD_IN_CONTEXT_SYSTEM


# ---------- Sentence translate (reusable for "translate the question") ----------

def translate_text_prompt(text: str) -> tuple[str, dict, str]:
    """Alias for `translate_sentence_prompt` for clarity in the comprehension feature."""
    return translate_sentence_prompt(text)
