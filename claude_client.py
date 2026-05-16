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
