import os
import pickle
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry

import db
import claude_client as cc

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "."


@app.on_event("startup")
def _startup() -> None:
    db.init()


@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            if item.endswith("_data") and os.path.isdir(item):
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine)
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})


@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(request: Request, book_id: str):
    """Helper to just go to chapter 0."""
    return await read_chapter(request=request, book_id=book_id, chapter_index=0)


@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_index: int):
    """The main reader interface."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]

    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx
    })


@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)


# =====================================================================
# Highlights + translations
# =====================================================================

class CreateHighlight(BaseModel):
    book_id: str
    chapter_index: int
    kind: str  # 'translate' | 'chat'
    text: str
    start_hint: str = ""
    end_hint: str = ""


@app.post("/api/highlights")
async def api_create_highlight(payload: CreateHighlight):
    if payload.kind not in ("translate", "chat"):
        raise HTTPException(status_code=400, detail="invalid kind")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    hid = db.create_highlight(
        payload.book_id, payload.chapter_index, payload.kind,
        text, payload.start_hint, payload.end_hint,
    )

    translation_payload = None
    if payload.kind == "translate":
        prompt, schema, system = cc.translate_sentence_prompt(text)
        try:
            result = await cc.call_json(prompt, schema, system)
        except cc.ClaudeError as e:
            db.delete_highlight(hid)
            raise HTTPException(status_code=502, detail=f"translation failed: {e}")
        db.save_translation(hid, result["translation"], result.get("context", ""))
        translation_payload = {
            "sentence_en": result["translation"],
            "context_note": result.get("context", ""),
        }

    return {
        "id": hid,
        "kind": payload.kind,
        "text": text,
        "translation": translation_payload,
    }


@app.get("/api/highlights/{highlight_id}/words")
async def api_highlight_words(highlight_id: int):
    h = db.get_highlight(highlight_id)
    if not h:
        raise HTTPException(status_code=404, detail="highlight not found")
    if h["kind"] != "translate":
        raise HTTPException(status_code=400, detail="not a translate highlight")

    tr = db.get_translation(highlight_id)
    if not tr:
        raise HTTPException(status_code=409, detail="translation not yet ready")

    if tr.get("words"):
        return {"words": tr["words"]}

    prompt, schema, system = cc.word_by_word_prompt(h["text"], tr["sentence_en"])
    try:
        result = await cc.call_json(prompt, schema, system)
    except cc.ClaudeError as e:
        raise HTTPException(status_code=502, detail=f"word breakdown failed: {e}")
    words = result.get("words", [])
    db.save_words(highlight_id, words)
    return {"words": words}


@app.get("/api/highlights/{highlight_id}/verbs")
async def api_highlight_verbs(highlight_id: int):
    h = db.get_highlight(highlight_id)
    if not h:
        raise HTTPException(status_code=404, detail="highlight not found")
    if h["kind"] != "translate":
        raise HTTPException(status_code=400, detail="not a translate highlight")

    tr = db.get_translation(highlight_id)
    if not tr:
        raise HTTPException(status_code=409, detail="translation not yet ready")

    if tr.get("verbs") is not None:
        return {"verbs": tr["verbs"]}

    prompt, schema, system = cc.verbs_prompt(h["text"])
    try:
        result = await cc.call_json(prompt, schema, system)
    except cc.ClaudeError as e:
        raise HTTPException(status_code=502, detail=f"verb breakdown failed: {e}")
    verbs = result.get("verbs", [])
    db.save_verbs(highlight_id, verbs)
    return {"verbs": verbs}


@app.get("/api/highlights/{book_id}/{chapter_index}")
async def api_list_highlights(book_id: str, chapter_index: int):
    return {"highlights": db.list_highlights(book_id, chapter_index)}


@app.delete("/api/highlights/{highlight_id}")
async def api_delete_highlight(highlight_id: int):
    db.delete_highlight(highlight_id)
    return {"ok": True}


# =====================================================================
# Chat (streaming)
# =====================================================================

class ChatMessage(BaseModel):
    message: str


@app.get("/api/chat/{highlight_id}")
async def api_chat_history(highlight_id: int):
    h = db.get_highlight(highlight_id)
    if not h:
        raise HTTPException(status_code=404, detail="highlight not found")
    return {
        "passage": h["text"],
        "messages": db.list_chat_messages(highlight_id),
    }


@app.post("/api/chat/{highlight_id}")
async def api_chat_send(highlight_id: int, payload: ChatMessage):
    h = db.get_highlight(highlight_id)
    if not h:
        raise HTTPException(status_code=404, detail="highlight not found")

    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="empty message")

    db.add_chat_message(highlight_id, "user", user_msg)
    messages = [{"role": m["role"], "content": m["content"]}
                for m in db.list_chat_messages(highlight_id)]

    system = (
        cc.CHAT_SYSTEM
        + "\n\nPASSAGE BEING DISCUSSED (French):\n«"
        + h["text"] + "»"
    )

    async def event_stream():
        collected: list[str] = []
        try:
            async for chunk in cc.stream_chat(messages, system):
                collected.append(chunk)
                yield f"data: {_sse_escape(chunk)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)[:300]}\n\n"
        finally:
            full = "".join(collected).strip()
            if full:
                db.add_chat_message(highlight_id, "assistant", full)
            yield "event: done\ndata: end\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_escape(s: str) -> str:
    # SSE separates events with blank lines; literal \n inside data must be escaped
    # as multiple data: lines. Simpler here: JSON-encode the chunk.
    import json as _json
    return _json.dumps(s, ensure_ascii=False)


# =====================================================================
# Vocab + SRS
# =====================================================================

class AddVocab(BaseModel):
    fr: str
    en: str
    context_sentence: str = ""
    source_book_id: Optional[str] = None


@app.post("/api/vocab")
async def api_add_vocab(payload: AddVocab):
    fr = payload.fr.strip()
    en = payload.en.strip()
    if not fr or not en:
        raise HTTPException(status_code=400, detail="fr and en required")
    vid = db.add_vocab(fr, en, payload.context_sentence.strip(), payload.source_book_id)
    return {"id": vid}


@app.get("/api/vocab/due")
async def api_vocab_due():
    return {"cards": db.list_due_vocab(), "stats": db.vocab_stats()}


@app.get("/api/vocab")
async def api_vocab_list():
    return {"cards": db.list_all_vocab(), "stats": db.vocab_stats()}


class ReviewVocab(BaseModel):
    grade: int


@app.post("/api/vocab/{vocab_id}/review")
async def api_review_vocab(vocab_id: int, payload: ReviewVocab):
    if payload.grade not in (0, 1, 2, 3):
        raise HTTPException(status_code=400, detail="grade must be 0..3")
    card = db.review_vocab(vocab_id, payload.grade)
    if card is None:
        raise HTTPException(status_code=404, detail="vocab not found")
    return {"card": card}


@app.get("/study", response_class=HTMLResponse)
async def study_view(request: Request):
    return templates.TemplateResponse("study.html", {"request": request})


# =====================================================================
# Comprehension (per-chapter quiz)
# =====================================================================

VALID_DIFFICULTIES = ("debutant", "intermediaire", "avance")


class ComprehensionReq(BaseModel):
    book_id: str
    chapter_index: int
    difficulty: str
    regenerate: bool = False


@app.post("/api/comprehension")
async def api_comprehension(payload: ComprehensionReq):
    if payload.difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail="invalid difficulty")

    book = load_book_cached(payload.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="book not found")
    if payload.chapter_index < 0 or payload.chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="chapter not found")

    if not payload.regenerate:
        cached = db.get_comprehension(payload.book_id, payload.chapter_index, payload.difficulty)
        if cached is not None:
            return {"questions": cached, "cached": True}

    chapter_text = book.spine[payload.chapter_index].text
    if not chapter_text or len(chapter_text.strip()) < 80:
        raise HTTPException(status_code=422, detail="chapter is too short for comprehension")

    prompt, schema, system = cc.comprehension_prompt(chapter_text, payload.difficulty)
    try:
        result = await cc.call_json(prompt, schema, system)
    except cc.ClaudeError as e:
        raise HTTPException(status_code=502, detail=f"comprehension failed: {e}")
    questions = result.get("questions", [])
    db.save_comprehension(payload.book_id, payload.chapter_index, payload.difficulty, questions)
    return {"questions": questions, "cached": False}


@app.get("/api/comprehension/{book_id}/{chapter_index}")
async def api_get_comprehension(book_id: str, chapter_index: int, difficulty: str):
    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail="invalid difficulty")
    qs = db.get_comprehension(book_id, chapter_index, difficulty)
    if qs is None:
        raise HTTPException(status_code=404, detail="no comprehension cached")
    return {"questions": qs}


class WordInContext(BaseModel):
    word: str
    sentence: str


@app.post("/api/word-in-context")
async def api_word_in_context(payload: WordInContext):
    word = payload.word.strip()
    sentence = payload.sentence.strip()
    if not word:
        raise HTTPException(status_code=400, detail="empty word")
    prompt, schema, system = cc.word_in_context_prompt(word, sentence or word)
    try:
        result = await cc.call_json(prompt, schema, system)
    except cc.ClaudeError as e:
        raise HTTPException(status_code=502, detail=f"word translation failed: {e}")
    return result


class TranslateText(BaseModel):
    text: str


@app.post("/api/translate-text")
async def api_translate_text(payload: TranslateText):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    prompt, schema, system = cc.translate_text_prompt(text)
    try:
        result = await cc.call_json(prompt, schema, system)
    except cc.ClaudeError as e:
        raise HTTPException(status_code=502, detail=f"translation failed: {e}")
    return result


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
