#!/usr/bin/env python3
"""
Web server per l'agente culinario.

Avvio:
    python server.py
    # oppure
    uvicorn server:app --reload
"""

import json
import re
import secrets
import datetime
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import bcrypt
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

from agent import stream_agent, load_config, _get_collection  # noqa: E402 (after load_dotenv)

import os
# DATA_DIR can be set via env to decouple user data from the app directory.
# Defaults to ./data/ next to server.py.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = DATA_DIR / "users.json"
SAVED_DIR = DATA_DIR / "recipes"
SAVED_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Il tempo, i luoghi, la gente e i sapori")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.on_event("startup")
async def startup():
    # Load the embedding model in the background so the health check passes
    # immediately. The first chat request may be slow if the model isn't ready yet.
    threading.Thread(target=_get_collection, daemon=True).start()

ADMIN_USERNAME = "wurzia"


def _hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_pw(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# In-memory: sessions {token -> username}, histories {username -> [messages]}
_sessions: dict[str, str] = {}
_sessions_lock = threading.Lock()
_histories: dict[str, list[dict]] = {}


# ── User store (users.json) ───────────────────────────────────────────────────

def load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


def user_dir(username: str) -> Path:
    d = SAVED_DIR / username
    d.mkdir(exist_ok=True)
    return d


# ── Auth helpers ──────────────────────────────────────────────────────────────

def require_user(request: Request) -> str:
    token = request.cookies.get("session")
    if not token:
        raise HTTPException(status_code=401, detail="Non autenticato")
    with _sessions_lock:
        username = _sessions.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Sessione non valida")
    return username


def require_admin(request: Request) -> str:
    username = require_user(request)
    if username != ADMIN_USERNAME:
        raise HTTPException(status_code=403, detail="Accesso riservato all'amministratore")
    return username


def _new_session(response: Response, username: str) -> None:
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        _sessions[token] = username
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=30 * 24 * 3600)


# ── Auth routes ───────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str
    password: str


@app.get("/api/first-run")
def first_run():
    return {"first_run": not USERS_FILE.exists() or not load_users()}


@app.post("/api/register")
def register(req: AuthRequest, response: Response, request: Request):
    users = load_users()
    if users:
        # Not first run — require admin caller
        try:
            caller = require_user(request)
        except HTTPException:
            raise HTTPException(status_code=403, detail="Solo l'amministratore può creare account")
        if caller != ADMIN_USERNAME:
            raise HTTPException(status_code=403, detail="Solo l'amministratore può creare account")

    username = req.username.strip()
    if len(username) < 2 or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Username (min 2 car.) e password (min 4 car.) richiesti")
    if username in users:
        raise HTTPException(status_code=409, detail="Username già in uso")

    users[username] = {
        "password_hash": _hash_pw(req.password),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_users(users)
    _new_session(response, username)
    return {"username": username}


@app.post("/api/login")
def login(req: AuthRequest, response: Response):
    users = load_users()
    username = req.username.strip()
    entry = users.get(username)
    if not entry or not _verify_pw(req.password, entry["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenziali non valide")
    _new_session(response, username)
    return {"username": username}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token:
        with _sessions_lock:
            _sessions.pop(token, None)
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    with _sessions_lock:
        username = _sessions.get(token)
    if not username:
        return None
    return {"username": username, "is_admin": username == ADMIN_USERNAME}


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.get("/admin")
def admin_page():
    return FileResponse(ROOT / "static" / "admin.html")


@app.get("/api/admin/users")
def admin_list_users(request: Request):
    require_admin(request)
    users = load_users()
    result = []
    for uname, info in users.items():
        recipe_count = len(list((SAVED_DIR / uname).glob("*.md"))) if (SAVED_DIR / uname).exists() else 0
        result.append({
            "username": uname,
            "created_at": info.get("created_at", ""),
            "recipes": recipe_count,
        })
    return result


@app.post("/api/admin/users")
def admin_create_user(req: AuthRequest, request: Request):
    require_admin(request)
    users = load_users()
    username = req.username.strip()
    if len(username) < 2 or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Username (min 2) e password (min 4) richiesti")
    if username in users:
        raise HTTPException(status_code=409, detail="Username già in uso")
    users[username] = {
        "password_hash": _hash_pw(req.password),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_users(users)
    return {"username": username}


@app.delete("/api/admin/users/{username}")
def admin_delete_user(username: str, request: Request):
    require_admin(request)
    if username == ADMIN_USERNAME:
        raise HTTPException(status_code=400, detail="Non puoi eliminare l'amministratore")
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="Utente non trovato")
    del users[username]
    save_users(users)
    # Revoke sessions
    with _sessions_lock:
        for token, uname in list(_sessions.items()):
            if uname == username:
                del _sessions[token]
    # Delete recipe folder
    udir = SAVED_DIR / username
    if udir.exists():
        shutil.rmtree(udir)
    _histories.pop(username, None)
    return {"ok": True}


# ── App routes ────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    username = require_user(request)
    history = _histories.setdefault(username, [])
    config = load_config()

    def generate():
        assistant_text = ""
        for event in stream_agent(req.message, config, history):
            if event["type"] == "text":
                assistant_text += event["content"]
            data = json.dumps(event, ensure_ascii=False)
            yield f"data: {data}\n\n"
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": assistant_text})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SaveRequest(BaseModel):
    title: str
    content: str
    sources: list[dict] = []


@app.post("/api/save")
def save_recipe(req: SaveRequest, request: Request):
    username = require_user(request)
    safe_title = re.sub(r"[^\w\s-]", "", req.title).strip().replace(" ", "_")[:60] or "ricetta"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = user_dir(username) / f"{timestamp}_{safe_title}.md"

    lines = [
        f"# {req.title}\n",
        f"*Salvata il {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}*\n\n",
        req.content,
    ]
    if req.sources:
        lines.append("\n\n---\n## Fonti\n")
        for s in req.sources:
            name = s.get("titolo") or s.get("fonte") or "fonte"
            url = s.get("url", "")
            fonte = s.get("fonte", "")
            lines.append(f"- [{name}]({url}) — {fonte}\n" if url else f"- {name} — {fonte}\n")

    filename.write_text("".join(lines), encoding="utf-8")
    return {"saved": filename.name}


@app.post("/api/clear")
def clear_session(request: Request):
    username = require_user(request)
    _histories.pop(username, None)
    return {"ok": True}


@app.get("/api/saved")
def list_saved(request: Request):
    username = require_user(request)
    files = sorted(user_dir(username).glob("*.md"), reverse=True)
    return [{"name": f.stem, "file": f.name} for f in files[:50]]


@app.get("/api/saved/{filename}")
def get_saved(filename: str, request: Request):
    username = require_user(request)
    path = user_dir(username) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Non trovata")
    return {"content": path.read_text(encoding="utf-8")}


@app.delete("/api/saved/{filename}")
def delete_saved(filename: str, request: Request):
    username = require_user(request)
    path = user_dir(username) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Non trovata")
    path.unlink()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
