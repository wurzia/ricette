#!/usr/bin/env python3
"""
Agente culinario — cerca, compone e varia ricette
basandosi esclusivamente sulle fonti configurate come grounding.

Uso:
    python agent.py
    python agent.py "cerca una ricetta per la carbonara"
"""

import sys
import re
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent
SOURCES_DIR = ROOT / "sources"
CONFIG_FILE = ROOT / "config.yaml"

import yaml
import requests
import pdfplumber
import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

INDEX_DIR = ROOT / "index"
_chroma_collection = None  # lazy-loaded


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None and (INDEX_DIR / "manifest.json").exists():
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        ef = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
        try:
            _chroma_collection = client.get_collection(
                name="fonti_classiche", embedding_function=ef
            )
        except Exception:
            _chroma_collection = None
    return _chroma_collection

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8",
}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


# ── Grounding tools (called by the agent) ─────────────────────────────────────

def _gz_search(query: str, max_results: int, timeout: int) -> list[dict]:
    slug = re.sub(r"\s+", "-", query.strip())
    url = f"https://www.giallozafferano.it/ricerca-ricette/{slug}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except Exception as exc:
        return [{"errore": str(exc)}]
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for card in soup.select(".gz-card")[:max_results]:
        a = card.find("a", href=True)
        if not a:
            continue
        title_el = card.select_one(".gz-title")
        title = title_el.get_text(strip=True) if title_el else (a.get("title") or "")
        results.append({"titolo": title, "url": a["href"], "fonte": "GialloZafferano"})
    return results


def _sp_search(query: str, max_results: int, timeout: int) -> list[dict]:
    slug = re.sub(r"\s+", "-", query.strip())
    url = f"https://www.salepepe.it/ricerca/{slug}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except Exception as exc:
        return [{"errore": str(exc)}]
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for card in soup.select(".sp-grid-recipes-single")[:max_results]:
        link = next(
            (a for a in card.find_all("a", href=True) if "/ricette/" in a["href"]), None
        )
        if not link:
            continue
        h3 = card.select_one("h3")
        title = h3.get_text(strip=True) if h3 else link.get_text(strip=True)
        results.append({"titolo": title, "url": link["href"], "fonte": "Sale & Pepe"})
    return results


def _gr_search(query: str, max_results: int) -> list[dict]:
    """Gambero Rosso via Google site: search (sito blocca scraper diretti)."""
    q = requests.utils.quote(f"site:gamberorosso.it {query}")
    url = f"https://www.google.com/search?q={q}&num={max_results}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as exc:
        return [{"errore": str(exc)}]
    results = []
    for block in soup.select(".tF2Cxc, .g"):
        title_el = block.select_one("h3")
        link_el = block.select_one("a[href]")
        snippet_el = block.select_one(".VwiC3b, .s3v9rd")
        if not (title_el and link_el):
            continue
        href = link_el.get("href", "")
        if href.startswith("/url?q="):
            href = href[7:].split("&")[0]
        if "gamberorosso.it" not in href:
            continue
        results.append({
            "titolo": title_el.get_text(strip=True),
            "url": href,
            "estratto": snippet_el.get_text(strip=True)[:250] if snippet_el else "",
            "fonte": "Gambero Rosso",
        })
    return results[:max_results]


def _semantic_search(query: str, source_filter: str, max_results: int) -> list[dict] | None:
    """Semantic search via chromadb. Returns None if index not available.

    When chunks carry a recipe_id, fetches all sibling chunks for each matched
    recipe and returns the full recipe text instead of a single fragment.
    """
    col = _get_collection()
    if col is None:
        return None
    where = {"source": source_filter} if source_filter else None
    try:
        res = col.query(
            query_texts=[query],
            n_results=min(max_results * 3, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return None

    # Group by (source, recipe_id); keep best score per recipe.
    seen_recipes: dict[tuple, dict] = {}
    bare_results: list[dict] = []

    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        score = round(1 - dist, 3)
        source = meta.get("source", "")
        recipe_id = meta.get("recipe_id")

        if recipe_id:
            key = (source, recipe_id)
            if key not in seen_recipes or score > seen_recipes[key]["score"]:
                seen_recipes[key] = {"score": score, "source": source,
                                     "recipe_id": recipe_id}
        else:
            entry = {"fonte": source, "estratto": doc, "score": score}
            if meta.get("page"):
                entry["pagina"] = meta["page"]
            else:
                entry["passaggio"] = meta.get("passage", "")
            bare_results.append(entry)

    # Reconstruct full recipe text from all chunks sharing the same recipe_id.
    recipe_results: list[dict] = []
    for (source, recipe_id), info in seen_recipes.items():
        try:
            fetched = col.get(
                where={"$and": [{"source": {"$eq": source}},
                                {"recipe_id": {"$eq": recipe_id}}]},
                include=["documents", "metadatas"],
            )
            pairs = sorted(
                zip(fetched["documents"], fetched["metadatas"]),
                key=lambda x: x[1].get("passage", 0),
            )
            full_text = " ".join(doc for doc, _ in pairs)
            recipe_results.append({
                "fonte": source,
                "estratto": full_text,
                "score": info["score"],
            })
        except Exception:
            pass

    all_results = recipe_results + bare_results
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:max_results]


def _pdf_keyword_search(
    source_path: Path,
    query: str,
    source_name: str,
    context_chars: int,
    max_results: int,
) -> list[dict]:
    """Fallback keyword search (used when semantic index not built)."""
    if not source_path.exists():
        return []
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return []
    results = []
    if source_path.suffix.lower() == ".txt":
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        text_lower = text.lower()
        window, pos, n = context_chars * 2, 0, 0
        while pos < len(text) and len(results) < max_results:
            chunk_lower = text_lower[pos: pos + window]
            if all(t in chunk_lower for t in terms):
                n += 1
                chunk = text[pos: pos + window]
                first = min(chunk_lower.find(t) for t in terms if t in chunk_lower)
                start = max(0, first - context_chars // 3)
                excerpt = chunk[start: start + context_chars].replace("\n", " ").strip()
                results.append({"fonte": source_name, "passaggio": n, "estratto": excerpt})
                pos += window
            else:
                pos += context_chars
    else:
        try:
            with pdfplumber.open(source_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    text_lower = text.lower()
                    if not all(t in text_lower for t in terms):
                        continue
                    first = min(text_lower.find(t) for t in terms if t in text_lower)
                    start = max(0, first - context_chars // 3)
                    excerpt = text[start: start + context_chars].replace("\n", " ").strip()
                    results.append({"fonte": source_name, "pagina": page_num, "estratto": excerpt})
                    if len(results) >= max_results:
                        break
        except Exception:
            return []
    return results


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def run_tool(name: str, params: dict, config: dict) -> str:
    cfg_s = config["search"]
    max_r = cfg_s["max_results_per_source"]
    timeout = cfg_s["timeout_seconds"]
    ctx = cfg_s["pdf_context_chars"]
    srcs = config["sources"]

    if name == "cerca_giallozafferano":
        if not srcs["giallozafferano"]["enabled"]:
            return "Fonte disabilitata."
        results = _gz_search(params["query"], max_r, timeout)
        return json.dumps(results, ensure_ascii=False, indent=2)

    if name == "cerca_salepepe":
        if not srcs["salepepe"]["enabled"]:
            return "Fonte disabilitata."
        results = _sp_search(params["query"], max_r, timeout)
        return json.dumps(results, ensure_ascii=False, indent=2)

    if name == "cerca_gamberorosso":
        if not srcs["gamberorosso"]["enabled"]:
            return "Fonte disabilitata."
        results = _gr_search(params["query"], max_r)
        return json.dumps(results, ensure_ascii=False, indent=2)

    if name == "cerca_fonti_classiche":
        query = params["query"]
        alias = params.get("fonte", "").lower()

        # Resolve short alias (e.g. "apicio") to the full source name stored in the index
        full_source = ""
        if alias:
            for pdf_cfg in srcs["pdfs"]:
                if alias in pdf_cfg["name"].lower():
                    full_source = pdf_cfg["name"]
                    break

        # try semantic search first
        sem = _semantic_search(query, full_source, max_r)
        if sem is not None:
            if not sem:
                return "Nessun risultato trovato nelle fonti classiche."
            return json.dumps(sem, ensure_ascii=False, indent=2)

        # fallback: keyword search per ogni PDF
        all_results = []
        for pdf_cfg in srcs["pdfs"]:
            if not pdf_cfg.get("enabled", True):
                continue
            pdf_name: str = pdf_cfg["name"]
            if alias and alias not in pdf_name.lower():
                continue
            path = SOURCES_DIR / pdf_cfg["file"]
            hits = _pdf_keyword_search(path, query, pdf_name, ctx, max_r)
            all_results.extend(hits)
        if not all_results:
            return "Nessun risultato trovato nelle fonti classiche."
        return json.dumps(all_results, ensure_ascii=False, indent=2)

    return f"Tool sconosciuto: {name}"


# ── Tool definitions for Claude ───────────────────────────────────────────────

TOOLS = [
    {
        "name": "cerca_giallozafferano",
        "description": (
            "Cerca ricette su GialloZafferano.it. "
            "Restituisce titolo e URL delle ricette trovate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Termine di ricerca, es. 'carbonara'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cerca_salepepe",
        "description": (
            "Cerca ricette su Sale&Pepe.it. "
            "Restituisce titolo e URL delle ricette trovate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Termine di ricerca"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cerca_gamberorosso",
        "description": (
            "Cerca ricette su GamberoRosso.it (tramite Google, il sito blocca gli scraper). "
            "Restituisce titolo, URL e estratto."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Termine di ricerca"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cerca_fonti_classiche",
        "description": (
            "Cerca nelle fonti PDF classiche scaricate: "
            "Apicio (De re coquinaria), Artusi (La Scienza in Cucina), "
            "Ada Boni (Il Talismano della Felicità). "
            "Restituisce estratti di testo rilevanti con numero di pagina/passaggio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Termine di ricerca"},
                "fonte": {
                    "type": "string",
                    "description": (
                        "Filtra per fonte specifica (opzionale). "
                        "Valori: 'apicio', 'artusi', 'ada boni'. "
                        "Lascia vuoto per cercare in tutte."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_message: str, config: dict, history: list | None = None) -> str:
    """Synchronous agent run. Returns final text."""
    full_text = ""
    for event in stream_agent(user_message, config, history):
        if event["type"] == "text":
            full_text += event["content"]
    return full_text


def stream_agent(user_message: str, config: dict, history: list | None = None):
    """Generator that yields SSE-style dicts: {type, content}.

    Event types:
      {"type": "tool_start", "content": "cerca_giallozafferano"}
      {"type": "text",       "content": "...streaming text..."}
      {"type": "sources",    "content": [{titolo, url, fonte}, ...]}
      {"type": "done"}
    """
    client = anthropic.Anthropic()
    system = config["agent"]["system"]
    messages = list(history or []) + [{"role": "user", "content": user_message}]
    sources_seen: list[dict] = []

    while True:
        # stream the model response
        text_buf = ""
        response_content = []

        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield {"type": "tool_start", "content": block.name}

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_buf += delta.text
                        yield {"type": "text", "content": delta.text}

            final = stream.get_final_message()
            response_content = final.content
            stop_reason = final.stop_reason

        if stop_reason == "end_turn":
            break

        if stop_reason != "tool_use":
            break

        # execute tool calls
        tool_results = []
        for block in response_content:
            if block.type != "tool_use":
                continue
            raw = run_tool(block.name, block.input, config)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": raw,
            })
            # collect sources from tool output
            try:
                items = json.loads(raw) if isinstance(raw, str) else []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and ("url" in item or "fonte" in item):
                            sources_seen.append(item)
            except (json.JSONDecodeError, TypeError):
                pass

        messages.append({"role": "assistant", "content": response_content})
        messages.append({"role": "user", "content": tool_results})

    if sources_seen:
        yield {"type": "sources", "content": sources_seen}
    yield {"type": "done"}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()

    # single-shot mode: argument passed on command line
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(run_agent(query, config))
        return

    # interactive REPL
    print("Agente culinario — scrivi la tua richiesta (Ctrl+C o 'esci' per uscire)")
    print("Fonti: GialloZafferano · Sale&Pepe · GamberoRosso · Apicio · Artusi · Ada Boni\n")

    while True:
        try:
            user_input = input("Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nArrivederci!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("esci", "exit", "quit"):
            print("Arrivederci!")
            break

        print("\nAgente: ", end="", flush=True)
        answer = run_agent(user_input, config)
        # wrap long lines for readability
        for line in answer.splitlines():
            print(textwrap.fill(line, width=100) if len(line) > 100 else line)
        print()


if __name__ == "__main__":
    main()
