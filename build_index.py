#!/usr/bin/env python3
"""
Costruisce (o ricostruisce) l'indice semantico delle fonti PDF/TXT.

Uso:
    python build_index.py              # build incrementale: reindicizza solo le fonti cambiate
    python build_index.py --rebuild    # cancella tutto e reindicizza da zero
    python build_index.py --status     # mostra lo stato dell'indice senza modificarlo
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path

import yaml
import pdfplumber
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).parent
SOURCES_DIR = ROOT / "sources"
INDEX_DIR = ROOT / "index"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
CONFIG_FILE = ROOT / "config.yaml"

CHUNK_SIZE = 600      # caratteri per chunk
CHUNK_OVERLAP = 100   # sovrapposizione tra chunk contigui
COLLECTION_NAME = "fonti_classiche"


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Spezza il testo in chunk con overlap."""
    text = " ".join(text.split())  # normalizza spazi/newline
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 60:  # scarta frammenti troppo corti
            chunks.append(chunk)
        start += size - overlap
    return chunks


def extract_text_from_source(path: Path) -> list[tuple[str, dict]]:
    """Ritorna lista di (testo_chunk, metadata)."""
    results = []
    if path.suffix.lower() == ".txt":
        try:
            full = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  ! Errore lettura {path.name}: {e}")
            return []
        for i, chunk in enumerate(chunk_text(full)):
            results.append((chunk, {"passage": i + 1, "page": 0}))
    else:
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    for i, chunk in enumerate(chunk_text(text)):
                        results.append((chunk, {"page": page_num, "passage": i + 1}))
        except Exception as e:
            print(f"  ! Errore lettura PDF {path.name}: {e}")
    return results


# ── Manifest (change detection) ───────────────────────────────────────────────

def file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


# ── Index management ──────────────────────────────────────────────────────────

def get_collection():
    INDEX_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    ef = DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def delete_source_from_index(collection, source_name: str) -> int:
    """Rimuove tutti i chunk di una fonte dall'indice."""
    results = collection.get(where={"source": source_name})
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def index_source(collection, path: Path, source_name: str) -> int:
    """Indicizza una fonte. Ritorna il numero di chunk aggiunti."""
    chunks = extract_text_from_source(path)
    if not chunks:
        return 0

    batch_docs, batch_meta, batch_ids = [], [], []
    for i, (text, meta) in enumerate(chunks):
        doc_id = f"{source_name}::{i}"
        batch_docs.append(text)
        batch_meta.append({"source": source_name, "file": path.name, **meta})
        batch_ids.append(doc_id)

    # upsert in batches of 100
    for start in range(0, len(batch_docs), 100):
        collection.upsert(
            documents=batch_docs[start:start + 100],
            metadatas=batch_meta[start:start + 100],
            ids=batch_ids[start:start + 100],
        )
    return len(batch_docs)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_pdf_configs() -> list[dict]:
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    return [p for p in config["sources"]["pdfs"] if p.get("enabled", True)]


def cmd_status(collection):
    manifest = load_manifest()
    pdf_configs = load_pdf_configs()
    total = collection.count()
    print(f"Indice semantico — {total} chunk totali\n")
    for pdf_cfg in pdf_configs:
        name = pdf_cfg["name"]
        path = SOURCES_DIR / pdf_cfg["file"]
        exists = path.exists()
        indexed = name in manifest
        changed = exists and indexed and manifest[name]["hash"] != file_hash(path)
        state = ("✓ indicizzata" if indexed and not changed
                 else "⚠ modificata — esegui build" if changed
                 else "✗ non indicizzata" if exists
                 else "✗ file mancante")
        print(f"  [{state}] {name}")
        if indexed:
            print(f"          {manifest[name]['chunks']} chunk — {path.name}")


def cmd_build(rebuild: bool = False):
    collection = get_collection()
    manifest = load_manifest()
    pdf_configs = load_pdf_configs()

    if rebuild:
        print("Ricostruzione completa dell'indice…")
        for pdf_cfg in pdf_configs:
            name = pdf_cfg["name"]
            removed = delete_source_from_index(collection, name)
            if removed:
                print(f"  Rimossi {removed} chunk: {name}")
        manifest = {}

    changed = 0
    for pdf_cfg in pdf_configs:
        name = pdf_cfg["name"]
        path = SOURCES_DIR / pdf_cfg["file"]

        if not path.exists():
            print(f"  ⚠ File mancante, salto: {path.name}")
            continue

        current_hash = file_hash(path)
        if not rebuild and name in manifest and manifest[name]["hash"] == current_hash:
            print(f"  — Invariata, salto: {name}")
            continue

        print(f"  ↻ Indicizzazione: {name}…", end=" ", flush=True)
        delete_source_from_index(collection, name)
        n = index_source(collection, path, name)
        manifest[name] = {"hash": current_hash, "file": path.name, "chunks": n}
        save_manifest(manifest)
        print(f"{n} chunk")
        changed += 1

    if changed == 0:
        print("Indice già aggiornato — nessuna modifica.")
    else:
        total = collection.count()
        print(f"\nIndice aggiornato: {total} chunk totali.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true",
                        help="Cancella e ricostruisce l'indice da zero")
    parser.add_argument("--status", action="store_true",
                        help="Mostra lo stato dell'indice senza modificarlo")
    args = parser.parse_args()

    if args.status:
        cmd_status(get_collection())
    else:
        cmd_build(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
