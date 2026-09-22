"""
HippoRAG over Gray's Anatomy (dmedhi/medical-textbooks), fully local with Ollama.

Requirements:
    pip install datasets
    pip install -U git+https://github.com/OSU-NLP-Group/HippoRAG.git   # PyPI's 2.0.0a4 is too old
    openie_list_shim.py must sit next to this file.

Ollama:
    ollama pull nomic-embed-text
    OLLAMA_CONTEXT_LENGTH=16384 ollama serve

The dataset has one `train` split (~16k rows) with columns `text` and `book`.
"""

import random
from collections import Counter

from datasets import load_dataset

import openie_list_shim  # patches OpenIE parsing; must be imported before HippoRAG is created
from hipporag import HippoRAG

# ---------------------------------------------------------------- config ----
BOOK_MATCH = "gray"                      # case-insensitive substring match on `book`
MIN_CHARS = 200                          # drop tiny/empty fragments
SAVE_DIR = "outputs/grays_anatomy_ollama"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:7b-instruct"   # use whichever model worked for you
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Test queries, each paired with a keyword that should appear in a passage that answers it.
QUERIES = [
    ("Which nerve innervates the deltoid muscle?", "deltoid"),
    ("What structures pass through the carpal tunnel?", "carpal tunnel"),
    ("What is the blood supply of the scaphoid bone?", "scaphoid"),
]

# Indexing costs roughly 20 s per passage with a local 7-9B model (NER + triples), so a
# full run over all matching rows takes hours. The default builds a small test set:
# passages that mention the query keywords + random distractors, so queries are answerable.
FULL_RUN = False
PER_KEYWORD = 10        # passages kept per keyword
N_DISTRACTORS = 40      # random other passages

# ------------------------------------------------------------- load data ----
ds = load_dataset("dmedhi/medical-textbooks", split="train")

print("Books in dataset:")
for name, n in Counter(ds["book"]).most_common():
    print(f"  {n:>6}  {name}")

grays = ds.filter(lambda r: BOOK_MATCH in r["book"].lower())
print(f"\nMatched {len(grays)} rows for '{BOOK_MATCH}'")

docs, seen = [], set()
for text in grays["text"]:
    text = (text or "").strip()
    if len(text) < MIN_CHARS or text in seen:
        continue
    seen.add(text)
    docs.append(text)
print(f"{len(docs)} unique passages after cleaning")

if FULL_RUN:
    selected = docs
else:
    selected = []
    for _, kw in QUERIES:
        hits = [d for d in docs if kw in d.lower()]
        print(f"  '{kw}': {len(hits)} matching passages (keeping {min(len(hits), PER_KEYWORD)})")
        selected += hits[:PER_KEYWORD]
    chosen = set(selected)
    others = [d for d in docs if d not in chosen]
    selected += random.Random(0).sample(others, min(N_DISTRACTORS, len(others)))
    selected = list(dict.fromkeys(selected))     # de-duplicate, keep order
print(f"Indexing {len(selected)} passages")

# ---------------------------------------------------------------- index -----
hipporag = HippoRAG(
    save_dir=SAVE_DIR,
    llm_model_name=LLM_MODEL,
    llm_base_url=OLLAMA_BASE_URL,
    embedding_model_name=EMBEDDING_MODEL,
    embedding_provider="openai",
    embedding_base_url=OLLAMA_BASE_URL,
)
openie_list_shim.apply(hipporag)   # tolerate local-model JSON quirks in NER / triple extraction

# Builds OpenIE triples, knowledge graph and embeddings. Re-running with the same SAVE_DIR
# only processes passages that haven't been indexed yet.
hipporag.index(docs=selected)

# ------------------------------------------------------ retrieve + answer ---
questions = [q for q, _ in QUERIES]

retrieval_results = hipporag.retrieve(queries=questions, num_to_retrieve=5)
for sol, (_, kw) in zip(retrieval_results, QUERIES):
    print(f"\nQ: {sol.question}")
    for doc, score in zip(sol.docs, sol.doc_scores):
        hit = "[hit]" if kw in doc.lower() else "     "
        print(f"  {hit} [{score:.3f}] {doc[:430].replace(chr(10), ' ')}...")

qa_out = hipporag.rag_qa(queries=questions)
for sol in qa_out[0]:
    print(f"\nQ: {sol.question}\nA: {sol.answer}")
