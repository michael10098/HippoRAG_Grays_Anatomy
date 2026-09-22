"""
Tolerant adapter between local LLM output and HippoRAG's OpenIE parser.

HippoRAG expects:
    NER step:      {"named_entities": ["a", "b"]}
    Triple step:   {"triples": [["s", "p", "o"], ...]}

Local models often return something close but different: a bare list, a different
key, numbers inside the list, ```json fences, prose around the JSON, or output cut
off mid-list. This shim repairs those cases for the OpenIE steps only (QA and fact
filtering are not touched).

If a response is truly unusable, it is appended to openie_failures.jsonl and an empty
result is used for that passage, so one bad response cannot abort a long indexing run.
Set FALLBACK_TO_EMPTY = False to get the strict behaviour back.

Usage:
    import openie_list_shim                 # before creating HippoRAG
    from hipporag import HippoRAG
    hipporag = HippoRAG(...)
    openie_list_shim.apply(hipporag)        # after creating it, before index()
    hipporag.index(docs=docs)
"""
import json
import re
import threading
import time

from hipporag.information_extraction import openie_openai as oo

FALLBACK_TO_EMPTY = True
FAILURE_LOG = "openie_failures.jsonl"

_state = threading.local()
_lock = threading.Lock()
_fallback_count = 0

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
_STR = r'"((?:[^"\\]|\\.)*)"'
_STR_RE = re.compile(_STR)
_TRIPLE_RE = re.compile(r"\[\s*" + _STR + r"\s*,\s*" + _STR + r"\s*,\s*" + _STR + r"\s*\]")
_KEY_NAMES = {"named_entities", "entities", "triples"}

_cls = next(
    c for c in vars(oo).values()
    if isinstance(c, type) and hasattr(c, "ner") and hasattr(c, "triple_extraction")
)


def _track(step, fn):
    """Remember which OpenIE step (and chunk) is running in this thread."""
    def inner(self, *args, **kwargs):
        _state.step = step
        _state.chunk = args[0] if args else kwargs.get("chunk_key")
        try:
            return fn(self, *args, **kwargs)
        finally:
            _state.step = None
            _state.chunk = None
    return inner


_cls.ner = _track("ner", _cls.ner)
_cls.triple_extraction = _track("triples", _cls.triple_extraction)


# ----------------------------------------------------------------- parsing ---
def _load(text):
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):   # JSON embedded in prose
        i, j = text.find(open_c), text.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                pass
    return None


def _first_list(data, key):
    if isinstance(data, dict):
        if isinstance(data.get(key), list):
            return data[key]
        for value in data.values():        # different key name: take the first list
            if isinstance(value, list):
                return value
        return None
    return data if isinstance(data, list) else None


def _clean_entities(items):
    out = []
    for x in items:
        if isinstance(x, bool) or x is None:
            continue
        if isinstance(x, (str, int, float)):
            s = str(x).strip()
            if s:
                out.append(s)
    return out


def _clean_triples(items):
    out = []
    for t in items:
        if isinstance(t, (list, tuple)) and len(t) == 3:
            parts = [str(p).strip() for p in t
                     if p is not None and not isinstance(p, (list, dict, bool))]
            if len(parts) == 3 and all(parts):
                out.append(parts)
    return out


def _salvage(step, raw):
    """Last resort for broken/truncated JSON: pull quoted strings straight from the text."""
    if step == "ner":
        found = [s.strip() for s in _STR_RE.findall(raw)]
        return [s for s in found if s and s.lower() not in _KEY_NAMES]
    return [[a.strip(), b.strip(), c.strip()] for a, b, c in _TRIPLE_RE.findall(raw)]


def _fail(step, key, raw):
    global _fallback_count
    with _lock:
        _fallback_count += 1
        n = _fallback_count
        try:
            with open(FAILURE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": step,
                    "chunk": getattr(_state, "chunk", None),
                    "raw": raw[:3000],
                }) + "\n")
        except Exception:
            pass
    print(f"[openie_list_shim] unusable {step} response (#{n}); "
          f"{'using empty result' if FALLBACK_TO_EMPTY else 'passing through'}; see {FAILURE_LOG}",
          flush=True)
    return json.dumps({key: []}) if FALLBACK_TO_EMPTY else raw


def _normalize(raw):
    step = getattr(_state, "step", None)
    if step not in ("ner", "triples") or not isinstance(raw, str):
        return raw
    key = "named_entities" if step == "ner" else "triples"
    clean = _clean_entities if step == "ner" else _clean_triples

    items = _first_list(_load(raw), key)
    if items is not None:
        result = clean(items)
        if result or not items:                     # usable, or a genuinely empty list
            return json.dumps({key: result})
        return _fail(step, key, raw)                # a list, but nothing usable in it
    salvaged = _salvage(step, raw)
    if salvaged:
        return json.dumps({key: salvaged})
    return _fail(step, key, raw)


class _TolerantLLM:
    """Proxy: forwards everything to the real LLM, normalizing infer() output."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def infer(self, *args, **kwargs):
        result = self._inner.infer(*args, **kwargs)
        return (_normalize(result[0]), *result[1:])


def apply(hipporag):
    """Give the OpenIE component its own wrapped LLM; the QA LLM stays untouched."""
    openie = hipporag.openie
    if not isinstance(openie.llm_model, _TolerantLLM):
        openie.llm_model = _TolerantLLM(openie.llm_model)
    print("[openie_list_shim] OpenIE LLM wrapped")