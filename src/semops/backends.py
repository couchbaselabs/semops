"""Model providers (embeddings + LLM oracle). Orthogonal to the engine mode.

`ModelClient` is the seam. `OpenAICompatClient` talks to any OpenAI-compatible
endpoint (OpenAI, vLLM, Ollama, many gateways). `CapellaAIClient` is a thin
subclass pointing at Capella AI Services' OpenAI-compatible surface. `FakeModelClient`
is a deterministic in-process oracle used by the test suite and offline demos so
the whole cascade is exercisable with zero network / zero keys.

All clients report a rough per-call cost so the Budget guard can do its job.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from typing import Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ModelClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def judge(self, predicate: str, text: str) -> bool: ...
    # cost accounting
    spent_usd: float
    calls: int


class _CostMixin:
    def __init__(self) -> None:
        self.spent_usd = 0.0
        self.calls = 0

    def _charge(self, usd: float) -> None:
        self.spent_usd += usd
        self.calls += 1


class OpenAICompatClient(_CostMixin):
    """Zero-dependency client (stdlib urllib) for OpenAI-compatible servers."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        chat_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        timeout: float = 60.0,
        embed_cost_per_call: float = 0.00002,
        chat_cost_per_call: float = 0.0005,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout
        self.embed_cost_per_call = embed_cost_per_call
        self.chat_cost_per_call = chat_cost_per_call
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _post(self, path: str, payload: dict, _attempt: int = 0) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # retry transient rate-limit / server errors with backoff
            if e.code in (429, 500, 503) and _attempt < self.max_retries:
                ra = e.headers.get("Retry-After") if e.headers else None
                delay = (float(ra) if ra and str(ra).replace(".", "", 1).isdigit()
                         else min(self.backoff_base * (2 ** _attempt), 30.0))
                time.sleep(delay)
                return self._post(path, payload, _attempt + 1)
            raise

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            out = self._post("/embeddings", {"model": self.embed_model, "input": list(texts)})
            self._charge(self.embed_cost_per_call)
            return [d["embedding"] for d in out["data"]]
        except urllib.error.HTTPError:
            # some OpenAI-compat servers (e.g. Gemini) reject batched input;
            # fall back to one request per text.
            vecs = []
            for t in texts:
                out = self._post("/embeddings", {"model": self.embed_model, "input": t})
                self._charge(self.embed_cost_per_call)
                vecs.append(out["data"][0]["embedding"])
            return vecs

    def judge(self, predicate: str, text: str) -> bool:
        prompt = (
            "You are a strict boolean classifier. Answer with exactly one word: "
            "true or false.\n\n"
            f"Question: {predicate}\n\nTEXT:\n{text}\n\nAnswer:"
        )
        out = self._post(
            "/chat/completions",
            {
                "model": self.chat_model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        self._charge(self.chat_cost_per_call)
        content = out["choices"][0]["message"]["content"]
        return _parse_bool(content)

    def match_block(self, predicate, query, candidates, max_tokens=256):
        """Block-join adjudication: one call returns all matching candidate indices.
        Returns (indices, complete) — complete=False signals output overflow."""
        prompt = build_block_prompt(predicate, query, candidates)
        out = self._post("/chat/completions", {
            "model": self.chat_model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]})
        self._charge(self.chat_cost_per_call)
        return parse_block(out["choices"][0]["message"]["content"], len(candidates))

    def generate(self, prompt, max_tokens=64):
        out = self._post("/chat/completions", {
            "model": self.chat_model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]})
        self._charge(self.chat_cost_per_call)
        return out["choices"][0]["message"]["content"].strip()


class CapellaAIClient(OpenAICompatClient):
    """Capella AI Services exposes an OpenAI-compatible endpoint; point base_url at it.

    Keeps embeddings + LLM in the Couchbase ecosystem (one bill, one security
    domain) while reusing the exact OpenAI-compatible wire format.
    """

    def __init__(self, base_url: str, api_key: str = "", **kw):
        super().__init__(base_url=base_url, api_key=api_key, **kw)


def _stable_hash(s: str) -> int:
    """Deterministic across processes (Python's built-in hash() is salted)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


class LocalHashingEmbedder:
    """Zero-dependency lexical embedder: hashed bag-of-words -> fixed-dim vector.

    A *lexical* proxy (cosine ~ shared-word overlap), not neural — so it gives
    real but weaker signal than a trained embedding model. It exists so a
    Claude-only setup (Anthropic has no embeddings API) works end-to-end. For a
    stronger proxy, pass an OpenAI/Voyage embeddings client as the embedder.
    """

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.embed_model = f"local-hash-{dims}"
        self.spent_usd = 0.0
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dims
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[_stable_hash(w) % self.dims] += 1.0
            out.append(v)
        return out


class AnthropicClient(_CostMixin):
    """Claude as the LLM oracle (Messages API). Cheap/fast models like Haiku 4.5
    are ideal for the boolean judge.

    Anthropic has no embeddings endpoint, so embed() delegates to `embedder`
    (default: LocalHashingEmbedder). Pass an OpenAICompatClient/Voyage client as
    `embedder` for a neural proxy.
    """

    def __init__(
        self,
        api_key: str,
        chat_model: str = "claude-haiku-4-5-20251001",
        embedder=None,
        base_url: str = "https://api.anthropic.com/v1",
        version: str = "2023-06-01",
        timeout: float = 60.0,
        chat_cost_per_call: float = 0.0003,
        max_tokens: int = 5,
    ):
        super().__init__()
        self.api_key = api_key
        self.chat_model = chat_model
        self._embedder = embedder or LocalHashingEmbedder()
        self.embed_model = getattr(self._embedder, "embed_model", "external")
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.timeout = timeout
        self.chat_cost_per_call = chat_cost_per_call
        self.max_tokens = max_tokens

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def judge(self, predicate: str, text: str) -> bool:
        payload = {
            "model": self.chat_model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "system": "You are a strict boolean classifier. Reply with exactly one "
                      "word: true or false.",
            "messages": [{"role": "user",
                          "content": f"Question: {predicate}\n\nTEXT:\n{text}\n\nAnswer:"}],
        }
        req = urllib.request.Request(
            self.base_url + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.version,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        self._charge(self.chat_cost_per_call)
        text_out = "".join(p.get("text", "") for p in out.get("content", [])
                           if p.get("type") == "text")
        return _parse_bool(text_out)

    def match_block(self, predicate, query, candidates):
        prompt = build_block_prompt(predicate, query, candidates)
        payload = {"model": self.chat_model, "max_tokens": 256, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            self.base_url + "/messages", data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": self.version}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        self._charge(self.chat_cost_per_call)
        text_out = "".join(p.get("text", "") for p in out.get("content", [])
                           if p.get("type") == "text")
        return parse_block(text_out, len(candidates))

    def generate(self, prompt, max_tokens=64):
        payload = {"model": self.chat_model, "max_tokens": max_tokens, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            self.base_url + "/messages", data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": self.version}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        self._charge(self.chat_cost_per_call)
        return "".join(p.get("text", "") for p in out.get("content", [])
                       if p.get("type") == "text").strip()


class FakeModelClient(_CostMixin):
    """Deterministic oracle for tests/offline demos.

    - embed(text): a small deterministic vector whose direction encodes which of
      `topics` the text mentions (so cosine similarity to a topic vector is a
      meaningful proxy), plus a stable per-text jitter to create a realistic
      ESCALATE band.
    - judge(predicate, text): TRUE iff every keyword parsed from `predicate`
      (quoted words, or words after 'about') appears in the text. This is the
      ground-truth oracle the cascade is measured against.
    """

    def __init__(self, dims: int = 16, block_limit: int = 10 ** 9):
        super().__init__()
        self.dims = dims
        self.block_limit = block_limit  # simulate block-join overflow when a batch exceeds this

    def _hash_unit(self, s: str) -> float:
        h = 0
        for ch in s:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return (h % 1000) / 1000.0  # in [0,1)

    def embed(self, texts):
        self._charge(0.0)
        vecs = []
        for t in texts:
            tl = t.lower()
            v = [0.0] * self.dims
            # deterministic content signal: each word bumps a coordinate
            for w in re.findall(r"[a-z0-9]+", tl):
                v[(sum(ord(c) for c in w)) % self.dims] += 1.0
            # stable jitter so identical-topic docs aren't collinear (creates a middle band)
            j = self._hash_unit(t)
            v[0] += 0.3 * j
            vecs.append(v)
        return vecs

    def judge(self, predicate: str, text: str) -> bool:
        self._charge(0.0)
        return _predicate_true(predicate, text)

    def match_block(self, predicate, query, candidates):
        self._charge(0.0)
        got = candidates[:self.block_limit]  # simulate output overflow past block_limit
        idxs = [i for i, c in enumerate(got)
                if _predicate_true(predicate, query + " " + c)]
        return idxs, len(candidates) <= self.block_limit

    def generate(self, prompt, max_tokens=64):
        self._charge(0.0)
        # deterministic stand-in: return the most frequent salient token in the prompt
        from collections import Counter
        words = [w for w in re.findall(r"[a-z0-9]+", prompt.lower()) if len(w) > 2]
        return Counter(words).most_common(1)[0][0] if words else ""


# --- parsing helpers -------------------------------------------------------

def build_block_prompt(predicate: str, query: str, candidates: Sequence[str]) -> str:
    """Block-join adjudication prompt (Trummer Fig. 2, one left row vs many rights):
    ask the LLM to return ALL candidate numbers matching the query, then 'Finished'
    so truncation (overflow) is detectable."""
    lst = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        f"Given a QUERY item and a numbered list of CANDIDATE items, output the "
        f"numbers of ALL candidates for which this is true: {predicate} "
        f"(make sure to catch all!). Separate numbers with ';'. "
        f"Write \"Finished\" after the last number.\n\n"
        f"QUERY:\n{query}\n\nCANDIDATES:\n{lst}\n\nMatching numbers:"
    )


def parse_block(text: str, n: int) -> tuple[list[int], bool]:
    """Parse matching 0-based indices and whether output completed ('Finished')."""
    complete = "finished" in text.lower()
    head = re.split(r"finished", text, flags=re.IGNORECASE)[0]
    idxs = []
    for m in re.findall(r"\d+", head):
        j = int(m) - 1
        if 0 <= j < n:
            idxs.append(j)
    return idxs, complete


_TRUE_RE = re.compile(r"\btrue\b", re.IGNORECASE)
_FALSE_RE = re.compile(r"\bfalse\b", re.IGNORECASE)


def _parse_bool(content: str) -> bool:
    c = content.strip()
    if _TRUE_RE.search(c) and not _FALSE_RE.search(c):
        return True
    if _FALSE_RE.search(c) and not _TRUE_RE.search(c):
        return False
    # ambiguous: fall back to leading token
    return c[:4].lower() == "true"


def keywords_of(predicate: str) -> list[str]:
    """Extract the salient keywords a fake oracle checks for."""
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", predicate)
    words = [a or b for a, b in quoted]
    if words:
        toks = []
        for w in words:
            toks += re.findall(r"[a-z0-9]+", w.lower())
        return toks
    m = re.search(r"about (.+)$", predicate.lower())
    tail = m.group(1) if m else predicate.lower()
    stop = {"is", "this", "the", "a", "an", "about", "does", "it", "of", "to", "?"}
    return [w for w in re.findall(r"[a-z0-9]+", tail) if w not in stop]


def _predicate_true(predicate: str, text: str) -> bool:
    kws = keywords_of(predicate)
    tl = text.lower()
    return all(kw in tl for kw in kws) if kws else False
