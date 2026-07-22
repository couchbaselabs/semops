"""Shared helpers for the Couchbase demo: REST admin, local neural embedder
(fastembed), and connection constants.

Connection is read from the environment so the examples run against any cluster
(Docker, Capella, self-managed, cluster_run) without editing code. Defaults are
the cluster_run dev ports.

    CB_QUERY_URL   query service   (default http://localhost:9499)
    CB_REST_URL    management REST (default http://localhost:9000)
    CB_USER        (default Administrator)
    CB_PASSWORD    (default asdasd)
    CB_BUCKET      (default default)

For a stock Couchbase Server (Docker / self-managed), the ports are 8093 (query)
and 8091 (REST):

    CB_QUERY_URL=http://localhost:8093 CB_REST_URL=http://localhost:8091 \\
    CB_PASSWORD=yourpass python examples/quickstart.py
"""
import base64
import os
import urllib.error
import urllib.parse
import urllib.request

REST = os.environ.get("CB_REST_URL", "http://localhost:9000")
QUERY = os.environ.get("CB_QUERY_URL", "http://localhost:9499")
USER = os.environ.get("CB_USER", "Administrator")
PW = os.environ.get("CB_PASSWORD", "asdasd")
BUCKET = os.environ.get("CB_BUCKET", "default")
SCOPE, COLL = "_default", "docs"


def _auth():
    return "Basic " + base64.b64encode(f"{USER}:{PW}".encode()).decode()


def rest(method, path, form=None):
    data = urllib.parse.urlencode(form).encode() if form else None
    req = urllib.request.Request(REST + path, data=data, method=method,
                                 headers={"Authorization": _auth()})
    if form:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class FastEmbedEmbedder:
    """Local ONNX neural embedder (default BAAI/bge-small-en-v1.5, 384-d).
    Free, fast, no rate limits — the right tool for bulk-embedding a corpus."""

    def __init__(self, model="BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self._m = TextEmbedding(model_name=model)
        self.embed_model = model
        self.spent_usd = 0.0
        self.calls = 0

    def embed(self, texts):
        return [[float(x) for x in v] for v in self._m.embed(list(texts))]
