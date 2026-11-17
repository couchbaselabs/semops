"""Resumable cascades: a small, reproducible experiment.

Question: in a semops sem_filter cascade, what happens if the cheap proxy is the
oracle's own early layers (a probe on Qwen3-8B's residual stream at layer L), and
escalated rows *resume* the forward pass from layer L instead of starting over?

Three subcommands, run in order:

  extract   one forward pass per row (manual layer loop), saving the last-token
            residual after every layer, the model's yes/no margin, ground truth,
            plus baseline features (bge-small row embeddings, Qwen3-0.6B states).
            Also checks that the manual loop reproduces the HF forward exactly and
            that "resume from layer L" gives the same answer as one full pass.
  timing    wall-clock per row on this machine at a fixed batch size: layers
            1..L, resume L+1..N, full pass, and the two baselines.
  analyze   runs semops.cascade.calibrate (the real three-band cascade) with
            out-of-fold proxy scores, over many random calibration samples, and
            reports escalation rate, guarantee coverage and cost.

The oracle is the full model's own single-token yes/no answer (no decode), so
every precision/recall number is relative to what Qwen3-8B itself would answer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = Path(os.environ.get("RC_CACHE", HERE / "cache"))
RESULTS = Path(os.environ.get("RC_RESULTS", HERE / "results"))
BIG = "Qwen/Qwen3-8B"
SMALL = "Qwen/Qwen3-0.6B"
BGE = "BAAI/bge-small-en-v1.5"
N_ROWS = 2000
BATCH = 16
LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36]

PREDICATES = {
    "rt_positive": ("rt", 1, "Does this movie review express a positive opinion of the movie?"),
    "ag_business": ("ag", 2, "Is this news article mainly about business or the economy?"),
    "ag_scitech": ("ag", 3, "Is this news article mainly about science or technology?"),
}


# --------------------------------------------------------------------------- data

def load_rows(source: str):
    from datasets import load_dataset, concatenate_datasets
    if source == "rt":
        ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        ds = concatenate_datasets([ds["validation"], ds["test"]])
    elif source == "ag":
        ds = load_dataset("fancyzhx/ag_news")["test"]
    else:
        raise ValueError(source)
    ds = ds.shuffle(seed=0).select(range(N_ROWS))
    return list(ds["text"]), np.array(ds["label"])


def prompt_text(tok, question: str, text: str) -> str:
    # instruction first, row last: the shared prefix is the whole instruction
    content = f"{question} Answer with only yes or no.\n\nText: {text}"
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


# --------------------------------------------------------------------------- model

def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def synchronizer():
    import torch
    return {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device().type, lambda: None)


def load_lm(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device()).eval()
    return tok, model


def answer_ids(tok):
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes"]})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No"]})
    return yes, no


def prep(model, input_ids, attn):
    """Everything a decoder layer needs, built the same way Qwen3Model.forward builds it."""
    from transformers.masking_utils import create_causal_mask
    emb = model.model.embed_tokens(input_ids)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    mask = create_causal_mask(config=model.config, inputs_embeds=emb, attention_mask=attn,
                              past_key_values=None, position_ids=pos)
    rope = model.model.rotary_emb(emb, pos)
    return emb, mask, rope, pos


def run_layers(model, x, mask, rope, pos, start, end):
    for i in range(start, end):
        x = model.model.layers[i](x, attention_mask=mask, position_embeddings=rope, position_ids=pos)
    return x


def head_margin(model, x_last, yes, no):
    """log P(yes) - log P(no) from the final layer, reading only the answer rows of
    the output head (the Quail trick: a single-token answer needs no decode)."""
    import torch
    h = model.model.norm(x_last)
    logits = model.lm_head(h).float()
    m = torch.logsumexp(logits[:, yes], -1) - torch.logsumexp(logits[:, no], -1)
    top = logits.argmax(-1)
    return m, top


def batches(n, size):
    for i in range(0, n, size):
        yield slice(i, min(n, i + size))


# --------------------------------------------------------------------------- extract

def extract(args):
    import torch
    CACHE.mkdir(parents=True, exist_ok=True)
    tok, model = load_lm(BIG)
    yes, no = answer_ids(tok)
    allowed = set(yes) | set(no)
    n_layers = model.config.num_hidden_layers
    report = {"model": BIG, "n_layers": n_layers, "hidden": model.config.hidden_size,
              "yes_ids": yes, "no_ids": no, "predicates": {}}

    for pname, (src, pos_label, question) in PREDICATES.items():
        out = CACHE / f"{pname}.npz"
        if out.exists() and not args.force:
            print("skip", pname, flush=True)
            continue
        texts, labels = load_rows(src)
        prompts = [prompt_text(tok, question, t) for t in texts]
        enc_len = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
        order = np.argsort(enc_len)  # length-sorted batches waste less padding
        feats = np.zeros((N_ROWS, n_layers, model.config.hidden_size), dtype=np.float32)
        margin = np.zeros(N_ROWS, dtype=np.float32)
        top_ok = np.zeros(N_ROWS, dtype=bool)
        t0 = time.time()
        with torch.no_grad():
            for sl in batches(N_ROWS, BATCH):
                idx = order[sl]
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device())
                x, mask, rope, pos = prep(model, enc.input_ids, enc.attention_mask)
                buf = torch.empty(len(idx), n_layers, model.config.hidden_size, dtype=torch.float32, device=x.device)
                for i in range(n_layers):
                    x = model.model.layers[i](x, attention_mask=mask, position_embeddings=rope, position_ids=pos)
                    buf[:, i] = x[:, -1, :].float()  # float32: residual outliers can overflow float16
                feats[idx] = buf.cpu().numpy()
                m, top = head_margin(model, x[:, -1, :], yes, no)
                margin[idx] = m.cpu().numpy()
                top_ok[idx] = np.array([t in allowed for t in top.cpu().tolist()])
        secs = time.time() - t0
        gt = labels == pos_label
        oracle = margin > 0
        np.savez(out, feats=feats, margin=margin, gt=gt, top_ok=top_ok)
        acc = float((oracle == gt).mean())
        report["predicates"][pname] = {
            "rows": N_ROWS, "base_rate_gt": float(gt.mean()), "base_rate_model": float(oracle.mean()),
            "model_accuracy_vs_gt": acc, "top1_is_yes_or_no": float(top_ok.mean()),
            "mean_prompt_tokens": float(np.mean(enc_len)), "extract_seconds": secs,
        }
        print(pname, report["predicates"][pname], flush=True)

    # exactness checks on one batch of the first predicate
    pname, (src, pos_label, question) = next(iter(PREDICATES.items()))
    texts, _ = load_rows(src)
    prompts = [prompt_text(tok, question, t) for t in texts[:BATCH]]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device())
    with torch.no_grad():
        x, mask, rope, pos = prep(model, enc.input_ids, enc.attention_mask)
        full = run_layers(model, x, mask, rope, pos, 0, n_layers)
        m_full, _ = head_margin(model, full[:, -1, :], yes, no)
        ref = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos).logits[:, -1].float()
        m_ref = torch.logsumexp(ref[:, yes], -1) - torch.logsumexp(ref[:, no], -1)
        checks = {}
        for L in [8, 12, 20]:
            mid = run_layers(model, x, mask, rope, pos, 0, L)
            saved = mid.clone()          # what stage 1 would hand to stage 2
            res = run_layers(model, saved, mask, rope, pos, L, n_layers)
            m_res, _ = head_margin(model, res[:, -1, :], yes, no)
            checks[f"resume_from_{L}_max_abs_diff"] = float((m_res - m_full).abs().max())
            checks[f"resume_from_{L}_same_answer"] = bool(((m_res > 0) == (m_full > 0)).all())
        checks["manual_vs_hf_forward_max_abs_diff"] = float((m_full - m_ref).abs().max())
        checks["manual_vs_hf_forward_same_answer"] = bool(((m_full > 0) == (m_ref > 0)).all())
    report["exactness"] = checks
    print("exactness", checks, flush=True)

    # baseline features: bge-small on the row text alone (the current semops proxy),
    # and Qwen3-0.6B's last residual on the same prompt (a separate small model)
    from transformers import AutoModel, AutoTokenizer
    btok = AutoTokenizer.from_pretrained(BGE)
    bge = AutoModel.from_pretrained(BGE).to(device()).eval()
    stok, small = load_lm(SMALL)
    syes, sno = answer_ids(stok)
    for src in sorted({v[0] for v in PREDICATES.values()}):
        out = CACHE / f"bge_{src}.npy"
        if not out.exists() or args.force:
            texts, _ = load_rows(src)
            embs = []
            with torch.no_grad():
                for sl in batches(N_ROWS, 64):
                    e = btok(texts[sl], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device())
                    cls = bge(**e).last_hidden_state[:, 0]
                    embs.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
            np.save(out, np.concatenate(embs))
    for pname, (src, _, question) in PREDICATES.items():
        out = CACHE / f"small_{pname}.npz"
        if out.exists() and not args.force:
            continue
        texts, _ = load_rows(src)
        prompts = [prompt_text(stok, question, t) for t in texts]
        feats = np.zeros((N_ROWS, small.config.hidden_size), dtype=np.float32)
        margin = np.zeros(N_ROWS, dtype=np.float32)
        with torch.no_grad():
            for sl in batches(N_ROWS, BATCH):
                enc = stok(prompts[sl], return_tensors="pt", padding=True, add_special_tokens=False).to(device())
                x, mask, rope, pos = prep(small, enc.input_ids, enc.attention_mask)
                x = run_layers(small, x, mask, rope, pos, 0, small.config.num_hidden_layers)
                feats[sl] = x[:, -1, :].float().cpu().numpy()
                m, _ = head_margin(small, x[:, -1, :], syes, sno)
                margin[sl] = m.cpu().numpy()
        np.savez(out, feats=feats, margin=margin)
    RESULTS.mkdir(parents=True, exist_ok=True)
    prev = RESULTS / "extract.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        old["predicates"].update(report["predicates"])
        report["predicates"] = old["predicates"]
    prev.write_text(json.dumps(report, indent=2))


# --------------------------------------------------------------------------- timing

def timing(args):
    """Seconds per row at batch BATCH on this machine, median of repeats."""
    import torch
    sync = synchronizer()
    gpu = torch.cuda.get_device_name(0) if device().type == "cuda" else str(device())
    res = {"device": gpu, "batch": BATCH, "predicates": {}}
    tok, model = load_lm(BIG)
    yes, no = answer_ids(tok)
    n_layers = model.config.num_hidden_layers
    n_time = 256
    for pname, (src, _, question) in PREDICATES.items():
        texts, _ = load_rows(src)
        prompts = [prompt_text(tok, question, t) for t in texts[:n_time]]
        encs = [tok(prompts[sl], return_tensors="pt", padding=True, add_special_tokens=False).to(device())
                for sl in batches(n_time, BATCH)]

        def stage1(L):
            with torch.no_grad():
                sync(); t = time.time()
                for enc in encs:
                    x, mask, rope, pos = prep(model, enc.input_ids, enc.attention_mask)
                    run_layers(model, x, mask, rope, pos, 0, L)
                sync()
                return (time.time() - t) / n_time

        def full():
            with torch.no_grad():
                sync(); t = time.time()
                for enc in encs:
                    x, mask, rope, pos = prep(model, enc.input_ids, enc.attention_mask)
                    x = run_layers(model, x, mask, rope, pos, 0, n_layers)
                    head_margin(model, x[:, -1, :], yes, no)
                sync()
                return (time.time() - t) / n_time

        def resume(L):
            with torch.no_grad():
                states = []
                for enc in encs:
                    x, mask, rope, pos = prep(model, enc.input_ids, enc.attention_mask)
                    states.append((run_layers(model, x, mask, rope, pos, 0, L), mask, rope, pos))
                sync(); t = time.time()
                for x, mask, rope, pos in states:
                    x = run_layers(model, x, mask, rope, pos, L, n_layers)
                    head_margin(model, x[:, -1, :], yes, no)
                sync()
                return (time.time() - t) / n_time

        full()  # warm-up
        r = {"full": float(np.median([full() for _ in range(3)]))}
        for L in LAYERS:
            r[f"stage1_{L}"] = float(np.median([stage1(L) for _ in range(3)]))
            if L < n_layers:
                r[f"resume_{L}"] = float(np.median([resume(L) for _ in range(3)]))
        res["predicates"][pname] = r
        print(pname, {k: round(v * 1000, 2) for k, v in r.items()}, "ms/row", flush=True)
    del model
    import gc
    gc.collect()
    if device().type == "cuda":
        torch.cuda.empty_cache()
    elif device().type == "mps":
        torch.mps.empty_cache()

    from transformers import AutoModel, AutoTokenizer
    stok, small = load_lm(SMALL)
    btok = AutoTokenizer.from_pretrained(BGE)
    bge = AutoModel.from_pretrained(BGE).to(device()).eval()
    for pname, (src, _, question) in PREDICATES.items():
        texts, _ = load_rows(src)
        sp = [prompt_text(stok, question, t) for t in texts[:n_time]]
        sencs = [stok(sp[sl], return_tensors="pt", padding=True, add_special_tokens=False).to(device())
                 for sl in batches(n_time, BATCH)]
        bencs = [btok(texts[sl], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device())
                 for sl in batches(n_time, 64)]

        def t_small():
            with torch.no_grad():
                sync(); t = time.time()
                for enc in sencs:
                    x, mask, rope, pos = prep(small, enc.input_ids, enc.attention_mask)
                    run_layers(small, x, mask, rope, pos, 0, small.config.num_hidden_layers)
                sync()
                return (time.time() - t) / n_time

        def t_bge():
            with torch.no_grad():
                sync(); t = time.time()
                for enc in bencs:
                    bge(**enc)
                sync()
                return (time.time() - t) / n_time

        t_small(); t_bge()
        res["predicates"][pname]["small_full"] = float(np.median([t_small() for _ in range(3)]))
        res["predicates"][pname]["bge"] = float(np.median([t_bge() for _ in range(3)]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "timing.json").write_text(json.dumps(res, indent=2))


# --------------------------------------------------------------------------- analyze

def oof_scores(X, y, folds=5, seed=0, C=0.1):
    """Out-of-fold probabilities, same idea as semops.operators._oof_proba."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    oof = np.zeros(len(y))
    k = min(folds, int(y.sum()), int((~y).sum()))
    if k < 2:
        return None
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    final = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, y)
    return oof, final


def run_cascade(X, oracle, sample_idx, rest_idx, recall, precision, delta, seed):
    from semops.cascade import calibrate, NEG_INF, POS_INF
    ys = oracle[sample_idx]
    got = oof_scores(X[sample_idx], ys, seed=seed)
    if got is None:
        th, scores = None, None
    else:
        oof, final = got
        th = calibrate(list(zip(oof.tolist(), ys.tolist())), recall, precision, delta)
        scores = final.predict_proba(X[rest_idx])[:, 1]
    keep = oracle.copy()  # sample rows: oracle-labelled, exact
    if th is None:
        esc = np.ones(len(rest_idx), dtype=bool)
        est_e = 1.0
    else:
        acc = scores >= th.tau_plus
        rej = scores <= th.tau_minus
        esc = ~(acc | rej)
        keep[rest_idx] = acc | (esc & oracle[rest_idx])
        est_e = float(np.mean((oof > th.tau_minus) & (oof < th.tau_plus)))
    p = (keep & oracle).sum() / max(1, keep.sum())
    r = (keep & oracle).sum() / max(1, oracle.sum())
    return {"e": float(esc.mean()), "est_e": est_e, "precision": float(p), "recall": float(r), "keep": keep}


_G: dict = {}  # per-worker state, filled by _init_worker


def _init_worker(paths, oracle, gt, n, T, a):
    # spawned workers memory-map the feature matrices instead of inheriting them:
    # fork after numpy/BLAS start-up can deadlock on macOS
    _G.update(methods={k: np.load(p, mmap_mode="r") for k, p in paths.items()},
              oracle=oracle, gt=gt, n=n, T=T, args=a)


def _seed_job(seed):
    g = _G
    methods, oracle, gt, n, T, a = g["methods"], g["oracle"], g["gt"], g["n"], g["T"], g["args"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(oracle))
    s_idx, r_idx = perm[:n], perm[n:]
    out = {}
    for name, X in methods.items():
        r = run_cascade(X, oracle, s_idx, r_idx, a["recall"], a["precision"], a["delta"], seed)
        k = r.pop("keep")
        tp = (k & gt).sum()
        r["f1_gt"] = float(2 * tp / max(1, k.sum() + gt.sum()))
        out[name] = r
    # choose the exit layer from the calibration sample only (est_e), by expected resume cost
    best = min(LAYERS, key=lambda L: T[f"stage1_{L}"] + out[f"L{L}"]["est_e"] * T.get(f"resume_{L}", 0.0))
    out["chosen"] = dict(out[f"L{best}"], layer=best)
    return out


def analyze(args):
    import multiprocessing as mp
    tm = json.loads((RESULTS / "timing.json").read_text())
    workers = getattr(args, "workers", None) or os.cpu_count()
    out = {"settings": {"recall": args.recall, "precision": args.precision, "delta": args.delta,
                        "seeds": args.seeds, "sample_sizes": args.samples, "proxy": "standardized logistic regression, C=0.1, 5-fold out-of-fold calibration",
                        "oracle": "Qwen3-8B single-token yes/no answer", "timing_device": tm.get("device")},
           "predicates": {}}
    for pname, (src, _, _) in PREDICATES.items():
        d = np.load(CACHE / f"{pname}.npz")
        feats, margin, gt = d["feats"], d["margin"], d["gt"]
        oracle = margin > 0
        bge = np.load(CACHE / f"bge_{src}.npy")
        small = np.load(CACHE / f"small_{pname}.npz")["feats"]
        T = tm["predicates"][pname]
        t_full = T["full"]
        mdir = CACHE / f"_methods_{pname}"
        mdir.mkdir(parents=True, exist_ok=True)
        paths = {"bge": bge, "qwen3-0.6b": small}
        for L in LAYERS:
            paths[f"L{L}"] = feats[:, L - 1, :]
        for name in list(paths):
            p = mdir / f"{name}.npy"
            np.save(p, np.ascontiguousarray(paths[name], dtype=np.float32))
            paths[name] = str(p)
        del feats, bge, small
        pres = {"model_accuracy_vs_gt": float((oracle == gt).mean()), "base_rate_model": float(oracle.mean()),
                "t_full_ms": t_full * 1000, "by_sample_size": {}}
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[k] = "1"  # one BLAS thread per worker; parallelism comes from the pool
        for n in args.samples:
            a = {"recall": args.recall, "precision": args.precision, "delta": args.delta}
            with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                              initargs=(paths, oracle, gt, n, T, a)) as pool:
                per_seed = pool.map(_seed_job, list(range(args.seeds)))
            rows = {}
            for seed_res in per_seed:
                for name, rr in seed_res.items():
                    rows.setdefault(name, []).append(rr)

            def cost(name, e, layer=None):
                if name == "bge":
                    return T["bge"] + e * t_full
                if name == "qwen3-0.6b":
                    return T["small_full"] + e * t_full
                L = layer
                return {"restart": T[f"stage1_{L}"] + e * t_full,
                        "resume": T[f"stage1_{L}"] + e * T.get(f"resume_{L}", 0.0)}

            summary = {}
            for name, rr in rows.items():
                e = np.array([x["e"] for x in rr])
                prec = np.array([x["precision"] for x in rr])
                rec = np.array([x["recall"] for x in rr])
                met = (prec >= args.precision) & (rec >= args.recall)
                f1_gt = [x["f1_gt"] for x in rr]
                item = {"escalate_mean": float(e.mean()), "escalate_sd": float(e.std()),
                        "precision_vs_model_mean": float(prec.mean()), "recall_vs_model_mean": float(rec.mean()),
                        "both_targets_met_share": float(met.mean()), "f1_vs_ground_truth_mean": float(np.mean(f1_gt))}
                if name in ("bge", "qwen3-0.6b"):
                    c = np.mean([cost(name, x["e"]) for x in rr])
                    item["ms_per_row"] = float(c * 1000)
                    item["speedup_vs_full_pass"] = float(t_full / c)
                else:
                    layers_used = [x.get("layer") or int(name[1:]) for x in rr]
                    cr = np.mean([cost(name, x["e"], L)["restart"] for x, L in zip(rr, layers_used)])
                    cs = np.mean([cost(name, x["e"], L)["resume"] for x, L in zip(rr, layers_used)])
                    item["ms_per_row_restart"] = float(cr * 1000)
                    item["ms_per_row_resume"] = float(cs * 1000)
                    item["speedup_vs_full_pass_restart"] = float(t_full / cr)
                    item["speedup_vs_full_pass_resume"] = float(t_full / cs)
                    if name == "chosen":
                        vals, counts = np.unique(layers_used, return_counts=True)
                        item["layers_chosen"] = {int(v): int(c) for v, c in zip(vals, counts)}
                summary[name] = item
            pres["by_sample_size"][str(n)] = summary
        out["predicates"][pname] = pres
        print(pname, "done", flush=True)
    (RESULTS / "analyze.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({p: {n: {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                              for k, v in s.items() if k in ("bge", "qwen3-0.6b", "chosen", "L12", "L20")}
                          for n, s in r["by_sample_size"].items()} for p, r in out["predicates"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "timing", "analyze"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--samples", type=int, nargs="+", default=[100, 400])
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=None)
    main_args = ap.parse_args()
    {"extract": extract, "timing": timing, "analyze": analyze}[main_args.cmd](main_args)
