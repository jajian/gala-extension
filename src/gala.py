"""
GALA: Graph-Augmented LLM Agentic Workflow for RCA
====================================================

Setup:
    1. Place this file at:  RCAEval/e2e/gala.py
    2. Add to RCAEval/e2e/__init__.py:
           from .gala import gala
    3. Add "gala" to the --method choices in main.py

Usage:
    # Statistical only (no API key needed):
    python main.py --method gala --dataset re2-ob

    # With LLM agentic reasoning:
    export OPENAI_API_KEY="sk-..."
    python main.py --method gala --dataset re2-ob

Extra requirements:
    pip install openai
"""

import json
import os
import re
import logging
import time as time_mod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# ── RCAEval internals ─────────────────────────────────────────────────────────
from RCAEval.graph_construction.granger import granger
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.io.time_series import preprocess
from RCAEval.e2e import rca

log = logging.getLogger("GALA")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_service_metric(col: str):
    """'cartservice_cpu' -> ('cartservice', 'cpu')"""
    parts = col.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (col, "")


def _fuzzy_svc_match(a: str, b: str) -> bool:
    """Check if two service name strings likely refer to the same service."""
    a = a.lower().replace("-", "").replace("_", "")
    b = b.lower().replace("-", "").replace("_", "")
    return a in b or b in a


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — logs.csv and traces.csv from dataset directory
# ═══════════════════════════════════════════════════════════════════════════════

def _find_col(df_columns, candidates):
    """Find the first column in df_columns whose lowercase matches a candidate."""
    for c in df_columns:
        if c.lower() in candidates:
            return c
    return None


def _read_logs(dataset_dir):
    """Read logs.csv if present. Returns {service: [entries]}."""
    if dataset_dir is None:
        return {}
    p = Path(dataset_dir) / "logs.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return {}

    svc_col = _find_col(df.columns, {"service", "service_name", "cmdb_id", "servicename"})
    msg_col = _find_col(df.columns, {"message", "content", "msg", "log"})
    sev_col = _find_col(df.columns, {"severity", "level", "loglevel"})
    ts_col = _find_col(df.columns, {"timestamp", "time", "ts"})

    if svc_col is None or msg_col is None:
        return {}

    logs_by_svc = defaultdict(list)
    for _, row in df.iterrows():
        logs_by_svc[str(row[svc_col]).strip()].append({
            "timestamp": row.get(ts_col, 0) if ts_col else 0,
            "severity": str(row[sev_col]).upper() if sev_col else "INFO",
            "message": str(row[msg_col]),
        })
    return dict(logs_by_svc)


def _read_traces(dataset_dir):
    """Read traces.csv if present. Returns list of span dicts."""
    if dataset_dir is None:
        return []
    p = Path(dataset_dir) / "traces.csv"
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return []

    sid_col = None
    for c in df.columns:
        if "span_id" in c.lower() and "parent" not in c.lower():
            sid_col = c
            break
    tid_col = _find_col(df.columns, {"trace_id"})
    pid_col = None
    for c in df.columns:
        if "parent" in c.lower():
            pid_col = c
            break
    svc_col = _find_col(df.columns, {"cmdb_id", "service", "service_name"})
    op_col = None
    for c in df.columns:
        if "operation" in c.lower():
            op_col = c
            break
    dur_col = None
    for c in df.columns:
        if "duration" in c.lower():
            dur_col = c
            break
    ts_col = _find_col(df.columns, {"timestamp", "time", "start_time"})
    sc_col = None
    for c in df.columns:
        if "status" in c.lower():
            sc_col = c
            break

    if sid_col is None or tid_col is None:
        return []

    spans = []
    for _, row in df.iterrows():
        spans.append({
            "span_id": str(row.get(sid_col, "")),
            "trace_id": str(row.get(tid_col, "")),
            "parent_span_id": str(row.get(pid_col, "")) if pid_col else "",
            "service": str(row.get(svc_col, "")) if svc_col else "",
            "operation": str(row.get(op_col, "")) if op_col else "",
            "duration": float(row.get(dur_col, 0)) if dur_col else 0.0,
            "timestamp": float(row.get(ts_col, 0)) if ts_col else 0.0,
            "status_code": str(row.get(sc_col, "0")) if sc_col else "0",
        })
    return spans


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE I-A: Metrics-based ranking (reuses RCAEval's granger + page_rank)
# ═══════════════════════════════════════════════════════════════════════════════

def _granger_pr_rank(data: pd.DataFrame) -> list[str]:
    """
    Build Granger causal graph and PageRank it.
    Reuses RCAEval.graph_construction.granger and RCAEval.graph_heads.page_rank.
    """
    node_names = data.columns.to_list()
    adj = granger(data)

    # Fallback if no causal edges found
    if adj.sum().sum() == 0:
        return node_names

    ranks = page_rank(adj, node_names=node_names)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranks]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE I-B: TWIST (Trace-based Weighted Impact Scoring & Thresholding)
# ═══════════════════════════════════════════════════════════════════════════════

def _twist_rank(spans, metric_cols, w=(0.3, 0.3, 0.2, 0.2), z_thresh=2.0):
    """
    TWIST scoring from the GALA paper.
    Computes four per-service scores from trace spans:
      c1 = self-anomaly (fraction of a service's spans flagged anomalous)
      c2 = trace impact  (fraction of anomalous traces containing this service)
      c3 = blast radius  (avg downstream fan-out, normalized)
      c4 = delay severity (max latency excess, normalized)
    Returns metric_cols re-ordered by composite score.
    """
    if not spans:
        return []

    # ── Span-level anomaly detection via dynamic thresholding ──
    dur_by_svc = defaultdict(list)
    for s in spans:
        dur_by_svc[s["service"]].append(s["duration"])

    anom_span_ids = set()
    for svc, durs in dur_by_svc.items():
        arr = np.array(durs)
        mu, sigma = arr.mean(), arr.std()
        if sigma == 0:
            continue
        for s in spans:
            if s["service"] == svc and (s["duration"] - mu) / sigma > z_thresh:
                anom_span_ids.add(s["span_id"])

    # Also flag non-OK status codes
    for s in spans:
        if str(s.get("status_code", "0")) not in ("0", "OK", "200", ""):
            anom_span_ids.add(s["span_id"])

    # ── Group spans by trace and find anomalous traces ──
    by_trace = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)

    anom_trace_ids = set()
    for tid, sp_list in by_trace.items():
        if any(s["span_id"] in anom_span_ids for s in sp_list):
            anom_trace_ids.add(tid)

    # ── Per-service scoring ──
    svc_spans = defaultdict(list)
    for s in spans:
        svc_spans[s["service"]].append(s)

    # Pre-compute global max excess for c4 normalization
    all_excess = []
    for svc, sp_list in svc_spans.items():
        mu = np.mean([s["duration"] for s in sp_list])
        for s in sp_list:
            if s["span_id"] in anom_span_ids:
                all_excess.append(s["duration"] - mu)
    global_max_excess = max(max(all_excess, default=1.0), 1.0)

    svc_scores = {}
    for svc, sp_list in svc_spans.items():
        # c1: self-anomaly
        c1 = sum(1 for s in sp_list if s["span_id"] in anom_span_ids) / max(len(sp_list), 1)

        # c2: trace impact
        traces_with_svc = {s["trace_id"] for s in sp_list}
        c2 = len(traces_with_svc & anom_trace_ids) / max(len(anom_trace_ids), 1)

        # c3: blast radius (avg children per span)
        fan_outs = []
        for s in sp_list:
            children = [ss for ss in by_trace[s["trace_id"]]
                        if ss.get("parent_span_id") == s["span_id"]]
            fan_outs.append(len(children))
        max_fan = max(max(fan_outs, default=0), 1)
        c3 = np.mean(fan_outs) / max_fan if fan_outs else 0.0

        # c4: delay severity
        mu = np.mean([s["duration"] for s in sp_list])
        excess = [s["duration"] - mu for s in sp_list if s["span_id"] in anom_span_ids]
        c4 = max(excess, default=0) / global_max_excess

        score = float(np.dot(w, np.clip([c1, c2, c3, c4], 0, 1)))
        svc_scores[svc] = score

    # ── Map trace service names → RCAEval metric column names ──
    col_scores = {}
    for col in metric_cols:
        col_svc, _ = _parse_service_metric(col)
        best = 0.0
        for trace_svc, sc in svc_scores.items():
            if _fuzzy_svc_match(col_svc, trace_svc):
                best = max(best, sc)
        col_scores[col] = best

    return [c for c, _ in sorted(col_scores.items(), key=lambda x: -x[1])]


# ═══════════════════════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION
# ═══════════════════════════════════════════════════════════════════════════════

def _rrf(lists, k=60):
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores = defaultdict(float)
    for lst in lists:
        for rank, item in enumerate(lst):
            scores[item] += 1.0 / (k + rank + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE II: Diagnostic Synthesis (build text context for LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_diagnostic_bundle(candidate, data, logs, spans):
    """
    Build a concise text diagnostic bundle for one candidate.
    This is what gets fed to the Deep Dive Agent.
    """
    svc, _ = _parse_service_metric(candidate)
    parts = [f"=== Diagnostic Bundle: {candidate} ==="]

    # ── Metrics summary for columns belonging to this service ──
    svc_cols = [c for c in data.columns if c.startswith(svc)]
    for col in svc_cols[:8]:
        vals = data[col].dropna().values.astype(float)
        if len(vals) > 0:
            parts.append(
                f"  {col}: mean={np.mean(vals):.2f}, max={np.max(vals):.2f}, "
                f"p99={np.percentile(vals, 99):.2f}, std={np.std(vals):.2f}"
            )

    # ── Error-centric log abstraction ──
    svc_logs = []
    for log_svc, entries in logs.items():
        if _fuzzy_svc_match(svc, log_svc):
            svc_logs.extend(entries)
    errors = [e for e in svc_logs
              if e.get("severity", "") in ("ERROR", "CRITICAL", "FATAL", "WARN")]
    # De-duplicate by message prefix
    seen = set()
    deduped = []
    for e in errors:
        key = e.get("message", "")[:120]
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    if deduped:
        parts.append("  Logs (error/warn):")
        for e in deduped[:10]:
            parts.append(f"    [{e.get('severity','')}] {e.get('message','')[:200]}")
    else:
        parts.append("  Logs: no errors found.")

    # ── Service dependency subgraph from traces ──
    sid_to_svc = {s["span_id"]: s["service"] for s in spans}
    G = nx.DiGraph()
    for s in spans:
        G.add_node(s["service"])
        psid = s.get("parent_span_id", "")
        if psid and psid in sid_to_svc:
            parent_svc = sid_to_svc[psid]
            if parent_svc != s["service"]:
                G.add_edge(parent_svc, s["service"])

    target = None
    for n in G.nodes:
        if _fuzzy_svc_match(svc, n):
            target = n
            break
    if target and target in G:
        parts.append(f"  Predecessors: {list(G.predecessors(target))}")
        parts.append(f"  Successors: {list(G.successors(target))}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE III: LLM Agentic Reasoning & Re-ranking
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_chat(messages, model="gpt-4.1-mini", temperature=1.0, max_tokens=4096):
    """Call OpenAI chat completions. Returns raw string."""
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except ImportError:
        log.warning("openai package not installed — skipping LLM call.")
        return ""
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return ""


def _parse_json(raw):
    """Extract JSON from an LLM response that may have markdown fences."""
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


_RERANK_SYSTEM = (
    "You are an experienced DevOps engineer performing root cause analysis on a "
    "microservice system. You receive a ranked list of suspect root causes "
    "(formatted as service_metric) and diagnostic evidence.\n\n"
    "Tasks:\n"
    "1. Re-rank all candidates based on evidence.\n"
    "2. Decide 'Analyze Next' (pick a candidate to investigate) or 'Finish'.\n"
    "   If 'Analyze Next': pick a candidate NOT in the already-analyzed set.\n\n"
    "Justify using: predecessor/successor context, chronological errors, "
    "causal inference signals, trace-ranking alignment.\n\n"
    "Return ONLY valid JSON:\n"
    '{"thought":"<chain-of-thought reasoning>",'
    '"action":"Analyze Next" or "Finish",'
    '"updated_ranking":["svc_metric1","svc_metric2",...],'
    '"next_candidate":"svc_metric" or ""}'
)

_DEEPDIVE_SYSTEM = (
    "You are a microservice reliability expert. Given diagnostic evidence for a "
    "service (metrics, logs, trace dependencies), produce a concise diagnostic "
    "summary: key observations, causal hypotheses, and confidence assessment."
)


def _agentic_rerank(initial_ranking, data, logs, spans, model, max_iter):
    """
    Phase III: iterative Re-ranking Agent + Deep Dive Agent loop.
    Returns the refined ranking as a list of strings.
    """
    current = list(initial_ranking)
    analyzed = set()
    last_summary = "(no deep dives yet)"

    for i in range(max_iter):
        # ── Re-ranking Agent ──
        user_msg = (
            f"Current ranking: {json.dumps(current[:15])}\n"
            f"Already analyzed: {json.dumps(list(analyzed))}\n"
            f"Latest deep-dive summary:\n{last_summary[:2000]}\n"
        )
        result = _parse_json(
            _llm_chat(
                [{"role": "system", "content": _RERANK_SYSTEM},
                 {"role": "user", "content": user_msg}],
                model=model,
            )
        )

        action = result.get("action", "Finish")
        new_rank = result.get("updated_ranking", [])

        # Merge LLM ranking with existing (preserve any candidates LLM missed)
        if isinstance(new_rank, list) and new_rank:
            rest = [c for c in current if c not in new_rank]
            current = new_rank + rest

        if action == "Finish" or not result:
            break

        # ── Deep Dive Agent ──
        next_cand = result.get("next_candidate", "")
        if not next_cand or next_cand in analyzed:
            for c in current:
                if c not in analyzed:
                    next_cand = c
                    break
        if not next_cand:
            break

        analyzed.add(next_cand)
        bundle_text = _build_diagnostic_bundle(next_cand, data, logs, spans)
        last_summary = _llm_chat(
            [{"role": "system", "content": _DEEPDIVE_SYSTEM},
             {"role": "user", "content": bundle_text}],
            model=model,
            max_tokens=2048,
        )

    return current


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — the @rca-decorated e2e function
# ═══════════════════════════════════════════════════════════════════════════════

@rca
def gala(data, inject_time=None, dataset=None, sli=None, anomalies=None, **kwargs):
    """
    GALA: Graph-Augmented LLM Agentic Workflow for RCA.

    Phase I-A : Granger causality + PageRank  (from RCAEval internals)
    Phase I-B : TWIST trace-based scoring
    Phase I   : Reciprocal Rank Fusion of I-A and I-B
    Phase II  : Pod-centric diagnostic synthesis
    Phase III : Iterative LLM agentic re-ranking  (optional)

    kwargs
    ------
    model    : str  – OpenAI model name  (default: "gpt-4.1-mini")
    max_iter : int  – max agentic iterations (default: 6)
    use_llm  : bool – set False to skip LLM   (default: True)
    dk_select_useful : bool – passed to preprocess (default: False)
    """
    model = kwargs.get("model", "gpt-4.1-mini")
    max_iter = kwargs.get("max_iter", 6)
    use_llm = kwargs.get("use_llm", True)

    # ── Preprocess using RCAEval's standard pipeline ──
    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )
    node_names = data.columns.to_list()

    # ── Phase I-A: Granger + PageRank (reusing RCAEval internals) ──
    adj = granger(data)

    if adj.sum().sum() == 0:
        granger_ranked = node_names
    else:
        pr = page_rank(adj, node_names=node_names)
        pr = sorted(pr, key=lambda x: x[1], reverse=True)
        granger_ranked = [x[0] for x in pr]

    # ── Phase I-B: TWIST on traces ──
    spans = _read_traces(dataset)
    twist_ranked = _twist_rank(spans, node_names)

    # ── Merge via Reciprocal Rank Fusion ──
    lists_to_fuse = [granger_ranked]
    if twist_ranked:
        lists_to_fuse.append(twist_ranked)
    merged = _rrf(lists_to_fuse)

    # Ensure every node_name is present in the ranking
    rest = [c for c in node_names if c not in merged]
    merged = merged + rest

    # ── Phase III: Agentic Re-ranking (optional) ──
    logs = _read_logs(dataset)
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            merged = _agentic_rerank(
                merged, data, logs, spans,
                model=model, max_iter=max_iter,
            )
        except Exception as e:
            log.warning(f"LLM agentic loop failed ({e}); using statistical ranking.")

    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": merged,
    }