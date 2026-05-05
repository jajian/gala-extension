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

    # With OpenAI agentic reasoning:
    export OPENAI_API_KEY="sk-..."
    python main.py --method gala --dataset re2-ob

    # With Gemini agentic reasoning:
    export GEMINI_API_KEY="your-gemini-key"
    python main.py --method gala --dataset re2-ob --model gemini-2.5-flash

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
from urllib import error, request

import numpy as np
import pandas as pd
import networkx as nx

# ── RCAEval internals ─────────────────────────────────────────────────────────
from RCAEval.graph_construction.granger import granger
from RCAEval.graph_heads.page_rank import page_rank, page_rank_preprocess
from RCAEval.io.time_series import preprocess
from RCAEval.e2e import rca
from RCAEval.e2e.baro import baro as baro_ranker

log = logging.getLogger("GALA")

# ── Ensure GALA logger outputs if no handlers configured ──
if not log.handlers:
    import sys
    # Stream handler (stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[GALA] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    
    # File handler (for complete output unaffected by progress bars)
    try:
        log_file = Path.cwd() / "gala_debug.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        log.addHandler(file_handler)
        print(f"[GALA] Logging to file: {log_file}", file=sys.stderr, flush=True)
    except Exception as e:
        pass  # Silently skip file logging if it fails
        
log.setLevel(logging.DEBUG)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _load_local_env():
    """Load a nearby .env file without overriding existing environment variables."""
    candidates = []
    seen = set()
    for base in [Path.cwd(), Path(__file__).resolve().parent]:
        for parent in [base, *base.parents]:
            env_path = parent / ".env"
            if env_path in seen:
                continue
            seen.add(env_path)
            candidates.append(env_path)

    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            log.debug(f"Loading .env from {env_path}")
            loaded_keys = []
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded_keys.append(key)
            if loaded_keys:
                log.info(f"Loaded {len(loaded_keys)} env vars from {env_path}: {', '.join(loaded_keys)}")
            return env_path
        except Exception as e:
            log.warning(f"Failed to load .env from {env_path}: {e}")
            return None
    return None


_load_local_env()
log.debug("[GALA] Module initialized and .env loaded")


def _log_flush(level_name="INFO"):
    """Flush all handlers to ensure output is not buffered."""
    for handler in log.handlers:
        if hasattr(handler, "flush"):
            try:
                handler.flush()
            except Exception:
                pass


def _default_model():
    """Choose a sensible default model from the available credentials."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini-2.5-flash"
    return "gpt-4.1-mini"


def _resolve_case_dir(dataset, kwargs):
    """Locate the current case directory so logs.csv and traces.csv can be loaded."""
    run_args = kwargs.get("args")
    data_path = getattr(run_args, "data_path", None) if run_args else None
    if data_path:
        return str(Path(data_path).resolve().parent)

    if dataset and Path(str(dataset)).exists():
        return str(Path(dataset).resolve())

    return dataset


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

def _normalize_col_name(col):
    return str(col).lower().replace("_", "")


def _find_col(df_columns, candidates):
    """Find the first column whose normalized name matches a candidate."""
    normalized_candidates = {_normalize_col_name(c) for c in candidates}
    for c in df_columns:
        if _normalize_col_name(c) in normalized_candidates:
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
        normalized = _normalize_col_name(c)
        if "spanid" in normalized and "parent" not in normalized:
            sid_col = c
            break
    tid_col = _find_col(df.columns, {"trace_id", "traceid"})
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
    ts_col = _find_col(
        df.columns,
        {"timestamp", "start_time", "starttime", "start_time_millis", "starttimemillis"},
    )
    if ts_col is None:
        ts_col = _find_col(df.columns, {"time"})
    sc_col = None
    for c in df.columns:
        if "status" in c.lower():
            sc_col = c
            break

    if sid_col is None or tid_col is None:
        return []

    def _as_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    spans = []
    for _, row in df.iterrows():
        spans.append({
            "span_id": str(row.get(sid_col, "")),
            "trace_id": str(row.get(tid_col, "")),
            "parent_span_id": str(row.get(pid_col, "")) if pid_col else "",
            "service": str(row.get(svc_col, "")) if svc_col else "",
            "operation": str(row.get(op_col, "")) if op_col else "",
            "duration": _as_float(row.get(dur_col, 0)) if dur_col else 0.0,
            "timestamp": _as_float(row.get(ts_col, 0)) if ts_col else 0.0,
            "status_code": str(row.get(sc_col, "0")) if sc_col else "0",
        })
    return spans


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE I-A: Metrics-based ranking (BARO by default; Granger+PageRank retained)
# ═══════════════════════════════════════════════════════════════════════════════

def _baro_rank(raw_data, inject_time=None, dataset=None, sli=None, anomalies=None, **kwargs) -> list[str]:
    """Run BARO and return its metric ranking."""
    result = baro_ranker(
        raw_data,
        inject_time=inject_time,
        dataset=dataset,
        sli=sli,
        anomalies=anomalies,
        **kwargs,
    )
    return list(result.get("ranks", []))


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

    ranks = _page_rank_with_fallback(adj, node_names)
    return [x[0] for x in ranks]


def _page_rank_with_fallback(adj, node_names):
    """Run RCAEval PageRank, falling back when the installed sknetwork API differs."""
    try:
        ranks = page_rank(adj, node_names=node_names)
        return sorted(ranks, key=lambda x: x[1], reverse=True)
    except Exception:
        log.exception("[GALA] RCAEval page_rank failed; using NetworkX PageRank fallback.")

    pr_input = page_rank_preprocess(adj)
    graph = nx.from_numpy_array(pr_input, create_using=nx.DiGraph)
    scores = nx.pagerank(graph) if graph.number_of_edges() else {i: 0.0 for i in range(len(node_names))}
    ranks = [(node_names[i], scores.get(i, 0.0)) for i in range(len(node_names))]
    return sorted(ranks, key=lambda x: x[1], reverse=True)


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
        status_code = str(s.get("status_code", "0")).strip().upper()
        if status_code not in ("0", "0.0", "OK", "200", "200.0", ""):
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

def _llm_chat(messages, model="gemini-2.5-flash-lite", temperature=1.0, max_tokens=4096):
    """Call either Gemini or OpenAI chat completions. Returns raw string."""
    if model.startswith("gemini"):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            log.warning("Gemini model requested but GEMINI_API_KEY/GOOGLE_API_KEY is not set.")
            return ""
        log.info(f"Using Gemini model: {model}")

        system_parts = []
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if not text:
                continue
            if role == "system":
                system_parts.append({"text": text})
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["system_instruction"] = {"parts": system_parts}

        payload_json = json.dumps(payload)
        log.info(f"[Gemini Request] URL: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
        log.info(f"[Gemini Request] Payload ({len(payload_json)} bytes):\n{payload_json[:2000]}{'...(truncated)' if len(payload_json) > 2000 else ''}")

        req = request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=payload_json.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                response_data = resp.read().decode("utf-8")
                body = json.loads(response_data)
            log.info(f"[Gemini Response] Status: {resp.status}\n{response_data[:2000]}{'...(truncated)' if len(response_data) > 2000 else ''}")
            candidates = body.get("candidates", [])
            if not candidates:
                log.warning("[Gemini Response] No candidates in response")
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            result = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            log.info(f"[Gemini Response] Extracted text ({len(result)} chars):\n{result[:1000]}{'...(truncated)' if len(result) > 1000 else ''}")
            return result
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            log.warning(f"Gemini call failed: HTTP {e.code} {detail[:500]}")
            return ""
        except Exception as e:
            log.warning(f"Gemini call failed: {e}")
            return ""

    try:
        from openai import OpenAI
        client = OpenAI()
        log.info(f"Using OpenAI model: {model}")
        messages_json = json.dumps(messages)
        log.info(f"[OpenAI Request] Model: {model}")
        log.info(f"[OpenAI Request] Payload ({len(messages_json)} bytes):\n{messages_json[:2000]}{'...(truncated)' if len(messages_json) > 2000 else ''}")
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        response_text = resp.choices[0].message.content
        log.info(f"[OpenAI Response] ({len(response_text)} chars):\n{response_text[:1000]}{'...(truncated)' if len(response_text) > 1000 else ''}")
        return response_text
    except ImportError:
        log.warning("openai package not installed — skipping LLM call.")
        return ""
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return ""


def _has_llm_credentials(model):
    if model.startswith("gemini"):
        has_creds = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        log.debug(f"Checking Gemini credentials: GEMINI_API_KEY={bool(os.environ.get('GEMINI_API_KEY'))}, GOOGLE_API_KEY={bool(os.environ.get('GOOGLE_API_KEY'))} → {has_creds}")
        return has_creds
    has_creds = bool(os.environ.get("OPENAI_API_KEY"))
    log.debug(f"Checking OpenAI credentials: OPENAI_API_KEY={has_creds} → {has_creds}")
    return has_creds


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

    Phase I-A : BARO metrics ranking
    Phase I-B : TWIST trace-based scoring
    Phase I   : Reciprocal Rank Fusion of I-A and I-B
    Phase II  : Pod-centric diagnostic synthesis
    Phase III : Iterative LLM agentic re-ranking  (optional)

    kwargs
    ------
    model    : str  – LLM model name; use "gemini-*" for Gemini (default: auto-detect)
    max_iter : int  – max agentic iterations (default: 6)
    use_llm  : bool – set False to skip LLM   (default: True)
    dk_select_useful : bool – passed to preprocess (default: False)
    """
    model = kwargs.get("model") or _default_model()
    max_iter = kwargs.get("max_iter", 6)
    use_llm = kwargs.get("use_llm", True)
    case_dir = _resolve_case_dir(dataset, kwargs)

    log.info(f"[GALA] Starting with model='{model}', use_llm={use_llm}, max_iter={max_iter}")
    log.info(f"[GALA] Case directory: {case_dir}")
    _log_flush()

    raw_data = data

    # ── Preprocess using RCAEval's standard pipeline ──
    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )
    node_names = data.columns.to_list()
    log.debug(f"[GALA] Preprocessed data has {len(node_names)} metrics: {node_names[:10]}{'...' if len(node_names) > 10 else ''}")
    _log_flush()

    # ── Phase I-A: BARO metric ranking ──
    log.info("[GALA] Phase I-A: Running BARO metric ranking...")
    _log_flush()
    try:
        baro_ranked = _baro_rank(
            raw_data,
            inject_time=inject_time,
            dataset=dataset,
            sli=sli,
            anomalies=anomalies,
            **kwargs,
        )
        baro_ranked = [metric for metric in baro_ranked if metric in node_names]
        if not baro_ranked:
            baro_ranked = node_names
            log.debug("[GALA] Phase I-A: BARO returned no overlapping metrics, using all metrics")
        log.info(f"[GALA] Phase I-A complete. Top 5 from BARO: {baro_ranked[:5]}")
    except Exception:
        log.exception("[GALA] Phase I-A BARO failed; falling back to the preprocessed metric order.")
        baro_ranked = node_names
    _log_flush()

    # Previous Phase I-A implementation: Granger + PageRank.
    # Kept here so it can be restored later if needed.
    #
    # log.info("[GALA] Phase I-A: Running Granger causality + PageRank...")
    # _log_flush()
    # try:
    #     adj = granger(data)
    #
    #     if adj.sum().sum() == 0:
    #         granger_ranked = node_names
    #         log.debug("[GALA] Phase I-A: No causal edges found, using all metrics")
    #     else:
    #         pr = _page_rank_with_fallback(adj, node_names)
    #         granger_ranked = [x[0] for x in pr]
    #     log.info(f"[GALA] Phase I-A complete. Top 5 from Granger+PageRank: {granger_ranked[:5]}")
    # except Exception:
    #     log.exception("[GALA] Phase I-A failed; falling back to the preprocessed metric order.")
    #     adj = np.zeros((len(node_names), len(node_names)))
    #     granger_ranked = node_names
    # _log_flush()

    # ── Phase I-B: TWIST on traces ──
    log.info("[GALA] Phase I-B: Running TWIST trace scoring...")
    _log_flush()
    spans = _read_traces(case_dir)
    log.debug(f"[GALA] Phase I-B: Loaded {len(spans)} span records from traces")
    twist_ranked = _twist_rank(spans, node_names)
    if twist_ranked:
        log.info(f"[GALA] Phase I-B complete. Top 5 from TWIST: {twist_ranked[:5]}")
    else:
        log.info("[GALA] Phase I-B: No trace-based ranking (no spans or traces)")
    _log_flush()

    # ── Merge via Reciprocal Rank Fusion ──
    lists_to_fuse = [baro_ranked]
    if twist_ranked:
        lists_to_fuse.append(twist_ranked)
    merged = _rrf(lists_to_fuse)

    # Ensure every node_name is present in the ranking
    rest = [c for c in node_names if c not in merged]
    merged = merged + rest
    log.info(f"[GALA] Phase I (statistical): Top 5 candidates: {merged[:5]}")
    _log_flush()

    # ── Phase III: Agentic Re-ranking (optional) ──
    logs = _read_logs(case_dir)
    if use_llm and _has_llm_credentials(model):
        log.info(f"[GALA] Phase III: Starting agentic re-ranking with model '{model}' (max {max_iter} iterations)")
        _log_flush()
        try:
            merged = _agentic_rerank(
                merged, data, logs, spans,
                model=model, max_iter=max_iter,
            )
            log.info(f"[GALA] Phase III complete. Final top 5: {merged[:5]}")
            _log_flush()
        except Exception as e:
            log.warning(f"[GALA] LLM agentic loop failed ({e}); using statistical ranking.")
            _log_flush()
    elif use_llm:
        log.warning(f"[GALA] Skipping Phase III: no credentials available for model '{model}'.")
        _log_flush()
    else:
        log.info("[GALA] Phase III skipped (use_llm=False)")
        _log_flush()

    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": merged,
    }
