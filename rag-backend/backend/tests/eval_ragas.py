import argparse
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from backend.config.models import initialize_models
from backend.agent.graph import RAGGraph
from backend.agent.contexts.raggraph_context import RAGContext
from backend.agent.models.raggraph_models import RetrievalMode


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_input(question: str) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": question}]}


def _safe_divide(numerator: float, denominator: float):
    if denominator == 0:
        return None
    return numerator / denominator


def _percentile(values: List[float], ratio: float):
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = int(round((len(sorted_values) - 1) * ratio))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workspace", default="eval_ws")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--retrieval-mode", default=RetrievalMode.VECTOR_ONLY)
    parser.add_argument("--max-retrieval-docs", type=int, default=3)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--legacy-output", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    total_rows = len(samples)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    chat_model, embeddings_model = initialize_models()
    rag_graph = RAGGraph(llm=chat_model, embedding_model=embeddings_model, enable_checkpointer=False, workspace=args.workspace)

    run_start = time.time()
    eval_rows = []
    items = []
    fusion_stats_rows = []
    for i, s in enumerate(samples):
        q = s.get("question") or s.get("query") or ""
        gts = s.get("ground_truths") or s.get("answers") or []
        if not isinstance(gts, list):
            gts = [str(gts)]
        context = RAGContext(
            user_id="eval_user",
            session_id=f"eval_{i}",
            retrieval_mode=args.retrieval_mode,
            max_retrieval_docs=args.max_retrieval_docs
        )
        started = time.time()
        state = rag_graph.invoke(build_input(q), context)
        latency_ms = int((time.time() - started) * 1000)
        answer = ""
        contexts = []
        retrieval_sources = {}
        fusion_stats = {}
        if isinstance(state, dict):
            answer = state.get("final_answer") or ""
            if not answer:
                msgs = state.get("messages") or []
                if msgs:
                    last_msg = msgs[-1]
                    answer = getattr(last_msg, "content", str(last_msg))
            retrieved = state.get("retrieved_docs") or []
            for d in retrieved:
                c = getattr(d, "page_content", None)
                if c is None and isinstance(d, dict):
                    c = d.get("page_content")
                if c:
                    contexts.append(c)
                metadata = getattr(d, "metadata", {}) or {}
                source_name = str(metadata.get("source") or "unknown")
                retrieval_sources[source_name] = retrieval_sources.get(source_name, 0) + 1
            fusion_stats = state.get("retrieval_fusion_stats") or {}
            if fusion_stats:
                fusion_stats_rows.append(fusion_stats)
        else:
            answer = str(state)
        reference = gts[0] if gts else ""
        eval_rows.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
        })
        items.append({
            "question": q,
            "reference": reference,
            "answer": answer,
            "latency_ms": latency_ms,
            "contexts_count": len(contexts),
            "contexts_total_chars": sum(len(ctx) for ctx in contexts),
            "retrieval_sources": retrieval_sources,
            "retrieval_fusion_stats": fusion_stats
        })

    ds = Dataset.from_list(eval_rows)
    warning = None
    if not args.skip_ragas:
        try:
            llm = LangchainLLMWrapper(chat_model)
            embeddings = LangchainEmbeddingsWrapper(embeddings_model)
            result = evaluate(
                ds,
                metrics=[context_precision, context_recall, answer_relevancy, faithfulness],
                llm=llm,
                embeddings=embeddings,
            )
            df = result.to_pandas()
            non_metric_cols = {"user_input", "response", "retrieved_contexts", "reference"}
            metric_cols = [c for c in df.columns if c not in non_metric_cols]
            scores = {}
            for c in metric_cols:
                try:
                    scores[c] = float(df[c].dropna().mean())
                except Exception:
                    continue
            for idx, row in df.iterrows():
                items[idx]["metrics"] = {col: (float(row[col]) if row[col] == row[col] else None) for col in metric_cols}
        except Exception:
            warning = "RAGAS评测失败，已回退到基础指标"
            def to_words(s: str) -> set[str]:
                if not s:
                    return set()
                tokens = []
                for ch in s.lower():
                    if ch.isspace():
                        continue
                    tokens.append(ch)
                return set(tokens)
            precs = []
            recs = []
            for row in eval_rows:
                ref = row["reference"]
                ctxs = row.get("retrieved_contexts") or []
                ctx = " ".join(ctxs) if ctxs else ""
                aw = to_words(ctx)
                bw = to_words(ref)
                inter = aw & bw
                prec = (len(inter) / len(aw)) if aw else 0.0
                rec = (len(inter) / len(bw)) if bw else 0.0
                precs.append(prec)
                recs.append(rec)
            scores = {
                "context_precision": sum(precs) / len(precs) if precs else 0.0,
                "context_recall": sum(recs) / len(recs) if recs else 0.0,
            }
            for item in items:
                item["metrics"] = {
                    "context_precision": None,
                    "context_recall": None,
                    "answer_relevancy": None,
                    "faithfulness": None
                }
    else:
        warning = "已跳过RAGAS，仅输出基础指标"
        def to_words(s: str) -> set[str]:
            if not s:
                return set()
            tokens = []
            for ch in s.lower():
                if ch.isspace():
                    continue
                tokens.append(ch)
            return set(tokens)
        precs = []
        recs = []
        for row in eval_rows:
            ref = row["reference"]
            ctxs = row.get("retrieved_contexts") or []
            ctx = " ".join(ctxs) if ctxs else ""
            aw = to_words(ctx)
            bw = to_words(ref)
            inter = aw & bw
            prec = (len(inter) / len(aw)) if aw else 0.0
            rec = (len(inter) / len(bw)) if bw else 0.0
            precs.append(prec)
            recs.append(rec)
        scores = {
            "context_precision": sum(precs) / len(precs) if precs else 0.0,
            "context_recall": sum(recs) / len(recs) if recs else 0.0,
        }
        for item in items:
            item["metrics"] = {
                "context_precision": None,
                "context_recall": None,
                "answer_relevancy": None,
                "faithfulness": None
            }
    run_elapsed_ms = int((time.time() - run_start) * 1000)
    latencies = [float(item.get("latency_ms") or 0) for item in items]
    contexts_counts = [int(item.get("contexts_count") or 0) for item in items]
    source_distribution = {}
    for item in items:
        for source_name, source_count in (item.get("retrieval_sources") or {}).items():
            if isinstance(source_count, int):
                source_distribution[source_name] = source_distribution.get(source_name, 0) + source_count
    fusion_summary = {}
    for key in {"vector_docs", "graph_docs", "merged_docs", "rrf_k", "mmr_lambda"}:
        values = [row.get(key) for row in fusion_stats_rows if isinstance(row.get(key), (int, float))]
        if values:
            fusion_summary[key] = round(sum(values) / len(values), 6)
    report = {
        "run": {
            "created_at": time.time(),
            "dataset": {
                "path": args.dataset,
                "total_rows": total_rows,
                "used_rows": len(samples)
            },
            "config": {
                "workspace": args.workspace,
                "retrieval_mode": args.retrieval_mode,
                "max_retrieval_docs": args.max_retrieval_docs,
                "skip_ragas": args.skip_ragas
            },
            "quality_summary": scores,
            "performance_summary": {
                "run_elapsed_ms": run_elapsed_ms,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "p99_latency_ms": _percentile(latencies, 0.99),
                "throughput_qps": round(_safe_divide(len(items), run_elapsed_ms / 1000), 4) if run_elapsed_ms > 0 else None
            },
            "retrieval_summary": {
                "avg_contexts_per_item": round(sum(contexts_counts) / len(contexts_counts), 4) if contexts_counts else None,
                "p95_contexts_per_item": _percentile([float(v) for v in contexts_counts], 0.95),
                "source_distribution": source_distribution,
                "fusion_summary": fusion_summary
            },
            "stability_summary": {
                "total_items": len(items),
                "empty_answer_count": sum(1 for item in items if not (item.get("answer") or "").strip())
            },
            "warning": warning
        },
        "summary": scores,
        "items": items
    }
    output_payload = scores if args.legacy_output else report
    print(json.dumps(output_payload, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
