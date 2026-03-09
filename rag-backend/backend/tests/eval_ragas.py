import argparse
import json
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


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workspace", default="eval_ws")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    chat_model, embeddings_model = initialize_models()
    rag_graph = RAGGraph(llm=chat_model, embedding_model=embeddings_model, enable_checkpointer=False, workspace=args.workspace)

    eval_rows = []
    for i, s in enumerate(samples):
        q = s.get("question") or s.get("query") or ""
        gts = s.get("ground_truths") or s.get("answers") or []
        if not isinstance(gts, list):
            gts = [str(gts)]
        context = RAGContext(user_id="eval_user", session_id=f"eval_{i}", retrieval_mode=RetrievalMode.VECTOR_ONLY)
        state = rag_graph.invoke(build_input(q), context)
        answer = ""
        if isinstance(state, dict):
            answer = state.get("final_answer") or ""
            if not answer:
                msgs = state.get("messages") or []
                if msgs:
                    last_msg = msgs[-1]
                    answer = getattr(last_msg, "content", str(last_msg))
            retrieved = state.get("retrieved_docs") or []
            contexts = []
            for d in retrieved:
                c = getattr(d, "page_content", None)
                if c is None and isinstance(d, dict):
                    c = d.get("page_content")
                if c:
                    contexts.append(c)
        else:
            answer = str(state)
            contexts = []
        reference = gts[0] if gts else ""
        eval_rows.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
        })

    ds = Dataset.from_list(eval_rows)
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
    except Exception:
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
    print(json.dumps(scores, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
