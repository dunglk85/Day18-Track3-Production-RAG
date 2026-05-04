"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("  ⚠️  OPENAI_API_KEY not found — returning dummy metrics (M4)")
        dummy_per_question = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            dummy_per_question.append(EvalResult(
                question=q, answer=a, contexts=c, ground_truth=gt,
                faithfulness=0.0, answer_relevancy=0.0,
                context_precision=0.0, context_recall=0.0
            ))
        return {
            "faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0,
            "per_question": dummy_per_question
        }

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings)
    )
    df = result.to_pandas()
    
    per_question = []
    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=row.get('question', row.get('user_input', '')),
            answer=row.get('answer', row.get('response', '')),
            contexts=row.get('contexts', row.get('retrieved_contexts', [])),
            ground_truth=row.get('ground_truth', row.get('reference', '')),
            faithfulness=row.get('faithfulness', 0.0),
            answer_relevancy=row.get('answer_relevancy', 0.0),
            context_precision=row.get('context_precision', 0.0),
            context_recall=row.get('context_recall', 0.0)
        ))

    import numpy as np
    
    return {
        "faithfulness": float(np.mean(result["faithfulness"])) if isinstance(result["faithfulness"], (list, np.ndarray)) else float(result["faithfulness"]),
        "answer_relevancy": float(np.mean(result["answer_relevancy"])) if isinstance(result["answer_relevancy"], (list, np.ndarray)) else float(result["answer_relevancy"]),
        "context_precision": float(np.mean(result["context_precision"])) if isinstance(result["context_precision"], (list, np.ndarray)) else float(result["context_precision"]),
        "context_recall": float(np.mean(result["context_recall"])) if isinstance(result["context_recall"], (list, np.ndarray)) else float(result["context_recall"]),
        "per_question": per_question
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []
        
    scored_results = []
    for res in eval_results:
        avg_score = (res.faithfulness + res.answer_relevancy + res.context_precision + res.context_recall) / 4.0
        scored_results.append((avg_score, res))
        
    scored_results.sort(key=lambda x: x[0])
    bottom = scored_results[:bottom_n]
    
    failures = []
    for _, res in bottom:
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        worst_metric = min(metrics, key=metrics.get)
        score = metrics[worst_metric]
        
        diagnosis = "Unknown issue"
        fix = "Investigate chunk content and LLM response"
        
        if worst_metric == "faithfulness" and score < 0.85:
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and score < 0.75:
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            fix = "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and score < 0.80:
            diagnosis = "Answer doesn't match question"
            fix = "Improve prompt template"
            
        failures.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": float(score),
            "diagnosis": diagnosis,
            "suggested_fix": fix
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
