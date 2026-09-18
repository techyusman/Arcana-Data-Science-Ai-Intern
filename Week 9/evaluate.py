"""Topic 5 (Evaluation of RAG outputs): score retrieval quality and answer
quality separately, per the intern's own notes ("RAG Evaluation" — retrieval
evaluation vs. generation evaluation).

Retrieval evaluation: did the expected source file get retrieved (hit rate)?
Generation evaluation: LLM-as-judge scores faithfulness (is the answer
grounded in the retrieved context?) and relevance (does it address the
question?) on a 1-5 scale.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_chain import RAGChatbot

load_dotenv(Path(__file__).parent / ".env")

EVAL_SET = Path(__file__).parent / "eval" / "eval_questions.json"
RESULTS_CSV = Path(__file__).parent / "eval" / "results.csv"

JUDGE_PROMPT = """You are grading a RAG chatbot's answer. Score two things from 1 (worst) to 5 (best):
- faithfulness: is every claim in the answer supported by the given context (no invented facts)?
- relevance: does the answer actually address the question?

Context:
{context}

Question: {question}
Answer: {answer}

Respond with JSON only, no other text: {{"faithfulness": <1-5>, "relevance": <1-5>, "reason": "<one sentence>"}}"""


def retrieval_hit(sources: list[str], expected_source: str | None) -> bool:
    if expected_source is None:
        return len(sources) == 0  # question has no answer in the KB: correct behaviour is to retrieve nothing
    return any(expected_source in source for source in sources)


def judge(llm: ChatGroq, question: str, answer: str, context: str) -> dict:
    response = llm.invoke(JUDGE_PROMPT.format(context=context, question=question, answer=answer))
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "reason": "judge did not return valid JSON"}


def run_evaluation() -> None:
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    chatbot = RAGChatbot()
    judge_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.environ.get("GROQ_API_KEY"))

    rows = []
    for case in cases:
        answer, chunks = chatbot.ask(case["question"])
        sources = [c.source for c in chunks]
        hit = retrieval_hit(sources, case["expected_source"])
        context = "\n\n".join(c.text for c in chunks)
        scores = (
            judge(judge_llm, case["question"], answer, context)
            if chunks
            else {"faithfulness": 5, "relevance": 5, "reason": "correctly refused with no context"}
        )
        rows.append(
            {
                "question": case["question"],
                "expected_source": case["expected_source"] or "(none - should refuse)",
                "retrieved_sources": ", ".join(sources) or "(none)",
                "retrieval_hit": hit,
                "answer": answer,
                "faithfulness": scores.get("faithfulness"),
                "relevance": scores.get("relevance"),
                "judge_reason": scores.get("reason"),
            }
        )

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hit_rate = sum(r["retrieval_hit"] for r in rows) / len(rows)
    valid_faith = [r["faithfulness"] for r in rows if isinstance(r["faithfulness"], (int, float))]
    valid_rel = [r["relevance"] for r in rows if isinstance(r["relevance"], (int, float))]

    print(f"Retrieval hit rate: {hit_rate:.0%} ({sum(r['retrieval_hit'] for r in rows)}/{len(rows)})")
    if valid_faith:
        print(f"Avg faithfulness:   {sum(valid_faith) / len(valid_faith):.2f} / 5")
    if valid_rel:
        print(f"Avg relevance:      {sum(valid_rel) / len(valid_rel):.2f} / 5")
    print(f"Full results -> {RESULTS_CSV}")


if __name__ == "__main__":
    run_evaluation()
