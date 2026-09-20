import math
import re
import json
from pathlib import Path
from .facts import Answer

def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

def grade_exact(answer: Answer, accepted_answers: list[str]) -> bool:
    expected = {normalize(item) for item in accepted_answers}
    if "not found" in expected:
        return not answer.found and normalize(answer.answer) == "not found"
    return answer.found and normalize(answer.answer) in expected

def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors have different lengths")
    product = sum(a * b for a, b in zip(left, right))
    a = math.sqrt(sum(x * x for x in left))
    b = math.sqrt(sum(x * x for x in right))
    return product / (a * b) if a and b else 0.0

def embeddings(client, texts, cache_path, model="text-embedding-3-small"):
    path = Path(cache_path)
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = list(dict.fromkeys(text for text in texts if f"{model}:{text}" not in cache))
    calls = 0
    tokens = 0
    for start in range(0, len(missing), 64):
        batch = missing[start:start + 64]
        response = client.embeddings.create(model=model, input=batch)
        for text, item in zip(batch, response.data):
            cache[f"{model}:{text}"] = item.embedding
        calls += 1
        tokens += response.usage.total_tokens
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return {text: cache[f"{model}:{text}"] for text in texts}, {"calls": calls, "tokens": tokens}

def judge_answer(client, gold_facts, question, answer):
    prompt = f"Grade the answer against the gold meeting facts as correct, incorrect, or uncertain. Extra true words are allowed. A changed or missing person, date modifier, negation, or proposal/decision status is incorrect. An unsupported answer is incorrect. Facts: {gold_facts}\nQuestion: {question}\nAnswer: {answer}"
    response = client.responses.parse(model="gpt-4.1-2025-04-14", input=prompt, text_format=JudgeVerdict)
    if response.output_parsed is None:
        raise ValueError("No judge verdict returned")
    result = response.output_parsed.model_dump()
    result["usage"] = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens} if response.usage else None
    return result

from pydantic import BaseModel
from typing import Literal
class JudgeVerdict(BaseModel):
    verdict: Literal["correct", "incorrect", "uncertain"]
    reason: str

def pilot_jev(client, pairs, legend=""):
    from typesafe_sdk import Noul
    results = []
    for pair in pairs:
        response = client.system_one(state={"original": pair["original"], "encoded": pair["encoded"], "legend": legend}, questions={"preserved": Noul(instructions="Does the encoding preserve every person, date, negation, and decision status?")})
        probability = response.answers["preserved"].noul
        results.append({"id": pair["id"], "probability": probability, "correct": (probability >= 0.5) == pair["preserved"]})
    agreement = sum(item["correct"] for item in results) / len(results)
    false_accepts = sum(not pair["preserved"] and result["probability"] >= 0.9 for pair, result in zip(pairs, results))
    return {"agreement": agreement, "high_confidence_false_accepts": false_accepts, "qualified": agreement >= 0.85 and false_accepts == 0, "pairs": results}
