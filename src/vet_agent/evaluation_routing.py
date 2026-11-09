import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from vet_agent.agent.graph import INTENTS, classify_intent
from vet_agent.config import get_settings
from vet_agent.dependencies import get_llm
from vet_agent.llm import LLMClient

DEFAULT_CASES_PATH = Path("data/eval/routing.json")


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_routing(cases: list[dict[str, str]], llm: LLMClient) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    errors = []
    for case in cases:
        predicted = classify_intent(llm, [{"role": "user", "content": case["message"]}])
        confusion[(case["intent"], predicted)] += 1
        if predicted != case["intent"]:
            errors.append(
                {"message": case["message"], "expected": case["intent"], "predicted": predicted}
            )
    total = len(cases)
    per_intent = {}
    for intent in sorted(INTENTS):
        expected = sum(count for (exp, _), count in confusion.items() if exp == intent)
        correct = confusion[(intent, intent)]
        per_intent[intent] = {
            "support": expected,
            "recall": correct / expected if expected else 1.0,
        }
    return {
        "total": total,
        "accuracy": (total - len(errors)) / total if total else 1.0,
        "per_intent": per_intent,
        "errors": errors,
    }


def main() -> int:
    settings = get_settings()
    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        sys.stderr.write(
            "Routing evaluation needs AZURE_OPENAI_* credentials in the environment.\n"
        )
        return 2
    report = evaluate_routing(load_cases(), get_llm())
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
