import json
from pathlib import Path
from typing import Any

from vet_agent.agent.guardrails import screen_input

DEFAULT_CASES_PATH = Path("data/eval/guardrails.json")
LABELS = ("emergency", "injection", "pass")


def load_cases(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_guardrails(cases: list[dict[str, str]]) -> dict[str, Any]:
    predictions = [(case, screen_input(case["text"]) or "pass") for case in cases]
    per_label = {}
    for label in LABELS:
        true_positive = sum(1 for case, pred in predictions if pred == label == case["expected"])
        predicted = sum(1 for _, pred in predictions if pred == label)
        expected = sum(1 for case, _ in predictions if case["expected"] == label)
        per_label[label] = {
            "precision": true_positive / predicted if predicted else 1.0,
            "recall": true_positive / expected if expected else 1.0,
            "support": expected,
        }
    errors = [
        {"text": case["text"], "expected": case["expected"], "predicted": pred}
        for case, pred in predictions
        if pred != case["expected"]
    ]
    return {
        "total": len(cases),
        "accuracy": (len(cases) - len(errors)) / len(cases) if cases else 1.0,
        "per_label": per_label,
        "errors": errors,
    }


def main() -> None:
    report = evaluate_guardrails(load_cases(DEFAULT_CASES_PATH))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
