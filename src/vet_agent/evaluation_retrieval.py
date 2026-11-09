import json
from pathlib import Path
from typing import Any

from vet_agent.rag.chunking import load_directory
from vet_agent.rag.embeddings import Embedder, HashingEmbedder
from vet_agent.rag.store import InMemoryVectorStore, VectorStore

DEFAULT_CASES_PATH = Path("data/eval/faq.json")
DEFAULT_FAQ_DIRECTORY = Path("data/faq")
TOP_K = 3


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(directory: Path, embedder: Embedder, store: VectorStore) -> None:
    chunks = load_directory(directory)
    texts = [chunk.content for chunk in chunks]
    embedder.fit(texts)
    store.reset()
    store.add(chunks, embedder.embed(texts))


def evaluate_retrieval(
    cases: list[dict[str, str]],
    embedder: Embedder,
    store: VectorStore,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    hits_at_1 = 0
    hits_at_k = 0
    misses = []
    for case in cases:
        results = store.search(embedder.embed([case["question"]])[0], top_k)
        matches = [
            case["source"] == r.chunk.source and case["section"] in r.chunk.title for r in results
        ]
        hits_at_1 += bool(matches[:1] and matches[0])
        hit = any(matches)
        hits_at_k += hit
        if not hit:
            misses.append(
                {
                    "question": case["question"],
                    "expected": f"{case['source']} / {case['section']}",
                    "retrieved": [f"{r.chunk.source} / {r.chunk.title}" for r in results],
                }
            )
    total = len(cases)
    return {
        "total": total,
        f"hit@{top_k}": hits_at_k / total if total else 1.0,
        "hit@1": hits_at_1 / total if total else 1.0,
        "misses": misses,
    }


def main() -> None:
    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    build_index(DEFAULT_FAQ_DIRECTORY, embedder, store)
    report = evaluate_retrieval(load_cases(), embedder, store)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
