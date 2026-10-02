import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Protocol

from openai import AzureOpenAI

from vet_agent.config import Settings

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a o as os de da do das dos e em no na nos nas um uma para por com que se ao ou "
    "qual quais como voce vcs tem ha sao meu minha".split()
)


class Embedder(Protocol):
    dimensions: int

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _tokenize(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return [token for token in TOKEN_PATTERN.findall(plain) if token not in STOPWORDS]


class HashingEmbedder:
    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        self._document_count = 0
        self._document_frequency: Counter[str] = Counter()
        self._token_index: dict[str, int] = {}

    def fit(self, texts: list[str]) -> None:
        self._document_count = len(texts)
        self._document_frequency = Counter()
        for text in texts:
            self._document_frequency.update(set(_tokenize(text)))
        self._token_index = {
            token: position % self.dimensions
            for position, token in enumerate(sorted(self._document_frequency))
        }

    def _bucket(self, token: str) -> int:
        if token in self._token_index:
            return self._token_index[token]
        digest = hashlib.sha256(token.encode()).digest()
        return int.from_bytes(digest[:4], "big") % self.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _idf(self, token: str) -> float:
        frequency = self._document_frequency.get(token, 0)
        return math.log((1 + self._document_count) / (1 + frequency)) + 1.0

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, count in Counter(_tokenize(text)).items():
            if self._document_count and token not in self._document_frequency:
                continue
            vector[self._bucket(token)] += (1.0 + math.log(count)) * self._idf(token)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class AzureOpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.dimensions = settings.embedding_dimensions
        self._settings = settings
        self._client: AzureOpenAI | None = None

    def _get_client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_key=self._settings.azure_openai_api_key,
                api_version=self._settings.azure_openai_api_version,
            )
        return self._client

    def fit(self, texts: list[str]) -> None:
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._get_client().embeddings.create(
            model=self._settings.azure_openai_embedding_deployment,
            input=texts,
        )
        return [item.embedding for item in response.data]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "azure":
        return AzureOpenAIEmbedder(settings)
    return HashingEmbedder()
