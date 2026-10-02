from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    source: str
    title: str
    content: str


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


class Source(BaseModel):
    source: str
    title: str
    score: float


class Answer(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source] = Field(default_factory=list)
