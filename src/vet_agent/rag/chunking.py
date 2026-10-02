from pathlib import Path

from vet_agent.rag.models import Chunk

SECTION_MARKER = "## "


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    document_title = ""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("# ") and not document_title:
            document_title = line[2:].strip()
        elif line.startswith(SECTION_MARKER):
            sections.append((line[len(SECTION_MARKER) :].strip(), []))
        elif sections:
            sections[-1][1].append(line)

    chunks = []
    for index, (heading, lines) in enumerate(sections):
        body = "\n".join(lines).strip()
        if not body:
            continue
        title = f"{document_title} - {heading}" if document_title else heading
        chunks.append(
            Chunk(id=f"{source}#{index}", source=source, title=title, content=f"{title}\n{body}")
        )
    return chunks


def load_directory(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
    return chunks
