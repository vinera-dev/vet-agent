from pathlib import Path

from vet_agent.config import get_settings
from vet_agent.dependencies import get_llm
from vet_agent.rag.routes import get_faq_service


def main() -> None:
    settings = get_settings()
    service = get_faq_service(get_llm())
    total = service.ingest(Path(settings.faq_directory))
    print(f"{total} trechos indexados a partir de {settings.faq_directory}")


if __name__ == "__main__":
    main()
