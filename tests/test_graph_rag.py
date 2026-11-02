from pathlib import Path

from app.services.graph_rag import GraphRagService


def test_graph_rag_finds_support_relationship() -> None:
    service = GraphRagService.from_csv(Path("docs/graph-neo4j/support.csv"))

    relations = service.related_to("Esomeprazol 20mg")

    assert relations
    assert any(relation.supported_by == "Cefpobiotic 200" for relation in relations)


def test_graph_rag_returns_explicit_empty_context() -> None:
    service = GraphRagService.from_csv(Path("docs/graph-neo4j/support.csv"))

    assert "không có quan hệ" in service.context("unknown medicine")
