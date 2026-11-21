"""OpenAI-compatible embedder, tested against a fake client."""

from types import SimpleNamespace

from remem.llm import OpenAIEmbedder


def test_openai_embedder_batches_and_normalises():
    calls = []

    def create(model, input):
        calls.append(len(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])

    emb = OpenAIEmbedder(SimpleNamespace(embeddings=SimpleNamespace(create=create)), "e", 2)
    out = emb.embed_texts(["a"] * 5, batch_size=2)
    assert calls == [2, 2, 1] and out[0] == [0.6, 0.8] and emb.embed_one("x") == [0.6, 0.8]
