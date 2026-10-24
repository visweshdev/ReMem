from remem.offline import ConceptHashEmbedder, content_tokens, count_tokens


def test_embedder_is_deterministic_and_normalised():
    e = ConceptHashEmbedder(64)
    text = "The payments service deploys to region eu-west-1."
    assert e.embed_one(text) == e.embed_one(text)
    assert abs(sum(x * x for x in e.embed_one(text)) - 1.0) < 1e-9


def test_lexicon_makes_synonyms_similar():
    plain = ConceptHashEmbedder(256)
    lex = ConceptHashEmbedder(256, {"hosted": "cmp", "deploys": "cmp"})

    def cos(e, x, y):
        return sum(p * q for p, q in zip(e.embed_one(x), e.embed_one(y)))

    assert cos(lex, "payments hosted", "payments deploys") > cos(plain, "payments hosted", "payments deploys") + 0.2


def test_count_tokens_and_content_tokens():
    assert count_tokens("a b, c.") == 5
    assert "the" not in content_tokens("the cat")
