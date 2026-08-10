from llmcascade.tokens import estimate_tokens


def test_estimate_tokens_floor():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abc") == 1


def test_estimate_tokens_scaling():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("a" * 41) == 10
