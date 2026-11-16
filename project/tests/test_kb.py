"""Knowledge-base checks that don't need the embedding model or a built index."""

from kb.chunk import CHUNKERS, DEFAULT_STRATEGY
from kb.evaluate import check_cases


def test_every_eval_phrase_is_on_its_page():
    check_cases()  # raises if a page was edited and a label no longer matches


def test_default_strategy_exists():
    assert DEFAULT_STRATEGY in CHUNKERS
