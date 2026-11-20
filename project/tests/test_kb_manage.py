"""The knowledge hub: adding/removing pages, the rebuild, and the /kb endpoints.

The embedding model is replaced by a tiny bag-of-words fake and every folder is
temporary, so nothing is downloaded and the real corpus and index are untouched.

Run from the project folder:
    uv run --with pytest pytest tests/test_kb_manage.py -v
"""

import hashlib
import os

import numpy as np
import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from kb import chunk, embed, load, manage, retrieve  # noqa: E402


class FakeModel:
    """Each word adds 1 to one of 64 slots, so texts sharing words are close."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture(autouse=True)
def kb_dirs(monkeypatch, tmp_path):
    base, user, index = tmp_path / "corpus", tmp_path / "user_corpus", tmp_path / "index"
    base.mkdir()
    (base / "builtin.md").write_text("# Built in\n\nQuery parameters come after the question mark.", encoding="utf-8")
    monkeypatch.setattr(load, "CORPUS_DIR", base)
    monkeypatch.setattr(load, "USER_CORPUS_DIR", user)
    monkeypatch.setattr(embed, "INDEX_DIR", index)
    monkeypatch.setattr(retrieve, "INDEX_DIR", index)
    monkeypatch.setattr(retrieve, "_indexes", {})
    monkeypatch.setattr(embed, "get_model", FakeModel)
    monkeypatch.setattr(retrieve, "get_model", FakeModel)
    monkeypatch.setattr(chunk, "DEFAULT_STRATEGY", "token")  # the semantic chunker would import the real model
    monkeypatch.setattr(manage, "DEFAULT_STRATEGY", "token")
    monkeypatch.setattr(main, "DEFAULT_KB_STRATEGY", "token")
    monkeypatch.setattr(manage, "_status", {"state": "ready", "pending": 0, "error": None,
                                            "built_at": None, "seconds": None})
    return base, user, index


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


# ------------------------------------------------------------------ page names


@pytest.mark.parametrize("raw, expected", [
    ("Course Notes", "course-notes.md"),
    ("notes.txt", "notes.md"),
    ("NOTES.MD", "notes.md"),
    ("../../main.py", "main.py.md"),
    ("..\\..\\secrets", "secrets.md"),
    ("  a/b/c.markdown ", "c.md"),
])
def test_clean_name_stays_inside_the_folder(raw, expected):
    assert manage.clean_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "...", "///", "#$%"])
def test_clean_name_rejects_names_without_letters(raw):
    with pytest.raises(manage.PageError):
        manage.clean_name(raw)


# ------------------------------------------------------------------ add / delete


def test_add_page_saves_to_the_user_folder(kb_dirs):
    _, user, _ = kb_dirs
    assert manage.add_page("My Notes", "# Notes\n\nhello") == {"name": "my-notes.md", "replaced": False}
    assert (user / "my-notes.md").read_text(encoding="utf-8") == "# Notes\n\nhello"
    assert manage.add_page("my-notes", "new text")["replaced"] is True
    assert not list(user.glob("*.tmp"))


def test_list_pages_marks_where_pages_come_from():
    manage.add_page("extra", "text")
    sources = {page["name"]: page["source"] for page in manage.list_pages()}
    assert sources == {"builtin.md": "base", "extra.md": "user"}


@pytest.mark.parametrize("name, text, message", [
    ("builtin", "text", "built-in page"),
    ("empty", "   \n", "empty"),
    ("big", "x" * (manage.MAX_PAGE_CHARS + 1), "over"),
], ids=["builtin-name", "empty", "too-big"])  # short ids: Windows caps the env var pytest puts them in
def test_add_page_rejects(name, text, message):
    with pytest.raises(manage.PageError, match=message):
        manage.add_page(name, text)


def test_delete_page_removes_only_user_pages(kb_dirs):
    base, user, _ = kb_dirs
    manage.add_page("extra", "text")
    assert manage.delete_page("extra.md") == "extra.md"
    assert not (user / "extra.md").exists()
    with pytest.raises(FileNotFoundError):
        manage.delete_page("builtin.md")  # looked up in the user folder only
    assert (base / "builtin.md").exists()
    with pytest.raises(manage.PageError):
        manage.delete_page("../builtin.md")


# ------------------------------------------------------------------ rebuild


def test_rebuild_swaps_in_an_index_with_the_new_page():
    manage.add_page("zebras", "Zebras have black and white stripes.")
    manage.mark_pending()
    assert manage.status()["state"] == "rebuilding"
    manage.rebuild()
    current = manage.status()
    assert current["state"] == "ready" and current["up_to_date"] and current["chunks"] == 2
    hits = retrieve.retrieve("zebras stripes", k=1, strategy="token")
    assert hits[0]["page"] == "zebras.md"


def test_failed_rebuild_reports_an_error(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("private detail")

    monkeypatch.setattr(embed, "build_index", broken)
    manage.mark_pending()
    manage.rebuild()
    current = manage.status()
    assert current["state"] == "error" and current["error"] == "RuntimeError"


def test_index_files_that_dont_match_are_rejected(kb_dirs):
    _, _, index = kb_dirs
    embed.build_index("token", index)
    np.save(index / "token" / "index.npy", np.zeros((5, 64), dtype=np.float32))
    with pytest.raises(RuntimeError, match="don't match"):
        retrieve.load_index("token")


# ------------------------------------------------------------------ HTTP


def test_add_list_search_and_delete_over_http(client):
    response = client.post("/kb/pages", json={"name": "Zebras", "text": "Zebras have black and white stripes."})
    assert response.status_code == 201 and response.json()["name"] == "zebras.md"

    # TestClient runs the background rebuild before returning
    listing = client.get("/kb/pages").json()
    assert {"name": "zebras.md", "source": "user"}.items() <= next(
        p for p in listing["pages"] if p["name"] == "zebras.md").items()
    assert listing["status"]["state"] == "ready" and listing["status"]["up_to_date"]

    hits = client.get("/kb/search", params={"q": "zebra stripes", "k": 1}).json()["hits"]
    assert hits[0]["page"] == "zebras.md"

    assert client.delete("/kb/pages/zebras.md").status_code == 200
    assert client.delete("/kb/pages/zebras.md").status_code == 404
    hits = client.get("/kb/search", params={"q": "zebra stripes", "k": 1}).json()["hits"]
    assert hits[0]["page"] == "builtin.md"


def test_http_errors(client):
    assert client.post("/kb/pages", json={"name": "builtin", "text": "x"}).status_code == 400
    assert client.post("/kb/pages", json={"name": "a", "text": ""}).status_code == 400
    assert client.post("/kb/pages", json={"name": "a"}).status_code == 422
    assert client.delete("/kb/pages/builtin.md").status_code == 404


def test_search_without_an_index_says_so(client):
    response = client.get("/kb/search", params={"q": "anything"})
    assert response.status_code == 409


def test_manual_rebuild_builds_a_missing_index(client):
    assert client.post("/kb/rebuild").status_code == 202
    assert client.get("/kb/status").json()["chunks"] == 1
