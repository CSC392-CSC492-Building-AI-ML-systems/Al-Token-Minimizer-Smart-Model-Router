"""Simple Streamlit front-end for the FastAPI app.

Run the API first:   uvicorn main:app --reload
Then run this:       streamlit run ui.py
"""

import time
from datetime import datetime, time as dtime, timezone
from urllib.parse import quote

import requests
import streamlit as st

# Pull the pipeline names and status list straight from the backend code when this
# file lives in the same project; otherwise fall back to a free-text / empty list.
try:
    from compressor import pipeline

    PIPELINE_NAMES = sorted(pipeline.PIPELINES)
    DEFAULT_PIPELINE = pipeline.DEFAULT_PIPELINE
except Exception:
    PIPELINE_NAMES, DEFAULT_PIPELINE = [], ""

try:
    import db

    STATUS_NAMES = list(db.STATUSES)
except Exception:
    STATUS_NAMES = []

st.set_page_config(page_title="Token Saver", layout="wide")
st.title("Token Saver")

api_url = st.sidebar.text_input("API base URL", "http://localhost:8000").rstrip("/")
timeout = st.sidebar.number_input("Request timeout (s)", min_value=5, max_value=300, value=60)

if "history" not in st.session_state:
    st.session_state.history = []  # newest first

tab_ask, tab_telemetry, tab_kb = st.tabs(["Ask", "Telemetry", "Knowledge Hub"])


# ---------------------------------------------------------------- Ask tab
with tab_ask:
    with st.form("ask_form"):
        prompt = st.text_area("Prompt", height=180, placeholder="Type your prompt here...")
        system = st.text_area(
            "System prompt (optional)", height=80, help="Sent as Gemini's system_instruction"
        )
        if PIPELINE_NAMES:
            compression = st.selectbox(
                "Compression pipeline",
                PIPELINE_NAMES,
                index=PIPELINE_NAMES.index(DEFAULT_PIPELINE) if DEFAULT_PIPELINE in PIPELINE_NAMES else 0,
            )
        else:
            compression = st.text_input("Compression pipeline", DEFAULT_PIPELINE)
        use_kb = st.checkbox(
            "Use knowledge base", help="Adds the closest knowledge-base chunks before the prompt (see the Knowledge Hub tab)"
        )
        submitted = st.form_submit_button("Send", type="primary")

    if submitted:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            body = {"prompt": prompt, "system": system.strip() or None}
            if compression:
                body["compression"] = compression
            if use_kb:
                body["kb"] = True
            try:
                with st.spinner("Waiting for Gemini..."):
                    resp = requests.post(f"{api_url}/ask", json=body, timeout=timeout)
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
            else:
                if resp.ok:
                    data = resp.json()
                    st.session_state.history.insert(
                        0,
                        {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "prompt": prompt,
                            "compression": compression + (" + kb" if use_kb else ""),
                            "text": data["text"],
                            "request_id": data["request_id"],
                        },
                    )
                else:
                    try:
                        detail = resp.json().get("detail")
                    except ValueError:
                        detail = resp.text
                    st.error(f"Request failed (HTTP {resp.status_code})")
                    if resp.status_code == 429:
                        st.info("Rate limited by the upstream API, wait a bit and try again.")
                    st.json(detail) if isinstance(detail, (dict, list)) else st.code(str(detail))

    for item in st.session_state.history:
        with st.container(border=True):
            st.caption(
                f"{item['time']} · pipeline: `{item['compression']}` · request_id: `{item['request_id']}`"
            )
            with st.expander("Prompt", expanded=False):
                st.text(item["prompt"])
            st.markdown(item["text"])

    if st.session_state.history and st.button("Clear history"):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------- Telemetry tab
with tab_telemetry:
    c1, c2, c3 = st.columns(3)
    model = c1.text_input("Model (optional)")
    if STATUS_NAMES:
        status = c2.selectbox("Status", ["(any)"] + STATUS_NAMES)
    else:
        status = c2.text_input("Status (optional)") or "(any)"
    use_dates = c3.checkbox("Filter by date (UTC)")

    since = until = None
    if use_dates:
        d1, d2 = st.columns(2)
        start_date = d1.date_input("Since (inclusive)")
        end_date = d2.date_input("Until (exclusive)")
        since = datetime.combine(start_date, dtime.min, tzinfo=timezone.utc).isoformat()
        until = datetime.combine(end_date, dtime.min, tzinfo=timezone.utc).isoformat()

    if st.button("Load summary"):
        params = {}
        if model:
            params["model"] = model
        if status and status != "(any)":
            params["status"] = status
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        try:
            resp = requests.get(f"{api_url}/telemetry/summary", params=params, timeout=timeout)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            if resp.ok:
                data = resp.json()
                # Show top-level numbers as metrics, everything else as raw JSON.
                scalars = {k: v for k, v in data.items() if isinstance(v, (int, float))} if isinstance(data, dict) else {}
                if scalars:
                    cols = st.columns(min(len(scalars), 4))
                    for i, (k, v) in enumerate(scalars.items()):
                        cols[i % len(cols)].metric(k, f"{v:,.2f}" if isinstance(v, float) else f"{v:,}")
                st.json(data)
            else:
                st.error(f"HTTP {resp.status_code}")
                st.code(resp.text)


# ------------------------------------------------------ Knowledge Hub tab
def kb_call(method, path, **kwargs):
    """Call a /kb endpoint. Returns the JSON, or None after showing the error."""
    try:
        resp = requests.request(method, f"{api_url}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return None
    if resp.ok:
        return resp.json()
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = resp.text
    st.error(f"HTTP {resp.status_code}: {detail}")
    return None


def wait_for_index(limit_seconds=180):
    """After a change, wait until the background rebuild is done."""
    with st.spinner("Rebuilding the knowledge-base index (embedding the pages)..."):
        deadline = time.monotonic() + limit_seconds
        while time.monotonic() < deadline:
            status = kb_call("GET", "/kb/status")
            if status is None or status["state"] != "rebuilding":
                return status
            time.sleep(1)
    st.warning("Still rebuilding, press Refresh in a moment.")
    return None


with tab_kb:
    st.caption(
        "Pages here are split into chunks and embedded. With **Use knowledge base** on (Ask tab), "
        "the closest chunks are added before the prompt. Built-in pages are read-only."
    )
    if msg := st.session_state.pop("kb_message", None):
        st.success(msg)

    listing = kb_call("GET", "/kb/pages")
    if listing is not None:
        status = listing["status"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pages", len(listing["pages"]))
        c2.metric("Added by users", sum(p["source"] == "user" for p in listing["pages"]))
        c3.metric("Chunks in index", status["chunks"])
        if status["state"] == "rebuilding":
            c4.metric("Index", "Rebuilding...")
        elif status["state"] == "error":
            c4.metric("Index", "Rebuild failed")
        elif status["index_version"] is None:
            c4.metric("Index", "Missing")
        else:
            c4.metric("Index", "Up to date" if status["up_to_date"] else "Out of date")

        if status["state"] == "error":
            st.error(f"The last rebuild failed ({status['error']}), see logs/app.log.")
        elif status["index_version"] is None or not status["up_to_date"]:
            st.warning("The index doesn't match the pages yet. Rebuild it so /ask uses the current pages.")

        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Refresh"):
            st.rerun()
        if b2.button("Rebuild index"):
            if kb_call("POST", "/kb/rebuild") is not None:
                wait_for_index()
                st.rerun()

    # ------------------------------------------------------------- add a page
    st.subheader("Add a page")
    with st.form("kb_add_form", clear_on_submit=True):
        uploaded = st.file_uploader("Upload a file", type=["md", "markdown", "txt"])
        name = st.text_input("Page name", placeholder="Defaults to the file name, e.g. course-notes")
        text = st.text_area("Or paste the content (Markdown or plain text)", height=200)
        add = st.form_submit_button("Add page", type="primary")

    if add:
        content = text
        if uploaded is not None and not content.strip():
            try:
                content = uploaded.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.error("That file isn't UTF-8 text.")
                content = ""
        page_name = name.strip() or (uploaded.name if uploaded is not None else "")
        if not page_name or not content.strip():
            st.warning("Give the page a name (or upload a file) and some content.")
        else:
            saved = kb_call("POST", "/kb/pages", json={"name": page_name, "text": content})
            if saved is not None:
                done = wait_for_index()
                verb = "Replaced" if saved["replaced"] else "Added"
                if done is not None and done["state"] == "ready":
                    st.session_state.kb_message = f"{verb} {saved['name']}, the index is up to date."
                else:
                    st.session_state.kb_message = f"{verb} {saved['name']}."
                st.rerun()

    # ------------------------------------------------------------- the pages
    if listing is not None:
        st.subheader("Pages")
        st.dataframe(
            [
                {"Name": p["name"], "Source": "added" if p["source"] == "user" else "built-in",
                 "Characters": p["chars"]}
                for p in listing["pages"]
            ],
            width="stretch",
            hide_index=True,
        )
        user_pages = [p["name"] for p in listing["pages"] if p["source"] == "user"]
        if user_pages:
            d1, d2 = st.columns([3, 1])
            to_delete = d1.selectbox("Remove an added page", user_pages)
            d2.write("")  # lines the button up with the select box
            if d2.button("Delete", type="secondary"):
                if kb_call("DELETE", f"/kb/pages/{quote(to_delete)}") is not None:
                    wait_for_index()
                    st.session_state.kb_message = f"Removed {to_delete}."
                    st.rerun()

    # ------------------------------------------------------------- try it
    st.subheader("Search the knowledge base")
    st.caption("Shows the chunks /ask would add for a question, without calling Gemini.")
    with st.form("kb_search_form"):
        question = st.text_input("Question", placeholder="e.g. How do I read a query parameter?")
        k = st.slider("Chunks", 1, 10, 3)
        search = st.form_submit_button("Search")
    if search and question.strip():
        found = kb_call("GET", "/kb/search", params={"q": question, "k": k})
        if found is not None:
            added = {p["name"] for p in listing["pages"] if p["source"] == "user"} if listing else set()
            for hit in found["hits"]:
                label = f"{hit['id']} · score {hit['score']:.3f}" + (" · added" if hit["page"] in added else "")
                with st.expander(label, expanded=True):
                    st.markdown(hit["text"])