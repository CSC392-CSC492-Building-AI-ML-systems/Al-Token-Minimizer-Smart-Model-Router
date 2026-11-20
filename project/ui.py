"""Simple Streamlit front-end for the FastAPI app.

Run the API first:   uvicorn main:app --reload
Then run this:       streamlit run ui.py
"""

from datetime import datetime, time as dtime, timezone

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

NO_COMPRESSION = "none"  # the preset that sends the prompt unchanged

# The compressed prompt can only be previewed when the compressor code is importable
# (this file lives in the same project as the API). The preview runs the same stages
# locally, without calling Gemini.
PREVIEW_AVAILABLE = bool(PIPELINE_NAMES)

CACHE_OPTIONS = {
    "off": "Off: always generate a fresh answer",
    "use": "Use: reuse a stored answer if the same (or a similar) prompt was asked before",
    "refresh": "Refresh: generate a fresh answer and replace the stored one",
}

st.set_page_config(page_title="Token Saver", layout="wide")
st.title("Token Saver")


# ------------------------------------------------------------------ helpers

def compressed_preview(prompt, system, compression):
    """Run the selected stages locally and return (compressed_text, compressed_system).

    Same stages the API runs, so this is what the model would get. One exception: the API may
    undo code compaction when it doesn't actually reduce the model's token count.
    """
    result = pipeline.compress(
        pipeline.Prompt(prompt, system or None), pipeline.resolve_stages(compression)
    )
    return result.render(), result.system


def send_request(body, prompt, compression, cache_mode):
    """POST the request to the API and add the result to the history."""
    try:
        with st.spinner("Waiting for Gemini..."):
            resp = requests.post(f"{api_url}/ask", json=body, timeout=timeout)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if resp.ok:
        data = resp.json()
        st.session_state.history.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "prompt": prompt,
                "compression": compression,
                "cache_mode": cache_mode,
                "cache_status": data.get("cache_status"),
                "cache_hit": data.get("cache_hit"),
                "over_limit": data.get("over_limit", False),
                "text": data["text"],
                "request_id": data["request_id"],
            },
        )
    else:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        st.error(f"Request failed (HTTP {resp.status_code}). Failures are logged too.")
        if resp.status_code == 429:
            st.info("Rate limited by the upstream API, wait a bit and try again.")
        st.json(detail) if isinstance(detail, (dict, list)) else st.code(str(detail))


def delete_from_cache(request_id):
    """Ask the API to remove the cached answer that this request used or created."""
    try:
        resp = requests.delete(f"{api_url}/cache/{request_id}", timeout=timeout)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if resp.status_code == 404:
        st.error("The API doesn't know this request_id (was the database reset?).")
    elif not resp.ok:
        st.error(f"Could not delete (HTTP {resp.status_code}).")
        st.code(resp.text)
    else:
        data = resp.json()
        if data.get("exact_cache_deleted"):
            st.success("Removed from the exact cache. Sending this prompt again will generate a fresh answer.")
        else:
            st.info("Nothing to remove from the exact cache (it already expired or was evicted).")
        if data.get("semantic_cache_deleted") is None:
            st.warning(
                "The similar-prompt (semantic) cache was not checked. It may still return this answer "
                "for a similar prompt. Use cache mode 'refresh' or 'off' to bypass it."
            )
 
 
def load_cache():
    """Fetch the cache listing into session state (used by the Cache tab)."""
    try:
        resp = requests.get(f"{api_url}/cache", timeout=timeout)
        resp.raise_for_status()
        st.session_state.cache_data = resp.json()
        st.session_state.cache_error = None
    except (requests.RequestException, ValueError) as exc:
        st.session_state.cache_error = f"Could not load the cache: {exc}"
 
 
def delete_cache_entry(key):
    try:
        resp = requests.delete(f"{api_url}/cache/key/{key}", timeout=timeout)
        resp.raise_for_status()
        deleted = resp.json().get("deleted")
        st.session_state.cache_msg = "Entry deleted." if deleted else "That entry had already expired."
    except (requests.RequestException, ValueError) as exc:
        st.session_state.cache_msg = f"Could not delete the entry: {exc}"
    load_cache()
 
 
def clear_whole_cache():
    try:
        requests.delete(f"{api_url}/cache", timeout=timeout).raise_for_status()
        st.session_state.cache_msg = "Exact cache cleared."
    except requests.RequestException as exc:
        st.session_state.cache_msg = f"Could not clear the cache: {exc}"
    st.session_state.confirm_clear = False
    load_cache()
 
def _set_action(name):
    st.session_state.pending_action = name
 

def privacy_summary():
    st.markdown(
        "- Your prompt (after any compression) **is sent to the upstream LLM (Gemini)**. "
        "Compressed prompts can still contain personal information.\n"
        "- This proxy is **not a privacy boundary**. Don't send anything you wouldn't send to Gemini directly.\n"
        "- Everything else runs locally: nothing is uploaded anywhere except the call to the model."
    )


# ------------------------------------------------------------------ sidebar
api_url = st.sidebar.text_input("API base URL", "http://localhost:8000").rstrip("/")
timeout = st.sidebar.number_input("Request timeout (s)", min_value=5, max_value=300, value=60)

st.sidebar.divider()
st.sidebar.subheader("Before you use this")
st.sidebar.warning(
    "Prompts are sent to an external LLM. Compression may remove information, "
    "and caching may return a stored answer. "
)

if "history" not in st.session_state:
    st.session_state.history = []  # newest first
if "pending" not in st.session_state:
    st.session_state.pending = None  # a request waiting for the user to review its compressed prompt

tab_ask, tab_telemetry, tab_cache = st.tabs(["Ask", "Telemetry", "Cache"])


# ---------------------------------------------------------------- Ask tab
with tab_ask:
    with st.expander("Privacy notice", expanded=False):
        privacy_summary()

    with st.form("ask_form"):
        prompt = st.text_area("Prompt", height=180, placeholder="Type your prompt here...")
        system = st.text_area(
            "System prompt (optional)", height=80, help="Sent as Gemini's system_instruction"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if PIPELINE_NAMES:
                compression = st.selectbox(
                    "Compression pipeline",
                    PIPELINE_NAMES,
                    index=PIPELINE_NAMES.index(DEFAULT_PIPELINE) if DEFAULT_PIPELINE in PIPELINE_NAMES else 0,
                    help=f"Choose '{NO_COMPRESSION}' to send your prompt unchanged. "
                    "The compression pipeline is open source and can be inspected.",
                )
            else:
                compression = st.text_input("Compression pipeline", DEFAULT_PIPELINE)
        with col_b:
            cache_mode = st.radio(
                "Response cache",
                list(CACHE_OPTIONS),
                format_func=CACHE_OPTIONS.get,
                index=0,
                help="Caching suits chatbots and helpbots. For creative work, where you may want "
                "different outcomes from the same prompt, leave it off.",
            )

        review = st.checkbox(
            "Show me the compressed prompt before sending",
            value=True,
            disabled=not PREVIEW_AVAILABLE,
            help="Compression changes your prompt before the model sees it. With this on, you review "
            "the compressed version and choose whether to send it. Has no effect with the "
            f"'{NO_COMPRESSION}' pipeline."
            + ("" if PREVIEW_AVAILABLE else " Unavailable: the compressor code couldn't be imported here."),
        )

        
        submitted = st.form_submit_button("Send", type="primary")

    if submitted:
        st.session_state.pending = None  # a new submit replaces any request still under review
        compressing = bool(compression) and compression != NO_COMPRESSION
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            body = {"prompt": prompt, "system": system.strip() or None, "cache": cache_mode}
            if compression:
                body["compression"] = compression
            if compressing and review and PREVIEW_AVAILABLE:
                st.session_state.pending = {
                    "body": body, "prompt": prompt, "compression": compression, "cache_mode": cache_mode,
                }
            else:
                send_request(body, prompt, compression, cache_mode)

    # ---- review step: show what compression did, then let the user decide
    action = st.session_state.pop("pending_action", None)
    pending = st.session_state.pending
    if pending and action == "cancel":
        st.session_state.pending = pending = None
        st.info("Cancelled. Nothing was sent.")
    elif pending and action in ("send", "send_none"):
        st.session_state.pending = None
        body = dict(pending["body"])
        compression_used = pending["compression"]
        if action == "send_none":
            body["compression"] = NO_COMPRESSION
            compression_used = NO_COMPRESSION
        send_request(body, pending["prompt"], compression_used, pending["cache_mode"])
        pending = None

    if pending:
        body = pending["body"]
        with st.container(border=True):
            st.subheader("Review before sending")
            st.caption(
                f"Pipeline `{pending['compression']}`. Nothing has been sent yet. This preview runs locally "
                "using the same stages as the API."
            )
            try:
                new_text, new_system = compressed_preview(
                    body["prompt"], body["system"], pending["compression"]
                )
            except Exception as exc:
                new_text = new_system = None
                st.error(f"Could not build the preview ({type(exc).__name__}). You can still cancel or send without compression.")

            if new_text is not None:
                if new_text == body["prompt"] and (new_system or None) == (body["system"] or None):
                    st.success("Compression made no changes to this prompt. It will be sent as you wrote it.")
                left, right = st.columns(2)
                left.markdown(f"**Your prompt** ({len(body['prompt']):,} characters)")
                left.code(body["prompt"], language=None, wrap_lines=True)
                right.markdown(f"**Compressed prompt that will be sent** ({len(new_text):,} characters)")
                right.code(new_text, language=None, wrap_lines=True)
                if (new_system or None) != (body["system"] or None):
                    st.markdown("**System instruction that will be sent** (your system prompt plus anything a stage added)")
                    st.code(new_system or "(none)", language=None, wrap_lines=True)
                st.caption(
                    "Check that every detail you need is still there. For code, the API may keep your original "
                    "formatting if compaction would not actually reduce the model's token count. The "
                    "token savings are shown in the Telemetry tab after sending."
                )

            b1, b2, b3 = st.columns(3)
            b1.button("Send compressed prompt", type="primary", disabled=new_text is None,
                      on_click=_set_action, args=("send",))
            b2.button(f"Send without compression ('{NO_COMPRESSION}')", on_click=_set_action, args=("send_none",))
            b3.button("Cancel", on_click=_set_action, args=("cancel",))

    for item in st.session_state.history:
        with st.container(border=True):
            st.caption(
                f"{item['time']} · pipeline: `{item['compression']}` · cache: `{item['cache_mode']}` · "
                f"request_id: `{item['request_id']}`"
            )
            st.caption("🤖 AI-generated answer. It may be inaccurate, check anything important.")

            if item["cache_hit"]:
                st.info(
                    f"This answer came from the **{item['cache_hit']} cache**, not a fresh generation. "
                    "It was written for an earlier prompt"
                    + (" that only *means the same* as yours" if item["cache_hit"] == "semantic" else "")
                    + ". If you wanted a different answer, send again with cache set to **refresh** or **off**."
                )
            if item["compression"] and item["compression"] != NO_COMPRESSION:
                st.caption(
                    "Your prompt was compressed before sending. If this answer looks wrong or incomplete, "
                    f"retry with the '{NO_COMPRESSION}' pipeline and compare."
                )
            if item["over_limit"]:
                st.warning("This reply went over the requested output length.")

            with st.expander("Prompt (as you typed it)", expanded=False):
                st.text(item["prompt"])
            st.markdown(item["text"])
            st.caption(
                "Settings for this request were logged with the request_id above "
                "(pipeline, cache mode, token counts, cost, latency). The prompt text itself is not logged."
            )

    if st.session_state.history and st.button("Clear history"):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------- Telemetry tab
with tab_telemetry:
    st.caption(
        "Every request, including failures, is one row in a local SQLite database: status, model, "
        "tokens (raw, compressed, reported, cached), cost and latency. A hash of the prompt is kept "
        "for cache keys, never the prompt or answer text. You can also run `python db.py summary` in the CLI."
    )
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

                # Headline numbers that matter for transparency: do failures, compression
                # and caching change what users get?
                if isinstance(data, dict) and data.get("requests"):
                    comp, cache = data.get("compression", {}), data.get("cache", {})
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Requests", f"{data['requests']:,}")
                    sr = data.get("success_rate")
                    m2.metric("Success rate", f"{sr:.1%}" if sr is not None else "-")
                    saved = comp.get("saved_rate")
                    m3.metric("Input tokens saved by compression", f"{saved:.1%}" if saved is not None else "-")
                    hr = cache.get("hit_rate")
                    m4.metric("Cache hit rate", f"{hr:.1%}" if hr is not None else "-")
                    st.caption(
                        "Tokens saved is a cost measure, not a quality measure. If an answer looks off, "
                        "use its request_id to check which pipeline and cache mode were used."
                    )
                st.json(data)
            else:
                st.error(f"HTTP {resp.status_code}")
                st.code(resp.text)

# ---------------------------------------------------------------- Cache tab
with tab_cache:
    st.caption(
        "Answers stored by the exact-match cache. Keys are hashes of the request, so the original prompt "
        "is not stored here, only the answer text. Entries expire on their own after the TTL. "
        "Anyone who can reach the API can list and delete them."
    )
    st.button("Load / refresh", on_click=load_cache)
 
    message = st.session_state.pop("cache_msg", None)
    if message:
        st.info(message)
    if st.session_state.get("cache_error"):
        st.error(st.session_state.cache_error)
 
    cache_data = st.session_state.get("cache_data")
    if cache_data is None:
        st.write("Click **Load / refresh** to see what is cached.")
    else:
        if not cache_data.get("enabled"):
            st.warning("The exact cache is disabled on the server, so nothing is stored in it.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Entries", f"{len(cache_data['entries']):,} / {cache_data['max_entries']:,}")
        m2.metric("TTL", f"{cache_data['ttl_seconds'] / 60:g} min")
        m3.metric("Size of stored answers", f"{sum(len(e['text']) for e in cache_data['entries']):,} chars")
 
        if not cache_data["entries"]:
            st.write("The exact cache is empty.")
        for entry in cache_data["entries"]:  # most recently used first
            with st.container(border=True):
                left, right = st.columns([6, 1])
                left.markdown(f"`{entry['key'][:12]}…` · expires in {entry['expires_in_seconds'] / 60:.1f} min")
                left.caption(
                    f"input tokens {entry['input_tokens_raw']} raw / {entry['input_tokens_compressed']} compressed · "
                    f"output tokens {entry['output_tokens'] if entry['output_tokens'] is not None else '-'}"
                )
                with left.expander("Cached answer"):
                    st.markdown(entry["text"])
                right.button("Delete", key=f"cache_del_{entry['key']}",
                             on_click=delete_cache_entry, args=(entry["key"],))
 
        if cache_data["entries"]:
            st.checkbox("I want to delete every entry in the exact cache", key="confirm_clear")
            st.button("Clear all", disabled=not st.session_state.get("confirm_clear"),
                      on_click=clear_whole_cache)
 
        if cache_data.get("semantic", {}).get("enabled"):
            st.info(
                "The similar-prompt (semantic) cache is on, but its entries can't be listed or deleted from "
                "here yet. It may still answer prompts that mean the same as one you deleted above. "
                "Use cache mode 'refresh' or 'off' to bypass it."
            )
 
