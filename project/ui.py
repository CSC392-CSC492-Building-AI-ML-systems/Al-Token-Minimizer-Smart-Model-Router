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

st.set_page_config(page_title="Token Saver", layout="wide")
st.title("Token Saver")

api_url = st.sidebar.text_input("API base URL", "http://localhost:8000").rstrip("/")
timeout = st.sidebar.number_input("Request timeout (s)", min_value=5, max_value=300, value=60)

if "history" not in st.session_state:
    st.session_state.history = []  # newest first

tab_ask, tab_telemetry = st.tabs(["Ask", "Telemetry"])


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
        submitted = st.form_submit_button("Send", type="primary")

    if submitted:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            body = {"prompt": prompt, "system": system.strip() or None}
            if compression:
                body["compression"] = compression
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
                            "compression": compression,
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