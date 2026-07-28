"""Streamlit chat UI.

    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

import intake
import links
import render
import research
import synthesize
from state import TripState

st.set_page_config(page_title="Travel Planner", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------- session ----
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("trip", TripState())
    ss.setdefault("messages", [{"role": "assistant", "content": intake.OPENING}])
    ss.setdefault("phase", "intake")          # intake -> researching -> plan
    ss.setdefault("registry", None)
    ss.setdefault("itinerary", None)
    ss.setdefault("markdown", None)
    ss.setdefault("coverage", None)
    ss.setdefault("queries", [])


init_state()
trip: TripState = st.session_state.trip


# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.subheader("Trip so far")
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set")
    lines = trip.summary_lines()
    if lines:
        for line in lines:
            label, _, value = line.partition(": ")
            st.markdown(f"**{label}**  \n{value}")
    else:
        st.caption("Nothing collected yet.")

    missing = trip.missing_required()
    if missing:
        st.caption("Still needed: " + ", ".join(missing))

    if st.session_state.coverage:
        c = st.session_state.coverage
        st.divider()
        st.subheader("Sourcing")
        st.metric("Claims with a source", f"{c['pct']}%")
        st.caption(f"{c['sources_used']} of {c['sources_retrieved']} retrieved sources used")

    if st.session_state.queries:
        with st.expander("Searches run"):
            for q in st.session_state.queries:
                st.caption(q)

    st.divider()
    if st.button("Start over", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ------------------------------------------------------------- transcript ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.markdown:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.markdown, unsafe_allow_html=True)
        st.download_button(
            "Download plan (.md)",
            st.session_state.markdown,
            file_name=f"{(trip.destination or 'trip').replace(' ', '_').lower()}_itinerary.md",
            mime="text/markdown",
        )


# ------------------------------------------------------------- pipeline ------
def run_research_and_plan() -> None:
    status = st.status("Researching your trip…", expanded=True)

    def progress(i, total, q):
        status.write(f"Search {i}/{total}: {q}")

    registry, queries = research.gather(trip, progress=progress)
    st.session_state.registry = registry
    st.session_state.queries = queries
    status.write(f"Collected {len(registry)} sources. Writing the itinerary…")

    itinerary = synthesize.build_itinerary(trip, registry)
    st.session_state.itinerary = itinerary
    st.session_state.markdown = render.itinerary_to_markdown(itinerary, trip, registry)
    st.session_state.coverage = synthesize.coverage_report(itinerary, registry)
    st.session_state.phase = "plan"
    status.update(label="Plan ready", state="complete", expanded=False)


# ---------------------------------------------------------------- input ------
placeholder = (
    "Tell me about your trip…"
    if st.session_state.phase == "intake"
    else "Want changes? e.g. 'make day 3 lighter' or 'swap the museum for something outdoors'"
)

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.phase == "plan":
        with st.chat_message("assistant"), st.spinner("Reworking the plan…"):
            st.session_state.itinerary = synthesize.refine_itinerary(
                st.session_state.itinerary, trip, st.session_state.registry, prompt
            )
            st.session_state.markdown = render.itinerary_to_markdown(
                st.session_state.itinerary, trip, st.session_state.registry
            )
            st.session_state.coverage = synthesize.coverage_report(
                st.session_state.itinerary, st.session_state.registry
            )
            st.markdown("Updated — see the revised plan below.")
        st.rerun()

    else:
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            result = intake.intake_turn(trip, st.session_state.messages[:-1], prompt)
            st.markdown(result["reply"])
            if result["assumptions"]:
                st.caption("Assuming: " + "; ".join(result["assumptions"]))
        st.session_state.messages.append({"role": "assistant", "content": result["reply"]})

        if result["ready"]:
            run_research_and_plan()
        st.rerun()
