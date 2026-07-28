"""Streamlit chat UI.

    streamlit run app.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()          # must run before config is imported and reads os.getenv

import streamlit as st

import config
import intake
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
        st.metric("Priced claims with a source", f"{c['pct']}%",
                    help="Citation density, not correctness — a cited claim may still "
                        "misread its source.")
        st.caption(f"{c['grounded']}/{c['factual_claims']} factual · "
                    f"{c['soft_suggestions']} soft suggestions")
        st.caption(f"Sources used: {c['sources_used']}/{c['sources_shown']} "
                    f"({c['source_utilisation']})")
        if c["unsupported"]:
            with st.expander(f"⚠️ {len(c['unsupported'])} unsourced claims"):
                for u in c["unsupported"]:
                    st.caption(u)

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

# before run_research_and_plan()
if st.session_state.get("researched_for") not in (None, trip.destination):
    st.session_state.registry = None          # destination moved — sources are void
st.session_state.researched_for = trip.destination
# ------------------------------------------------------------- pipeline ------
def run_research_and_plan() -> None:
    status = st.status("Researching your trip…", expanded=True)

    def on_search(i, total, q, added):
        status.write(f"Search {i}/{total} · +{added} · {q}")

    registry, queries = research.gather(trip, progress=on_search)
    st.session_state.registry = registry
    st.session_state.queries = queries
    status.write(f"**{len(registry)} sources collected.** Building the itinerary…")

    def on_build(kind, message):
        icon = {"skeleton": "🗺️", "day": "📍", "wait": "⏳"}.get(kind, "•")
        status.write(f"{icon} {message}")

    status.update(label="Writing your itinerary — this takes a few minutes on the free tier")
    itinerary = synthesize.build_itinerary(trip, registry, progress=on_build)

    st.session_state.itinerary = itinerary
    st.session_state.markdown = render.itinerary_to_markdown(itinerary, trip, registry)
    st.session_state.coverage = synthesize.coverage_report(
        itinerary, registry, sources_shown=config.SYNTH_SOURCES)
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
        # Re-run extraction first: "actually, make it Vancouver" is a new trip,
        # not a revision. Refinement never updates TripState, so without this
        # the sidebar and sources stay pinned to the old destination.
        probe = intake.intake_turn(trip, st.session_state.messages[:-1], prompt)
        HARD = {"destination", "origin", "duration_days"}
        if HARD & set(probe["changed"]):
            with st.chat_message("assistant"):
                st.markdown(f"Switching to **{trip.destination}** — starting fresh research.")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Switching to {trip.destination}."})
            st.session_state.update(registry=None, itinerary=None,
                                    markdown=None, coverage=None)
            run_research_and_plan()
            st.rerun()

        with st.chat_message("assistant"):
            status = st.status("Reworking the plan…", expanded=True)

            def on_refine(kind, message):
                icon = {"search": "🔍", "day": "📍", "section": "✏️", "wait": "⏳"}.get(kind, "•")
                status.write(f"{icon} {message}")

            st.session_state.itinerary = synthesize.refine_itinerary(
                st.session_state.itinerary, trip, st.session_state.registry,
                prompt, progress=on_refine,
            )
            st.session_state.markdown = render.itinerary_to_markdown(
                st.session_state.itinerary, trip, st.session_state.registry
            )
            st.session_state.coverage = synthesize.coverage_report(
                st.session_state.itinerary, st.session_state.registry,
                sources_shown=config.SYNTH_SOURCES)
            status.update(label="Plan updated", state="complete", expanded=False)
        st.rerun()

    else:
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            result = intake.intake_turn(trip, st.session_state.messages[:-1], prompt)
            st.caption(f"DEBUG dest={trip.destination!r} changed={result['changed']}")
            st.markdown(result["reply"])
            if result["assumptions"]:
                st.caption("Assuming: " + "; ".join(result["assumptions"]))
        st.session_state.messages.append({"role": "assistant", "content": result["reply"]})

        if result["ready"]:
            run_research_and_plan()
        st.rerun()
