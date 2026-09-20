import json
import math
import os
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from tokenese import MODEL
from tokenese.benchmark import choose_encoding, load_cases, method_input
from tokenese.facts import Answer, MeetingFacts
from tokenese.judges import grade_exact
from tokenese.model import answer_question
from tokenese.tokens import count_tokens
from tokenese.evo_view import render_evolution, render_workspace
from tokenese.v1_workspace import compilation_key, compile_once

st.set_page_config(page_title="Tokenese · V1 demo", page_icon="↳", layout="wide")
load_dotenv(Path(__file__).parent / ".env")
st.title("Tokenese")
st.caption("Compile meeting notes once. Compare compact representations for repeated questions.")
report_path = Path(__file__).parent / "reports" / "latest.json"
report = json.loads(report_path.read_text()) if report_path.exists() else {}
stress_path = Path(__file__).parent / "reports" / "stress.json"
stress = json.loads(stress_path.read_text()) if stress_path.exists() else {}
try_tab, benchmark_tab, evolution_tab, workspace_tab = st.tabs([
    "V1 demo",
    "V1 benchmark",
    "Research archive",
    "V2 transcript study",
])
with evolution_tab:
    render_evolution()
with workspace_tab:
    render_workspace()
with benchmark_tab:
    st.write(report.get("status", "Run `python -m tokenese.benchmark` to create a local report."))
    st.write("Model:", report.get("model", MODEL))
    st.write("Dataset:", report.get("case_counts", {}))
    if report.get("success_gate"):
        st.subheader("Success gate")
        st.json(report["success_gate"])
    if report.get("research_usage"):
        with st.expander("Research calls and diagnostics"):
            st.json({"usage": report["research_usage"], "embeddings": report.get("embedding_diagnostics"), "judge": report.get("judge_diagnostics"), "jev": report.get("jev_pilot")})
    if report.get("local_token_metrics"):
        st.subheader("Full input token counts by split")
        st.json(report["local_token_metrics"])
    if report.get("profile_screen"):
        st.subheader("Local token screening")
        st.dataframe(report["profile_screen"])
    for source, splits in report.get("summary", {}).items():
        st.subheader(f"{source.title()} facts")
        for split, summary in splits.items():
            st.write(split, summary)
    if report.get("adjudicated_summary"):
        with st.expander("Human-reviewed answer scores"):
            st.json(report["adjudicated_summary"])
    if stress:
        with st.expander("Speaker-labeled stress set"):
            st.json(stress["summary"])
    if report.get("runs"):
        run = st.selectbox("Inspect run", report["runs"], format_func=lambda item: f"{item['id']} · {item['fact_source']}")
        rows = []
        for question in run["questions"]:
            for name, method in question["methods"].items():
                rows.append({"question": question["question"], "method": name, "expected": ", ".join(question["answers"]), "answer": (method.get("answer") or {}).get("answer"), "correct": method.get("correct"), "input tokens": (method.get("usage") or {}).get("input_tokens"), "output tokens": (method.get("usage") or {}).get("output_tokens"), "error": method.get("error")})
        st.dataframe(rows)
        st.json(run)
with try_tab:
    st.subheader("Turn meeting notes into compact facts")
    st.caption("V1 extracts explicit facts once, then compares raw notes, compact English, and two experimental Tokenese representations on the same question.")
    if not report.get("qualified_profiles"):
        st.warning("No symbolic Tokenese profile passed the saved validation gate. The demo still shows those profiles for an honest side-by-side comparison; compact English remains the selected V1 route.")
    cases = load_cases()
    example = st.selectbox("Example", ["Custom"] + [case["id"] for case in cases])
    selected = next((case for case in cases if case["id"] == example), None)
    notes = st.text_area("Meeting notes", value=selected["notes"] if selected else "", height=160)
    question = st.text_input("Question", value=selected["questions"][0]["question"] if selected else "")
    if st.button("Run comparison"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Set OPENAI_API_KEY to run an interactive comparison.")
        elif not notes.strip() or not question.strip():
            st.error("Add meeting notes and a question.")
        else:
            try:
                client = OpenAI()
                compilation = compile_once(st.session_state, notes, client)
                facts = compilation.facts
                extraction_usage = compilation.usage
                qualified = report.get("qualified_profiles", [])
                chosen_context, chosen_method = choose_encoding(facts, question, MODEL, qualified)
                methods = ["raw", "english", "symbols", "mixed"]
                rows = {}
                for method in methods:
                    context, prompt = method_input(facts, notes, question, method)
                    answer, usage = answer_question(client, prompt, MODEL)
                    rows[method] = {"context": context, "prompt": prompt, "context_tokens": count_tokens(context, MODEL), "full_input_tokens": count_tokens(prompt, MODEL), "answer": answer.model_dump(), "usage": usage.model_dump()}
                gold = next((item for item in selected["questions"] if item["question"] == question), None) if selected else None
                st.session_state["v1_result"] = {"compilation_key": compilation_key(notes), "facts": facts.model_dump(), "extraction_usage": extraction_usage.model_dump(), "extraction_reused": compilation.reused, "methods": rows, "chosen_method": chosen_method, "chosen_context": chosen_context, "gold": gold}
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.get("v1_result", {}).get("compilation_key") == compilation_key(notes):
        result = st.session_state["v1_result"]
        st.subheader("Extracted facts")
        st.json(result["facts"])
        st.write("Extraction:", "reused from this session" if result["extraction_reused"] else "new model call")
        st.write("Extraction API usage:", result["extraction_usage"])
        st.write("Selected runtime encoding:", result["chosen_method"])
        raw_usage = result["methods"]["raw"]["usage"]
        chosen_usage = result["methods"][result["chosen_method"]]["usage"]
        extraction_tokens = 0 if result["extraction_reused"] else sum(result["extraction_usage"].values())
        original_extraction_tokens = sum(result["extraction_usage"].values())
        saved_per_question = sum(raw_usage.values()) - sum(chosen_usage.values())
        st.write("Full workflow tokens for this question:", sum(chosen_usage.values()) + extraction_tokens)
        st.write("Questions to recover extraction tokens:", math.ceil(original_extraction_tokens / saved_per_question) if saved_per_question > 0 else "never at this per-question rate")
        st.caption("The same structured answer schema is sent with every method. API token usage includes it.")
        with st.expander("Shared structured-output schemas"):
            st.json({"extraction": MeetingFacts.model_json_schema(), "answer": Answer.model_json_schema()})
        if result["gold"]:
            st.write("Expected answer:", result["gold"]["answers"])
        columns = st.columns(2)
        for index, (name, row) in enumerate(result["methods"].items()):
            with columns[index % 2]:
                st.subheader(name)
                st.write("Answer:", row["answer"])
                if result["gold"]:
                    st.write("Correct:", grade_exact(Answer.model_validate(row["answer"]), result["gold"]["answers"]))
                st.write("Context tokens:", row["context_tokens"])
                st.write("Full input text tokens:", row["full_input_tokens"])
                st.write("Actual API usage:", row["usage"])
                with st.expander("Context and exact input"):
                    st.code(row["context"], language="text")
                    st.code(row["prompt"], language="text")
        st.write("Same wording:", result["methods"]["raw"]["answer"]["answer"] == result["methods"]["symbols"]["answer"]["answer"])
