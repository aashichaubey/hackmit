# Tokenese Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based benchmark report that shows Tokenese text, exact LLM input text and shared output schema, token savings, and answer quality for meeting questions.

**Architecture:** A gold evaluation set and scoring rules come first. An OpenAI call extracts meeting facts; plain Python functions render those facts as compact English and Tokenese. A feedback-driven search qualifies the encoding on validation data before one untouched test run. Streamlit displays side-by-side prompts, answers, and recorded API usage.

**Tech Stack:** Python 3.11+, OpenAI Python SDK, Pydantic, tiktoken, Streamlit, pytest, LLMLingua-2 for the deletion baseline; optional TypeSafe Jev SDK for a grader pilot.

**Spec:** `docs/superpowers/specs/2026-09-19-tokenese-demo-design.md`

## Global Constraints

- Use `gpt-4.1-mini-2025-04-14` for the first benchmark. Do not silently switch models or tokenizers between methods.
- Keep personal names, project names, task text, and deadlines literal. Translate only fixed relationship labels in the first version.
- Count complete input strings, including instructions and grammar legends. Show the shared output schema and record actual API usage separately.
- Do not save pasted user notes or API keys. Save benchmark runs only for bundled example data.
- Use simple Python functions and small files. Do not add comments to implementation code.
- The browser page is primarily a report; the question box is for probing the result.
- Use development data to tune, validation data to select, and the test split once for the final claim. Never promote a candidate based on test feedback.
- Embedding similarity and model judges are supporting signals. Exact critical facts and checked question answers decide acceptance.

## File map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Dependencies and test command |
| `tokenese/facts.py` | Pydantic fact and answer shapes |
| `tokenese/encode.py` | Compact English and Tokenese renderers |
| `tokenese/prompts.py` | Exact extraction and answering prompts |
| `tokenese/model.py` | OpenAI extraction and answering calls, usage capture |
| `tokenese/tokens.py` | Model-specific local token count |
| `tokenese/baseline.py` | LLMLingua-2 deletion baseline |
| `tokenese/benchmark.py` | Run methods against examples and compute metrics |
| `tokenese/judges.py` | Optional semantic, LLM, and Jev grading |
| `tokenese/search.py` | Candidate screening, reward, and promotion log |
| `data/meetings.json` | Bundled meetings, gold facts, answers, split labels |
| `data/judge_pilot.json` | Human-labeled preserved and corrupted fact pairs |
| `data/candidates.json` | Named label and separator choices for the bounded search |
| `app.py` | Streamlit comparison and report |
| `tests/` | Focused unit tests for encoding, prompts, scoring, and failure paths |

---

### Task 0: Define the eval before implementing the language

**Files:** Create `pyproject.toml`, `data/meetings.json`, `data/judge_pilot.json`, `docs/evaluation-protocol.md`, `tests/test_eval_data.py`.

**Interfaces:** Each meeting has an `id`, `split`, `notes`, `facts`, and `questions`. Each question has text, accepted answers, and a kind such as owner, deadline, proposal, decision, agreement, or unknown. `judge_pilot.json` contains at least 40 fact pairs with human `preserved: true/false` labels.

- [ ] **Step 1: Add Python 3.11+ package setup with `openai`, `pydantic`, `tiktoken`, `streamlit`, `pytest`, and `llmlingua` dependencies. Write the scoring rules in `docs/evaluation-protocol.md`.** Define normalization for short answers, critical errors, extraction misses, false answers to unknown questions, token savings, full-pipeline token use, and model-judge disagreements. Specify the evaluation budget and version each frozen dataset.
- [ ] **Step 2: Create a 10-meeting, 30-question development smoke set with gold facts.** Include counterfactual pairs differing only in owner, deadline, negation, and proposal status. Expand it to 30 meetings with four questions each before the final benchmark: 18 development, six validation, six untouched test. Hand-check each fact and answer against the notes.

```json
{
  "id": "launch-01",
  "split": "dev",
  "notes": "Sarah proposed launching Friday. John agreed. David will finish the API by Thursday.",
  "facts": [
    {"kind": "proposal", "person": "Sarah", "text": "launch Friday", "deadline": "", "source_quote": "Sarah proposed launching Friday."},
    {"kind": "agreement", "person": "John", "text": "launch Friday", "deadline": "", "source_quote": "John agreed."},
    {"kind": "action", "person": "David", "text": "finish the API", "deadline": "Thursday", "source_quote": "David will finish the API by Thursday."}
  ],
  "questions": [
    {"question": "Who owns the API?", "answers": ["David"], "kind": "owner"},
    {"question": "When is the API due?", "answers": ["Thursday"], "kind": "deadline"},
    {"question": "Was Friday formally decided?", "answers": ["not found"], "kind": "unknown"}
  ]
}
```

- [ ] **Step 3: Create the 40-pair judge pilot.** Balance preserved and corrupted examples. Label the exact changed field and whether a judge saying "preserved" would be a false accept.
- [ ] **Step 4: Add data checks for unique IDs, valid split labels, all questions having accepted answers, and a paired corruption for each critical field.** Run `python -m pytest tests/test_eval_data.py -q` and inspect the corpus manually. This task is complete before writing an encoder or tuning a prompt.

### Task 1: Define supported facts and render them in three formats

**Files:** Create `tokenese/__init__.py`, `tokenese/facts.py`, `tokenese/encode.py`, `tests/test_encode.py`.

**Interfaces:** `Fact(kind, text, person, deadline, source_quote)` and `MeetingFacts(facts)` are the shared data contract. `render_english(facts)` and `render_tokenese(facts, profile)` each return a string; `tokenese_legend(profile)` returns the legend required to interpret it. A profile is a named dictionary loaded from `data/candidates.json`, initially `symbols` or `mixed`; include a version field.

- [ ] **Step 1: Write a failing renderer test.** Test one proposal, decision, action, agreement, and disagreement, including a missing deadline. Assert exact output and that all names, dates, and task text survive.

```python
def test_tokenese_keeps_facts():
    facts = MeetingFacts(facts=[
        Fact(kind="action", text="finish API integration", person="David", deadline="Thursday", source_quote="David will finish API integration by Thursday."),
    ])
    assert render_tokenese(facts, "symbols") == "A|David|finish API integration|@Thursday"
```

- [ ] **Step 2: Run `python -m pytest tests/test_encode.py -q`; confirm it fails because the modules do not exist.**
- [ ] **Step 3: Add the smallest Pydantic models and deterministic renderers.** Use five kinds: `proposal`, `decision`, `action`, `agreement`, `disagreement`. Keep one fact per line. For symbols use `P|person|text`, `D|text`, `A|person|text|@deadline`, `+|person|text`, and `-|person|text`. For mixed use `提`, `决`, `责`, `同`, and `反` in place of the first label. Put each label and the separator in `data/candidates.json` so the search can change one choice at a time. Omit the deadline field when empty. Keep source quotes out of the answer context. Reject literal `|` or newlines inside fields with a clear validation error so fields cannot change meaning.

```python
class Fact(BaseModel):
    kind: Literal["proposal", "decision", "action", "agreement", "disagreement"]
    text: str
    person: str = ""
    deadline: str = ""
    source_quote: str

class MeetingFacts(BaseModel):
    facts: list[Fact]
```

- [ ] **Step 4: Run `python -m pytest tests/test_encode.py -q`; confirm it passes.** Add a test for delimiter text and one for empty facts. Commit this task if the directory is a Git repository.

### Task 2: Build exact prompts and token measurement

**Files:** Create `tokenese/prompts.py`, `tokenese/tokens.py`, `tests/test_prompts.py`, `tests/test_tokens.py`.

**Interfaces:** `extraction_prompt(notes: str) -> str`; `answer_prompt(context: str, question: str, legend: str = "") -> str`; `count_tokens(text: str, model: str) -> int`. All methods call the same `answer_prompt`; the legend differs only when the encoding needs one.

- [ ] **Step 1: Write failing tests** proving a Tokenese prompt includes its legend, the raw prompt does not, the question is identical between methods, and a known string token count matches `len(tiktoken.encoding_for_model(MODEL).encode(text))`.

```python
def test_unknown_model_fails():
    with pytest.raises(KeyError):
        count_tokens("hello", "made-up-model")
```

- [ ] **Step 2: Run `python -m pytest tests/test_prompts.py tests/test_tokens.py -q`; confirm failure.**
- [ ] **Step 3: Implement one fixed extraction instruction and one fixed answering instruction.** The answering instruction limits answers to meeting facts and requires a short answer or `not found`. Put the context between clear delimiters. Use `tiktoken.encoding_for_model(model)` with no fallback for an unknown model.

```python
def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

- [ ] **Step 4: Run the two test files again; confirm pass.** Commit this task if Git is available.

### Task 3: Extract facts and answer questions through OpenAI

**Files:** Create `tokenese/model.py`, `tests/test_model.py`; update `tokenese/facts.py` with `Answer`.

**Interfaces:** `extract_facts(client, notes: str, model: str) -> tuple[MeetingFacts, Usage]`; `answer_question(client, prompt: str, model: str) -> tuple[Answer, Usage]`. `Usage` holds actual `input_tokens` and `output_tokens` from the API response. `Answer` has `found: bool` and `answer: str`.

- [ ] **Step 1: Write tests with a small fake client.** Assert that extraction passes the `MeetingFacts` schema, answering passes the `Answer` schema, actual usage is returned, and an absent `output_parsed` is an error. Keep API keys out of tests.

```python
def test_missing_parsed_output_raises(fake_client):
    fake_client.responses.parse_result.output_parsed = None
    with pytest.raises(ValueError, match="structured output"):
        extract_facts(fake_client, "meeting text", "gpt-4.1-mini-2025-04-14")
```

- [ ] **Step 2: Run `python -m pytest tests/test_model.py -q`; confirm failure.**
- [ ] **Step 3: Call `client.responses.parse(model=model, input=..., text_format=MeetingFacts)` for extraction and the same method with `Answer` for answering.** Read `response.output_parsed` and `response.usage.input_tokens` / `output_tokens`. Reject blank notes and malformed results. Let API exceptions reach the UI as a readable error; do not hide them behind a fabricated answer.

```python
response = client.responses.parse(
    model=model,
    input=extraction_prompt(notes),
    text_format=MeetingFacts,
)
if response.output_parsed is None:
    raise ValueError("No structured output returned")
```

- [ ] **Step 4: Run the model tests; confirm pass.** With an API key available, manually try one bundled meeting and inspect the extracted facts. Commit this task if Git is available.

### Task 4: Build the benchmark set and comparison runner

**Files:** Create `data/meetings.json`, `tokenese/benchmark.py`, `tests/test_benchmark.py`.

**Interfaces:** `run_case(client, case, model) -> dict` records facts, exact prompts, answers, local token counts, actual usage, and errors. `summarize(results) -> dict` computes per-method exact accuracy, owner/date errors, median prompt tokens, and total API tokens. `choose_encoding(facts, question, model, qualified_profiles) -> tuple[str, str]` returns the cheapest qualified profile or compact English.

- [ ] **Step 1: Load the gold cases from Task 0.** Keep the 10-meeting smoke run separate from the 30-meeting final corpus. A `run_case` call must accept a supplied gold fact list so encoding can be evaluated before the parser exists, and a parsed fact list for the later end-to-end run.

- [ ] **Step 2: Write failing tests for normalized exact answers, missing answers, average calculations, and the encoding fallback.** A profile that is not qualified or is longer than compact English must lose, even if it looks shorter visually.

```python
def test_fallback_when_tokenese_costs_more():
    facts = MeetingFacts(facts=[
        Fact(kind="decision", text="Friday", source_quote="Launch on Friday."),
    ])
    context, method = choose_encoding(facts, "What day?", "gpt-4.1-mini-2025-04-14", ["symbols"])
    assert method == "english"
```

- [ ] **Step 3: Run `python -m pytest tests/test_benchmark.py -q`; confirm failure.**
- [ ] **Step 4: Implement the runner in small functions.** Compare raw notes, compact English facts, and each Tokenese profile using the exact same question. Preserve failures in results. Store API output tokens separately from input tokens. For each method, save the full prompt string and its local count. Do not reuse an answer from one method for another. In end-to-end runs, extract once per meeting. Store the qualified profile and its version in `reports/latest.json` for the interactive page.
- [ ] **Step 5: Run the benchmark tests; confirm pass.** Add `python -m tokenese.benchmark` to run the bundled cases and write `reports/latest.json`. Do not write pasted user notes to this file. Commit this task if Git is available.

### Task 5: Add a real deletion baseline

**Files:** Create `tokenese/baseline.py`, `tests/test_baseline.py`; update `tokenese/benchmark.py`.

**Interfaces:** `compress_delete_only(notes: str, target_tokens: int, model: str) -> str` returns an LLMLingua-2 compressed context. Its answering prompt uses the same answering instruction as the other methods.

- [ ] **Step 1: Write a test with a fake compressor** that confirms the runner records the deletion method and counts its full prompt. Test that an unavailable model or compressor is recorded as an error, not a silent omission.
- [ ] **Step 2: Run `python -m pytest tests/test_baseline.py -q`; confirm failure.**
- [ ] **Step 3: Use the official LLMLingua-2 multilingual MeetingBank checkpoint.** Try a small fixed set of compression rates and choose the output whose full answering prompt is closest to the Tokenese prompt length. Record the actual achieved length, since a requested rate is not a guarantee. Reuse one loaded compressor across the benchmark.

```python
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
)
candidate = compressor.compress_prompt(notes, rate=0.5)["compressed_prompt"]
```

- [ ] **Step 4: Run baseline and benchmark tests; confirm pass.** On one example, inspect output and verify the recorded prompt length. Commit this task if Git is available.

### Task 6: Add graders and a bounded feedback-driven search

**Files:** Create `tokenese/judges.py`, `tokenese/search.py`, `tests/test_judges.py`, `tests/test_search.py`; update `tokenese/benchmark.py` and `data/candidates.json`.

**Interfaces:** `grade_exact(answer, accepted_answers) -> bool`; `cosine_similarity(left, right) -> float`; `judge_answer(client, gold_facts, question, answer) -> dict`; `pilot_jev(client, pairs) -> dict`; `qualify_candidate(results) -> bool`; `search_candidates(cases, budget) -> list[dict]`. Every run writes the candidate version, model ID, prompt version, split, score, token count, and failure labels to a local JSONL log.

- [ ] **Step 1: Write failing tests for the hard gate.** A candidate that changes David to John, Thursday to Friday, agreement to disagreement, or proposal to decision must fail regardless of a high cosine score or a favorable model judge. A missing answer to an unknown question is correct; an invented answer is not.

```python
def test_wrong_owner_blocks_candidate():
    result = {"critical_errors": ["owner"], "answer_accuracy": 1.0, "token_savings": 0.5}
    assert qualify_candidate(result) is False
```

- [ ] **Step 2: Run `python -m pytest tests/test_judges.py tests/test_search.py -q`; confirm failure.**
- [ ] **Step 3: Implement exact grading and simple cosine math first.** Cache embeddings from `text-embedding-3-small` by model and text; compare individual fact strings, not only whole meeting contexts. Show cosine as diagnostic data, never as acceptance. For non-exact answer variants, use a separate pinned `gpt-4.1-2025-04-14` judge with the gold facts and a fixed correct/incorrect/uncertain rubric; hide which method produced the answer. Queue uncertain or conflicting decisions for manual review.
- [ ] **Step 4: Pilot Jev only if `TYPESAFE_API_KEY` is available.** Add `typesafe-sdk` as an optional dependency, give Jev the grammar legend, and ask its typed yes/no question for each human-labeled fact pair in `data/judge_pilot.json`. Record agreement and false accepts. Treat Jev as a possible screening grader only if it reaches 85% human agreement and has zero corrupted pairs called preserved with probability at least 0.9. Keep checked answers as the final judge even if the pilot passes. If access is absent, record `not run: no key` instead of fabricating a score.

```python
response = client.system_one(
    state={"original": original_fact, "encoded": candidate_fact},
    questions={"preserved": Noul(instructions="Does encoded preserve every person, date, negation, and decision status in original?")},
)
probability = response.answers["preserved"].noul
```

- [ ] **Step 5: Search a bounded candidate set.** Change one label, separator, deadline marker, or field order per mutation. Screen syntax and local token counts for every candidate, run survivors on a development subset, then full development cases and validation cases. Keep a quality/token Pareto frontier. Accept only candidates with no added critical-field errors and at most one fewer correct answer than compact English across the 24 validation questions; among those, choose the lowest full-input-text token count. Stop after three rounds without a qualified gain or the recorded API-call budget. Never pass test cases into `search_candidates`.
- [ ] **Step 6: Run grader and search tests; confirm pass.** Run one smoke search, inspect the failure log, change one grammar choice based on a specific failure, and rerun the affected development cases. Commit this task if Git is available.

### Task 7: Build the report-first browser page

**Files:** Create `app.py`; optionally add one smoke test if UI logic needs extraction into a helper.

**Interfaces:** The page reads `reports/latest.json` for the fixed benchmark. `run_interactive(notes: str, question: str) -> dict` uses the same extraction, encoding, prompt, and answering functions for pasted notes, and reads the qualified profile from the report. Keep the most recent interactive result in `st.session_state` so rerenders do not repeat API calls.

- [ ] **Step 1: Create a two-tab page:** **Benchmark** shows summary metrics and per-question rows; **Try a meeting** accepts pasted notes and one question. Put raw, compact English, and Tokenese contexts and full prompts in labeled expandable sections. Show raw and Tokenese answers side by side, with actual usage.

```python
if st.button("Run comparison"):
    st.session_state["result"] = run_interactive(notes, question)

if "result" in st.session_state:
    result = st.session_state["result"]
    st.code(result["tokenese_prompt"], language="text")
```

- [ ] **Step 2: Show a clear breakdown:** context tokens, full input-text tokens, actual API input and output tokens, extraction overhead, and savings versus raw and compact English. Show the shared answer schema so the displayed input is not mistaken for the complete API payload. Count extraction once per meeting. Show the number of repeated questions needed for compressed total token use to beat raw total token use; if it never does, show that plainly. Label answer string equality as "same wording" rather than correctness. For bundled examples, show the gold answer and exact correctness.
- [ ] **Step 3: Show extracted facts and source excerpts; preserve line breaks in the encoded text.** Explain in one sentence that Tokenese is an encoding and the OpenAI tokenizer is used to measure it. Surface missing credentials, parse failures, and API errors in the page.
- [ ] **Step 4: Launch with `streamlit run app.py`.** Manually verify one bundled example and one pasted example; clicking other controls must not repeat paid calls. Commit this task if Git is available.

### Task 8: Run the fixed benchmark and report the evidence

**Files:** Update `reports/latest.json`, create `README.md`, and add a short results note at `reports/README.md`.

**Interfaces:** `reports/latest.json` is the data source for the browser report. The results note records model ID, dataset split, profile selected on development cases, held-out accuracy, prompt savings, extraction misses, and total API usage.

- [ ] **Step 1: Run the full test suite:** `python -m pytest -q`.
- [ ] **Step 2: Run the fixed benchmark once with the pinned model and LLMLingua baseline.** Freeze the winning profile using the 18 development and six validation meetings, then run the six untouched test meetings once. Keep raw results and errors. Report oracle-fact and end-to-end results separately.
- [ ] **Step 3: Inspect every owner/date mismatch and extraction omission.** Correct code or gold-data errors, then rerun only when there is a concrete correction. If the frozen profile fails on test, report the failure and add new test cases for a later version; do not tune the profile on this test split.
- [ ] **Step 4: Write the report with the actual numbers.** State whether the success gate in the spec passed. If Tokenese does not beat compact English with acceptable quality, keep the result visible and explain the likely cost driver rather than claiming a win.
- [ ] **Step 5: Document setup:** create a virtual environment, install from `pyproject.toml`, set `OPENAI_API_KEY`, run `python -m tokenese.benchmark`, then `streamlit run app.py`. State that the benchmark makes paid API calls and that the local report can be viewed without rerunning them. Commit the report and docs if Git is available.

## Review gates

After each task, check the behavior and keep changes limited to that task. Before calling the prototype complete, inspect the exact prompts displayed in the UI against the strings sent to the API, and compare summed API usage in the report with the per-call records. The final claim must distinguish savings from fact extraction from savings due to Tokenese encoding.
