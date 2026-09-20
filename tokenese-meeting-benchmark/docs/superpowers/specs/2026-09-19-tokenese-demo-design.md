# Tokenese Demo Design

## Goal

Build a small browser-based benchmark report that accepts pasted, speaker-labeled meeting notes or a short transcript, compresses supported meeting facts for one OpenAI model, and answers questions about those facts. Show the language, exact model inputs, measured token use, and answer quality beside a raw-text baseline.

## Scope

The first version supports proposals, decisions, action items, owners, deadlines, and agreements or disagreements. It accepts text only. Questions outside those facts receive an explicit "not found in the meeting facts" answer. Dates remain in the wording supplied by the meeting unless an absolute date is stated. The demo uses the pinned `gpt-4.1-mini-2025-04-14` OpenAI model for extraction and answering so benchmark runs remain comparable.

This is a prototype, not a general-purpose language, transcript archive, audio pipeline, or trained compressor. It does not claim that Mandarin or symbols always cost fewer tokens. Candidate strings win only when their complete input text is cheaper for the configured tokenizer and they pass the quality check.

## Approaches considered

1. **Extract facts, then encode them (chosen).** A model turns meeting text into a fixed JSON schema. Deterministic code emits compact English and Tokenese variants. This isolates extraction loss from encoding loss and makes the demo explainable. Its extra extraction call must be included in cost reporting.
2. **Rewrite the transcript directly.** One prompt could produce shorthand quickly, but there is no reliable intermediate record of lost owners, dates, or decisions. It is harder to diagnose incorrect answers.
3. **Require structured notes as input.** This removes extraction cost and is easy to build, but it dodges the motivating use case of pasted meeting text.

## User flow

1. Paste meeting notes or choose a bundled example.
2. Click **Run comparison**. See extracted facts, compact English, Tokenese, and the exact input text for each method, along with the shared structured-output schema. Count the whole input text, including instructions and any grammar legend.
3. Ask a meeting question. See raw-text and Tokenese answers side by side, with the same/different label and actual API token usage. Bundled questions also show a manually checked expected answer and correctness for each method.
4. Inspect a benchmark report for the fixed example set: per-question results, compression ratios, quality, extraction errors, and full workflow cost.

"Tokenese" is an encoding of meeting facts; it does not replace the OpenAI model's tokenizer. The UI uses this wording so the reported token savings are understandable.

## Data and processing

The extraction result is a list of facts. Each fact has a `kind` (`proposal`, `decision`, `action`, `agreement`, or `disagreement`), `text`, `person`, `deadline`, and `source_quote`. Empty fields are allowed where irrelevant; no fact may omit its kind, text, or source quote. A proposal is never silently promoted to a decision. The extraction prompt asks the model to copy a short supporting excerpt and omit uncertain facts. The app validates the schema before encoding. The visible source quote is for inspection and is not silently added to the compressed answering context.

The encoder renders the same facts in two forms: compact English and candidate Tokenese. Only fixed relationship words and field labels are eligible for Mandarin or symbol substitutions in the first version. Names, project names, task descriptions, and date strings remain literal. A small, versioned candidate table defines the substitutions. The first version does not alias names.

The optimizer counts the **entire answering input text**, including its instruction, grammar legend, question, and encoded meeting context. A candidate profile is qualified on development and validation data before the demo uses it. At runtime, the selected profile is used only when its complete input text has fewer tokenizer tokens than compact English; otherwise the app uses compact English. Runtime selection makes no extra model calls.

The answering model receives either raw meeting text or the selected encoded context, with otherwise equivalent instructions. Its answer is constrained to the supported facts and must state when the answer is missing.

## Measurement

Report these separately:

- **Context tokens:** tokenizer count of the raw, compact English, and Tokenese answering contexts.
- **Full input-text tokens:** tokenizer count of each answering input string, including instructions and legend. The shared structured-output schema is shown separately and may contribute to API usage.
- **Actual API tokens:** usage returned by extraction and answering requests; show input and output separately.
- **End-to-end use:** extraction plus all question answering calls, counting extraction once per meeting. Report the number of repeated questions needed to recover extraction overhead, if any. Show LLMLingua's local compression time separately because it is not an OpenAI token charge.
- **Research overhead:** grader, embedding, and candidate-search calls are tracked separately from the cost of serving a meeting question.

Start with 10 varied development meetings and 30 manually checked question-answer pairs as a smoke set. Expand to 30 meetings with four checked questions each before making a comparative claim. Split the main set by meeting into 18 development, six validation, and six untouched test meetings. Include counterfactual pairs that change only a person, deadline, negation, or proposal/decision status, plus ambiguity, disagreement, missing deadlines, and repeated names. Compare raw text, compact English facts, Tokenese facts, and LLMLingua-2 as a delete-based baseline at approximately matched prompt length. Score exact owner and date fields separately from general answer correctness. Record extraction omissions separately so an encoding is not blamed for facts the parser never captured. Show answer differences directly; do not call surface text similarity semantic correctness.

## Eval-first tuning loop

Write the gold facts, expected answers, method split, failure labels, and scoring rules before implementing the encoder. First evaluate candidate encodings against hand-checked facts so parser mistakes cannot hide encoding mistakes; then evaluate the full transcript-to-answer pipeline. Store the exact input, output, model IDs, prompt version, candidate profile, token counts, and errors for every run. Cache by those values so unchanged calls are not billed repeatedly.

Use several signals with distinct jobs:

- **Hard checks:** schema validity; preservation of names, owners, dates, negation, and proposal versus decision; exact or accepted-form answers to closed questions; false answers to unanswerable questions. These are the acceptance gate.
- **Embedding cosine similarity:** a cheap diagnostic for broad semantic drift. It must not approve an encoding on its own, because a wrong name or date can leave the overall meaning vector close.
- **Reference-grounded LLM judge:** grade non-exact answer variants against gold facts with a fixed rubric and a model separate from the target answerer. Hide method labels from the judge and send uncertain or conflicting verdicts for human review.
- **Jev pilot:** if access is available, ask typed yes/no questions about individual fact preservation, giving Jev the grammar legend as context. Compare it with human labels on 40 balanced preserved/corrupted pairs, including counterfactual names, dates, negation, and decision status. Jev qualifies only as a screening grader if it has at least 85% agreement with human labels and zero corrupted pairs called preserved with probability at least 0.9. This small pilot is not proof of calibration; retain the results for human review. Jev does not generate free-form answers, so it cannot replace the downstream question-answering eval.

Optimize a small set of explicit vocabulary and grammar candidates. Screen all candidates with hard syntax and tokenizer checks, run the survivors on a development subset, and evaluate finalists on full development and validation sets. Keep a frontier of quality and full-input-text token cost. When a candidate fails, record the failure category, change one label or grammar rule, and rerun only the affected checks. Accept a candidate only if validation owner/date/negation/decision errors do not increase relative to compact English and overall answer accuracy falls by no more than one of the 24 validation questions. Among qualified candidates, choose the fewest full-input-text tokens. Stop after three rounds without a qualified improvement or a fixed API-call budget. Freeze the candidate before running the untouched test set once. A later tuning cycle needs new test examples.

This is feedback-driven discrete search, not training model weights with reinforcement learning. Weight-level RL would require far more examples and compute than this prototype; the measured reward and failure feedback still drive each vocabulary change.

Success for the demo is a frozen Tokenese profile with lower full input-text token count than compact English on most test examples, no critical-field regression against compact English, and at least 95% of raw-baseline question accuracy for the full pipeline. If no profile meets that gate, show the measured result and use compact English in the interactive demo.

## Failure behavior

Missing API credentials, unknown tokenizer mappings, model refusal, malformed extraction, and API failures produce a clear UI error. Do not guess an encoding for an unknown model. Do not display unsupported facts as certain. A failed benchmark case is recorded as a failure rather than omitted from averages.

## Implementation boundaries

Keep extraction, fact validation, encoding, token counting, answering, evaluation, and browser display in small Python modules. Use a simple Streamlit page for the report and interactive question box, and JSON files for bundled examples and gold answers. Keep API credentials in the environment and meeting text in the active app session only. No database or deployment is needed for the first version.

## Deferred work

Audio transcription, open-domain meeting questions, unbounded vocabulary discovery, learned encoders, multiple target models, persistent storage, and production access controls come after the prototype demonstrates a real quality-preserving gain.
