# Tokenese V2 Design

This design turns the pasted Tokenese V2 proposal into an executable scope for the existing Python and Streamlit demo. The proposal supplies the product hypothesis; this document fixes the choices needed to implement and evaluate it.

## Goal

Discover a legend-free, target-model-specific textual encoding of meeting facts that reduces **actual answer input tokens** while retaining checked meaning. Compile facts once per meeting, then reuse the compiled text for questions, a fact-bounded summary, and action-item views. If no candidate qualifies, show that result and use raw notes.

## Boundaries

- Pin the V2 experiment to `gpt-4.1-mini-2025-04-14` and its `tiktoken` encoding. A different model requires a new search and evaluation.
- Keep the five existing fact kinds: action, proposal, decision, agreement, disagreement. Add blocker, status, and open question with annotated examples. Owner and deadline remain explicit fields of a fact, not separate fact kinds.
- Reuse the current `Fact`, `MeetingFacts`, `Answer`, model-call helper, and benchmark baselines. Add V2 extraction and answering prompts without changing V1's five-kind extraction behavior. Do not create a second tokenizer module or a trained model for the MVP.
- Generate hundreds of deterministic, versioned encoding profiles from short English, word order, punctuation, established multilingual words, and cautious abbreviations. A profile renders an entire meeting consistently. Symbol-only profiles may be screened but get no legend.
- Keep names, task/object terms, negation, and date modifiers exact. An alias may replace a name or term only if that alias appears explicitly in the source and is unambiguous within the meeting. A date abbreviation may enter the finalist set only after a local round-trip check and exact-answer evaluation; it must not remove `next`, `this`, `end of`, or a year.
- A compiled memory is held once per meeting in the current Streamlit session. It may be exported as JSON, but no database or cross-session account storage is part of this version. Independent API calls still send the memory each time; compilation alone does not remove those repeated input tokens.
- The fact-bounded summary and action-item view may only use supported extracted facts. A general transcript summary is outside this version.
- Preserve the existing V1 benchmark and report files as historical results. V2 uses separate data, search logs, and reports.

## Flow

1. Extract validated semantic facts from the meeting transcript. Keep literal source quotes and evidence locations for inspection. An extraction error is reported separately from an encoding error.
2. Generate a bounded catalog of deterministic profile variants. Local tokenization discards duplicate text and any variant that fails to improve the complete answering prompt over compact English and raw notes for the same facts and question workload.
3. Check fields before model calls: fact kind, person, text, negation, deadline, and unambiguous aliases. Any candidate with a local integrity failure is rejected. The model then answers atomic and whole-meeting questions from the candidate without a grammar legend.
4. Search on development meetings only, select one frozen profile on validation meetings, and run a fresh held-out test once. The test never drives candidate edits.
5. Compile the selected profile once per meeting into `MeetingMemory` containing the session-only raw notes, validated facts, exact encoded text, profile version, source hash, and measured compile usage. For every question, compare complete answer inputs against raw notes and apply the qualified profile only when its input is cheaper. Show the fallback reason otherwise. The report also compares full workflow totals, including extraction.
6. Answer multiple questions in one request when they are known together. Keep one-at-a-time question support and count each repeated memory input. Provide action items and a fact-bounded summary from the same memory.

## Evaluation data and gates

The old 30-meeting synthetic set and its test split have been used for iterative research and may be used only as development diagnostics for V2. Create a fresh, meeting-disjoint V2 corpus from [AMI meeting transcripts](https://groups.inf.ed.ac.uk/ami/download/), whose official download page lists manual transcriptions and a CC BY 4.0 license. Curate 36 speaker-labeled excerpts of 400–1,500 target-model tokens: 12 development, 12 validation, 12 test meetings, with ten manually checked questions each. Record source meeting ID, excerpt provenance, literal fact fields, and accepted answers. Include explicit and absent decisions, ownership, date modifiers, negation, agreement, disagreement, blockers, statuses, and open questions. Do not place excerpts from one source meeting in different splits. A second human checks all validation and test labels before model evaluation. Retain the source attribution in the dataset.

The initial `τ = 99%` is an **observed** answer-comprehension gate, not a statistical guarantee. On gold facts, a profile must answer at least 119/120 validation questions correctly, create zero additional critical-field errors versus raw facts, and have no integrity failures. It must also reduce total measured answer input tokens versus both raw notes and compact English on validation. The frozen profile must meet the same gates on the 120-question test. Report extraction-to-answer accuracy and total workflow tokens separately; extraction errors cannot be attributed to the encoder. If no profile passes, V2 has no qualified encoding and uses raw notes.

Measure the complete request, including instructions, question, output schema, and API framing. Use local `tiktoken` counts for free screening and API `usage.input_tokens` for finalist comparisons. For `N` questions about one meeting, report extraction input and output once plus all `N` answering inputs and outputs. Report Bear-2 provider usage and local LLMLingua-2 time separately. Plot observed exact accuracy against actual OpenAI input tokens and mark nondominated methods. Include raw notes, compact English, LLMLingua-2, Bear-2, Tokenese V2, and V2 compressed by Bear-2 when its key is available. Optional baselines without credentials are marked unavailable, never scored as zero.

The V1 batching experiment found the same question-level answers with about two-thirds fewer input tokens when four known questions shared one request. V2 must report both single-question and known-question batch workloads, since batching is the current measured gain and a compiled memory by itself is not an API token saving.

## Search limits and failure behavior

Generate at most 512 versioned profiles locally. Deduplicate exact rendered text. Rank by full prompt token count, preserving representation families, and run a fixed small development probe before full validation. Cache every model call by model, prompt, and output schema. Stop after three candidate revision rounds or the recorded API-call budget. Never tune from the held-out test.

Unknown model/tokenizer, invalid source quote, missing key, unsafe alias, candidate that changes a critical field, model refusal, and failed compression call all produce explicit rejected or unavailable records. The compiler never silently promotes a proposal to a decision. All benchmark artifacts contain only bundled public examples; pasted user meetings stay in session memory.
