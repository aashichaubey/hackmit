# Tokenese V2: cost strategy review

Research date: September 20, 2026. Status: research and proposed revisions; no new paid model evaluations performed.

**Recommendation: keep V2's evaluation discipline, but make evidence selection and avoiding unnecessary inference the main experiment. Treat legend-free encoding as one optional optimization.** The current plan is a reasonable encoding experiment; it does not yet establish the cheapest meeting assistant for The Token Company's challenge.

This review uses the track description supplied by the user. It makes no prediction about judging outcomes. Architectural recommendations below are hypotheses to test, not measured Tokenese gains.

## What the project already proves

The existing [research report](/Users/vasu/code/HackMIT%202026/reports/input_token_research.md) says meeting content averages only 24.7 of 169.8 input tokens per raw request. Most cost on that corpus comes from repeated instructions, questions, schema, and framing. Deleting all meeting content would remove only about 14.5% of input, while destroying its utility.

I recomputed the saved batched rows: [main](/Users/vasu/code/HackMIT%202026/reports/batched_questions.json) contains 30 calls, 6,990 input tokens, 1,203 output tokens, and 119/120 exact answers; [stress](/Users/vasu/code/HackMIT%202026/reports/batched_stress.json) contains six calls, 1,482 input tokens, 241 output tokens, and 22/24 exact answers. Against the recorded separate-call baselines of 20,370 and 4,434 input tokens, these are 65.7% and 66.6% input reductions. They are exploratory results on the existing corpus, not a fresh holdout.

The [V1 results](/Users/vasu/code/HackMIT%202026/reports/README.md) also document increased costs for fact extraction plus symbolic encoding, deadline-modifier losses with Bear-2, and high embedding similarity for meaningfully corrupted facts. These are stronger project-specific evidence than any external compression benchmark.

## Cross-check of the current plan

Reviewed the [implementation plan](/Users/vasu/code/HackMIT%202026/docs/superpowers/plans/2026-09-20-tokenese-v2.md), [design](/Users/vasu/code/HackMIT%202026/docs/superpowers/specs/2026-09-20-tokenese-v2-design.md), and existing fact, model, cache, tokenizer, and compression adapters.

| Finding | Why it matters | Proposed correction |
| --- | --- | --- |
| Tasks 4–6 optimize a whole-meeting rendering before testing query-specific selection | Small syntax savings may be dominated by excluding irrelevant facts | Benchmark local evidence selection first; keep the 480-profile search optional |
| Tasks 5–8 rank input tokens rather than billed dollars | Cached inputs, extraction outputs, compressors, and retries have different prices | Maintain token and dollar ledgers; choose the cheapest qualified route for the workload |
| Task 6 qualifies gold-fact comprehension while Task 10 only reports extraction misses | A perfect encoder can still power an unreliable product | Retain an encoder gate and add a separate end-to-end product gate |
| Task 7 expects empty facts to produce `not found` | Empty extraction can be a failure, not evidence that the transcript lacks an answer | Retrieve original evidence or use the raw fallback when extraction is empty or incomplete |
| Task 3 validates quotes by substring and cursor search | A quote's presence does not establish correct attribution; fallback can reuse the first occurrence | Give source turns stable IDs; represent explicit spans, ambiguity, and intentional shared evidence |
| `qualifies` checks critical regressions against raw rows only, using positional `zip` | It can miss a new failure relative to English, or compare mismatched rows | Join by case/question ID and compare against both baselines |
| Tasks 1–2 require 360 new questions and a second human before the core experiment | Annotation can consume the hackathon before testing the highest-value hypothesis | Start with a clearly labeled development pilot, then complete the independent release evaluation; never fabricate second-review status |
| Excerpts are limited to 400–1,500 tokens | They do not demonstrate full-meeting scaling or the economics of repeated long prefixes | Add real longer transcripts, scattered evidence, and repeated-question workloads |
| Task 8 reuses research compression adapters | Bear currently tries three settings; both adapters use the V1 prompt for targeting | Freeze a setting on development, charge every actual trial, and use the benchmark's actual prompt/schema |
| Existing `Usage` and `CachedClient` retain only input/output totals | They cannot distinguish provider cache savings or a replayed research result from a new invoice | Preserve usage details and separate logical workload cost, actual experiment spend, and product answer-cache hits |

Keep the pinned reference model, public-data provenance, local tokenizer screening, separate extraction/encoding diagnoses, explicit unavailable baselines, and a genuinely untouched final test. The existing statement that 119/120 is an observed gate rather than a statistical guarantee is correct. Questions within one meeting are correlated; report meeting-level variation as well.

## Research implications and alternative techniques

### 1. Select evidence before rewriting its syntax

LongLLMLingua combines question-aware selection, context ordering, and variable compression. Its paper also warns that question-specific recompression adds overhead and prevents direct reuse of one compressed context across different questions. The transferable lesson is to evaluate relevance and reuse together, not copy its benchmark percentages. [Paper](https://aclanthology.org/2024.acl-long.91.pdf).

The rate-distortion work formalizes query-aware versus query-agnostic compression and finds advantages from conditioning on the question. Its synthetic and small natural-language experiments do not prove performance on this product. [Paper](https://arxiv.org/html/2407.15504v1).

**Proposed implementation:** split transcripts into speaker turns with local offsets; use lexical retrieval over turns, optionally a local embedding index; select whole evidence spans plus needed neighboring turns. Return to raw source material for details the eight-kind fact schema does not represent. Try context budgets such as 256, 512, and 1,024 target-model tokens, chosen on development data. These are experimental settings, not recommended universal thresholds.

Keep related corrections, disagreements, and later revisions together. For example, retrieving “Maya will ship Friday” without “Actually, Raj takes it, next Friday” is cheap but wrong. Chronology and ownership relationships matter more than retaining a bag of names and dates. Global questions such as “list every blocker” need exhaustive relevant coverage, not top-k retrieval.

### 2. Answer supported lookups without another LLM call

**Proposed product technique:** once an action record is available, an explicit UI operation such as `owner(task_id)` or `deadline(task_id)` can read its fields and display its source citation. No answer-generation call is necessary. A recognized free-text question can use the same path only if entity resolution and intent are unambiguous. Otherwise, send evidence to the model.

This has zero *additional* model tokens for that answer, but the ingestion/extraction cost remains. It must not depend on gold facts or hardcoded benchmark questions. Preserve conflicting owners and ambiguous dates instead of silently selecting one. A missing field does not prove an explicit negative.

The creative opportunity is a meeting representation that serves both ordinary programmatic queries and LLM evidence requests. It extends V2's existing action-items view into reusable product functionality.

### 3. Use exact answer caching before semantic answer caching

**Proposed ordering:** cache exact requests by meeting content/version, question, prompt/policy version, and model where applicable. Repeated requests can reuse their answer and evidence. Distinguish this from the existing research cache, which avoids rebilling experiments but replays historical usage.

Semantic caching is a later experiment. Similar strings can ask about proposals versus decisions, this Friday versus next Friday, or current versus original ownership. The project's existing embedding corruption tests already show why similarity alone is insufficient. First canonicalize a small set of unambiguous supported query intents. Keep arbitrary semantic matches out of the default route until false-hit rates are measured.

### 4. Combine retrieval with protected compression

LLMLingua-2 treats compression as token classification and was distilled using MeetingBank. It is a relevant existing local baseline, but its paper acknowledges training-domain limitations. Local execution removes a paid compression API call, not compute or latency costs. [Paper](https://arxiv.org/html/2403.12968v2).

The Token Company's current documentation supports `protect()` / `<ttc_safe>` to retain selected text. The example output omits the protection tags. Its documentation describes Bear-2 as deterministic for identical input and aggressiveness, making compressed-result caching useful. [Protection](https://thetokencompany.com/docs/protect-text), [compression](https://thetokencompany.com/docs/compression).

**Proposed experiment:** retrieve evidence, protect complete clauses carrying ownership, dates, negation, and decision status, and compress the remaining prose. Protecting the word “not” alone cannot preserve what it negates. Recheck protected spans after compression; fall back to uncompressed evidence if preservation fails. Compare this against retrieval alone and protected compression of the whole transcript. Compressing already terse fact rows a second time may offer little headroom.

The public pricing page currently charges by tokens removed but publishes no numeric unit rate. Report Bear token metrics and actual account charges separately; do not invent a rate or treat hackathon credits as the recurring product price. [Pricing](https://thetokencompany.com/pricing).

### 5. Make context budgeting aware of provider caching

OpenAI caches matching prefixes; usage reports cached tokens. Current documentation distinguishes model generations and says earlier-model minimum cache lengths depend on request settings. Do not hardcode a universal 1,024-token rule or apply newer cache-write behavior to the pinned GPT-4.1-mini. Measure actual cache hits with the real schema. Keep reusable content before changing questions. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).

For GPT-4.1-mini, published rates are $0.40/million uncached input, $0.10/million cached input, and $1.60/million output. Consequently, a hypothetical fully cached 4,000-token context costs $0.0004, while an uncached 1,500-token context costs $0.0006. This comparison excludes the initial cache fill and other request tokens. Fewer tokens can cost more. [Model rates](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

**Proposed comparison:** dynamic evidence windows versus a stable compact meeting memory versus stable topic sections. Do not pad a prompt or retain irrelevant text just to chase a cache threshold without measured total savings. Cache hit rate and expected reuse belong in route selection.

### 6. Use batching and model routing for different workloads

Keep the project's measured four-question batching win. For many known questions, compare a shared context with the deduplicated union of their evidence; split unrelated questions if that union becomes too large. Account for output size and answer-slot correctness.

OpenAI's asynchronous Batch API is a different mechanism: it offers a 50% discount with completion within 24 hours. It is relevant to deferred product ingestion or reports, not immediate meeting chat. Saving on evaluation alone does not demonstrate the challenge's intended product savings. [Batch API](https://developers.openai.com/api/docs/guides/batch).

RouteLLM supplies research evidence for routing between models under quality constraints. It does not validate a router for meeting negation and deadlines. Add one available cheaper-model candidate after the context strategy works, and evaluate its full cost including failures and fallback calls. [RouteLLM](https://arxiv.org/abs/2406.18665). GPT-5 nano is a candidate to investigate, not an automatic winner; verify reasoning/output usage and snapshot availability before running it. [Model documentation](https://developers.openai.com/api/docs/models/gpt-5-nano).

### 7. Try adaptive expansion, with measured failure costs

**Proposed technique:** begin with a small evidence set and expand only when the route cannot support the question, rather than always sending the full meeting. Prefer local signals such as unresolved references, competing owners, missing required fields, and out-of-scope question types. A model's confidence alone is not a sufficiency certificate.

Research on sufficient context distinguishes retrieval failure from failure to use available evidence and shows that models can answer incorrectly when context is insufficient. This supports separate retrieval-sufficiency evaluation and abstention coverage reporting. [Paper](https://arxiv.org/abs/2411.06037).

If a small attempt costs `Cs`, a full fallback costs `Cf`, and fallback happens with probability `f`, expected cost is `Cs + f*Cf`. It beats always using the full route only if `Cs < (1-f)*Cf`, assuming equivalent quality. Charge failed first attempts. Freeze thresholds using development data, not held-out answers.

### 8. Keep representation experiments small and interpretable

Evaluate compact English, explicit field labels, and shared table headers on the same facts. Columnar records can amortize repeated field names across many actions; dictionaries can amortize repeated long entities. Include headers, dictionaries, and question/output reconstruction in the token count. A dictionary is worthwhile only when repeated savings exceed its introduction cost and comprehension remains acceptable.

A source-attested abbreviation is not necessarily understood by the answer model if the alias definition is omitted from its input. Retain the full term unless the compressed request itself supports the alias mapping. The same caution applies to multilingual labels and cryptic symbols. The current no-legend rule is a useful experimental condition, not an objective worth sacrificing quality for.

As a separate experiment, have the model select a fact/span ID and let code copy an exact owner/date from the source. This might reduce output and prevent transcription changes, but it still needs correct fact selection and unknown handling. The existing failed schema-shortening experiments mean it needs its own validation.

### Techniques to defer

Gist tokens, soft-prompt autoencoders, custom tokenizers, and KV-cache engineering are interesting for a controlled open-model serving stack. They are not drop-in textual encodings for this hosted API project. Gisting requires training with changed attention behavior; distinguish its compute benefits from billed input savings. [Gist paper](https://arxiv.org/abs/2304.08467), [compression survey](https://aclanthology.org/2025.naacl-long.368/).

Do not prioritize gzip/base64 payloads, translation, screenshot-based text encoding, or agent chains that repeatedly summarize one another. There is no evidence here that their decoding, fidelity, and extra-call costs beat ordinary evidence selection. Incremental compilation of new transcript turns is a useful later extension, but must revisit earlier records when later turns correct them.

## Correct cost accounting

For the pinned mini model, using dollars per million tokens:

`call_cost = ((input_tokens - cached_tokens)*0.40 + cached_tokens*0.10 + output_tokens*1.60) / 1_000_000`

This formula is specific to its published standard rates. Other models or service tiers need their own tariff and any write/tool charges. Do not add `cached_tokens` to `input_tokens`: it is a subset. Record missing usage as unknown, not zero.

`workflow_cost = ingestion + extraction + indexing + compression + all_answer_calls + retries_and_fallbacks`

Show external API charges and local compute/time separately when local compute has no reliable dollar estimate. Preserve three ledgers: actual spend during the research run, reconstructed cost for the simulated product workload, and measured product cache behavior. Reusing a saved benchmark result must not masquerade as a live provider cache hit.

If compilation costs `E`, baseline answer cost is `B`, and compiled-route cost is `A`, strict savings begin at `floor(E/(B-A)) + 1` questions when `B>A`; no break-even exists otherwise. Cache hits and heterogeneous questions require summing actual per-call costs instead of assuming constant `B` and `A`.

**Illustrative calculation, not a benchmark:** at mini rates, ingesting 10,000 tokens and producing 1,000 extraction tokens costs $0.0056. Reducing each uncached answer's context from 10,000 to 600 tokens saves $0.00376 in input, giving strict break-even after two questions under equal answer-output costs. Against an idealized $0.001 fully cached raw context, per-answer savings shrink to $0.00076 and break-even becomes eight questions. Cold fills, schemas, and actual cache boundaries change both results.

## Evaluation that can support the demo

Use a small development pilot to eliminate bad approaches, then preserve the original independent validation/test intention. QMSum provides real meetings, queries, and relevant text spans that can accelerate retrieval evaluation. Its summarization references are not exact owner/deadline labels. Retain a manually checked critical-fact suite. Deduplicate AMI source meetings across QMSum and any added AMI corpus; where feasible, separate related meeting series too. [QMSum paper](https://aclanthology.org/2021.naacl-main.472/), [official data](https://github.com/Yale-LILY/QMSum).

Test brief notes and real longer transcripts, one question and repeated questions, cold and warm provider caches, exact repeats and novel questions, known batches, and at least one correction/contradiction case. Position relevant evidence at different transcript locations; long-context work documents position sensitivity, though its magnitude on this pinned model still needs measurement. [Lost in the Middle](https://arxiv.org/abs/2307.03172).

Primary comparisons: raw; batched raw; locally retrieved source spans; direct supported lookups; protected Bear-2; local LLMLingua-2; retrieval plus each compressor; V2 whole-fact memory; and the selected combined policy. Test one cheaper model only after these isolate the context effect. Do not launch the full Cartesian product.

Record exact answers, critical-field regressions, extraction recall, retrieval evidence coverage, abstentions on answerable questions, request counts, input/cached/output tokens, dollar cost, compression latency, and fallback frequency. Freeze the whole routing policy, not just the renderer. Oracle facts and oracle evidence are upper-bound diagnostics only.

For any product claim of the existing 99% observed gate, require 119/120 on the full deployed path as well as the encoder diagnostic, with no new critical errors against the declared baselines. If that is not achieved, publish the actual result and limit claims; do not silently lower the threshold or ignore questions the router abstains on.

My recommended hackathon demonstration is a cost receipt for each answer: source evidence, chosen route, actual model inputs, compile cost amortized over the session, cache hits, and any fallback. The strongest proposed story is: **Tokenese turns a meeting into reusable, auditable evidence and spends LLM tokens only where needed.** Whether this beats every simpler baseline remains an empirical question.

## Research method and limits

Reviewed local plans, implementation contracts, historical reports, and recomputed saved batching totals. Searched primary papers and official provider documentation covering query-aware compression, learned compression, retrieval sufficiency, routing, caching, pricing, batching, soft prompts, and meeting datasets. Deep-read relevant methods/limitations in LongLLMLingua, LLMLingua-2, the rate-distortion study, and the compression survey; inspected current provider pages directly.

Context7 was queried for OpenAI and The Token Company. Its Token Company results described older Bear-1/TokenClient APIs, so current first-party Bear-2 documentation took precedence. Some AMI download and arXiv HTML URLs failed to load; no new corpus or license audit was completed. Public documentation establishes capabilities, not realized savings. This review cannot establish the best-performing route without the proposed end-to-end experiment.
