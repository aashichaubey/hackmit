# Proposed revision to Tokenese V2

> Direction update: the user prioritized novel language discovery over conventional optimization. Follow the [evolution and semantic repair plan](</Users/vasu/code/HackMIT 2026/docs/superpowers/plans/2026-09-20-tokenese-evolution.md>) for implementation; this document remains cost-accounting and baseline guidance.

Status: proposed after research; no implementation or qualification claimed. Read alongside the [research review](/Users/vasu/code/HackMIT%202026/reports/v2_cost_strategy_review.md). The original V2 plan remains the reference for its encoding experiment.

**Objective:** minimize measured product cost per correctly completed meeting task, preserving the critical-fact gates and reporting actual input reduction separately.

## Build order

1. **Add a cost ledger before choosing algorithms.** Extend the usage/report contract with provider, model, service tier, input, cached input, output, actual-call/replay status, latency, compressor charges, and fallback links. Preserve raw provider usage. Price calls with versioned rates and distinguish product simulation from research invoices. Keep existing historical reports readable.
2. **Run a development pilot on representative lengths.** Use existing short cases plus public longer meeting examples. Add exact questions covering correction, ownership, deadline modifiers, explicit negatives, missing answers, and multi-span evidence. Keep pilot findings exploratory. Maintain the independent validation/test requirement for release claims.
3. **Ship the already measured batching path into the experiment.** Use the existing four-question protocol as a baseline. Validate answer IDs/counts. Do not confuse a shared-context call with the asynchronous Batch API.
4. **Index source turns locally and retrieve evidence.** Retain speaker, order, source IDs, and text offsets. Start with lexical selection and neighboring turns; use optional local embeddings only if they improve evidence recall. Test budgeted evidence windows and a full-source fallback. No paid full-transcript extraction is required for this route.
5. **Add supported direct lookups to compiled memory.** Reuse V2's fact extraction only for the route/workload that needs it. Resolve owner/deadline requests deterministically when intent and entity are explicit; return citations. Route ambiguity, missing extraction, conflicting facts, and unsupported question types to source evidence. Measure ingestion cost and lookup coverage.
6. **Compare protected compression with retrieval alone.** Freeze Bear settings on development. Protect complete critical clauses, validate preserved content, cache by content/model/settings/protection version, and keep an uncompressed route. Compare existing local LLMLingua-2 fairly. Charge all compression attempts. Use one setting at runtime, not a three-setting search per new document.
7. **Measure reuse and choose a simple policy.** Compare cold/warm raw prompts, stable compact memory, dynamic retrieval, and exact answer caching at one, four, and ten questions per meeting. Select routes by projected dollars under measured cache behavior. Treat unknown cache hits conservatively. Add at most one cheaper-model experiment after the input strategy is isolated.
8. **Run the encoding search only if there is remaining headroom.** Start with compact English and explicit labeled/table formats. Count all headers, aliases, and schema. Expand into the original 480-profile catalog only if a simpler representation shows meaningful savings and comprehension. Qualification belongs to each model/renderer pair.
9. **Freeze and test the complete product path.** Freeze extraction, retrieval, protection rules, renderer, model, routing, thresholds, prompts, and cache policy before the held-out run. Apply the end-to-end quality gate, not only gold-fact comprehension. Publish failures and unavailable costs.
10. **Expose an auditable savings demo.** Show the question, supporting source, selected route, cached versus uncached tokens, billed/estimated cost, ingestion amortization, and fallback. Compare identical workloads with raw and the strongest simple baseline.

## Amendments to existing tasks

| Original task | Required change |
| --- | --- |
| 1–2: corpus | Add a pilot milestone; complete independent review before qualification. Keep raw transcripts and evidence labels separate from gold answer fields. Ensure source-disjointness across imported datasets. |
| 3: extraction | Use explicit turn/span provenance. Treat quote membership as a lexical check, not semantic validation. Support multiple facts sharing one span and flag ambiguous repeated quotes. |
| 4–5: candidates | Move behind retrieval experiments. Treat aliases as requiring evidence in the actual compressed request. Screen tokens locally but rank qualified workflows by dollars. |
| 6: qualification | Compare critical errors against raw and English by stable question IDs. Add separate encoder-qualified and product-qualified statuses. |
| 7: memory | Compile lazily or for a declared repeated-use workload. Empty extraction triggers evidence/raw fallback. Add deterministic supported lookups and exact answer caching with version invalidation. |
| 8: benchmark | Add retrieval, direct lookup, protected compression, cold/warm caches, and per-method ingestion charges. Avoid counting cached-token subsets twice or charging unused extraction to a route. |
| 9–10: UI/release | Display full-workflow cost and observed quality. Preserve the raw fallback even when a renderer qualifies. Do not imply universal 99% accuracy or guaranteed savings. |

## Focused checks

- Correct source for repeated text; speaker and temporal attribution; later corrections; intentionally shared evidence.
- Retrieval retains required evidence and handles global/exhaustive questions; empty extraction never proves absence.
- Direct lookup declines ambiguous intent and entity matches; missing fact does not turn into `no`.
- Protected compression preserves whole critical clauses, otherwise falls back.
- Batch slots match questions; exact answer cache invalidates on meeting/policy updates.
- Usage distinguishes provider cached inputs, product cache hits, and replayed research calls; retries and all ingestion are charged once where applicable.
- Product gate includes extraction/retrieval failures and wrong abstentions. Matching is by IDs, not row order.

**Stop rule:** when a simpler route meets the declared quality gate and the next optimization cannot demonstrate meaningful additional workload savings, spend the remaining hackathon time on the working product and its evidence. Do not make the demo depend on discovering a superior symbolic language.
