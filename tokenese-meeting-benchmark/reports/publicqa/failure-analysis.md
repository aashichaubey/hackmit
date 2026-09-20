# Public QA development failure analysis

This is an independent agent's qualitative source review, **not human adjudication**. It examines saved training outputs and released training contexts only. No labels, native scores, or qualification gates were changed; no model calls were made. The user now permits a modest quality tradeoff, but that preference does not retroactively turn these development results into qualification evidence.

The minimal-notation pilot has **four negative F1 pairs**, not five. The other pairs tie or improve. Two negative pairs show substantive retrieval/answerability regressions; two primarily expose quote-boundary or reference-scope sensitivity. Exact reconstruction preserves the source in all these cases: the failures concern model use of that source, not lost transcript characters. These are single paired generations, so causation and stability remain uncertain.

| Case ID (MeeQA) | Raw → encoded F1 | Source-supported assessment |
|---|---:|---|
| `5ce50fb2d7e083d23017` | 0.8000 → 0 | Genuine answerability regression. The question asks whether everyone is happy; explicit agreement opens the context and remains available in encoded text. Raw retrieves it; encoded abstains. This is also the sole negative pair in the four-question inheritance diagnostic. |
| `13cb5a6a539e8c76484d` | 0.5669 → 0.3289 | Genuine relevance/coverage regression. Asked what to observe, raw retrieves the opening observation instructions; encoded retrieves a later anecdote about a demo choosing the wrong activity. Neither output fully covers the instructions. The gold's admission-fee suggestion is subsequently retracted, so gold overlap does not measure final-state correctness. |
| `75e2680271e43f214223` | 0.1707 → 0.0645 | Reference-scope mismatch. The question concerns colour count, but the gold refers to three physical components. Encoded quotes the later explicit two-colour limit and agreement, which is substantively responsive. The source then defers further colour discussion; this is not evidence of a finalized design. Lower F1 here does not establish a worse substantive answer. |
| `cb0a8ae0d6644296a278` | 0.9032 → 0.8358 | Quote-boundary precision difference. Both outputs contain the same core confirmation and Linux-box evidence. Encoded adds an earlier acknowledgement. The extra span lowers lexical precision without demonstrating a changed factual answer. |

Short source anchors: the approval response says “yes, that seems good”; the colour discussion says “Not more than two colours.” These support the qualitative distinctions above, not replacement labels.

The remaining inheritance cases clarify the task's limits:

- `4e92c19c3c0a66b07473`: published gold is unanswerable, while both outputs quote a staff member saying their view has not changed. This is a plausible individual response, but does not necessarily establish the collective staff position or resolve the missing antecedent. Both native scores remain zero; no compression-specific regression is shown.
- `449bd704fba5685849ec`: published gold is unanswerable. Both outputs retrieve screen-function discussion containing explicit guesses and an admission that marketing did not specify the purpose. The outputs preserve uncertainty as quotations, but `found=true` can overstate whether a settled answer exists. Encoded also returns a broader block. This is a shared answerability/epistemic-status problem, not demonstrated source loss.
- `88dc16909004c36cc438`: published gold is answerable, with a brief affirmative reference and a much longer alternative. The question is elliptical and transcription-disfluent; both outputs abstain. Nearby negative-baseline discussion offers contextual support, but neither method resolves the question. This shared miss is not evidence against inheritance specifically.

## Next engineering experiment

Keep original F1, answerability, grounding, token savings, and cost visible. Add this qualitative taxonomy beside them, explicitly labeled agent review; never replace native grades with this analysis. For a user-selected small-loss operating point, show each candidate on a savings-versus-quality curve and state the observed loss, rather than calling it equivalent.

First test repeatability on the two genuine regressions using a fixed, balanced raw/encoded repeat schedule under the authorized ledger. Repeat raw as well as encoded, because a single deterministic-temperature response is not a stable quality estimate. Preserve prompts, output limits, labels, and exact reconstruction. Before further broad candidate search, separately diagnose answer content versus evidence-span selection: the native extractive task rewards annotation overlap, while general-purpose usefulness also needs corrected decisions, uncertainty, and relevant answers. Any additional semantic diagnostic must be separately named and predeclared, with original scores retained.
