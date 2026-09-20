# Public meeting-QA successor experiment

The user supplied MeetingQA, MeeQA and MISeD to replace the missing trustworthy evaluation target. Work resumes using these published annotations. The old frozen experiment and its failed results remain historical; the old annotation-review packet is no longer a prerequisite for this successor study.

Sources are pinned to repository commits in `data/publicqa/sources/index.json`. Downloaded data remain local and are excluded from version control. MeetingQA includes a CC BY-NC-SA 4.0 license; no top-level license was found in the pinned MeeQA or MISeD repository trees. This work uses local research copies and does not publish their datasets. Retain author attribution and source provenance.

## Tasks and evidence

* MeetingQA: participant questions with human extractive answer annotations, including unanswerable cases. Native context windows and labels are retained. Invalid answer offsets are quarantined, never silently repaired.
* MeeQA: natural participant questions, multi-span answers and unanswerable cases. Retain all annotated spans as jointly required evidence, rather than treating spans as interchangeable answer aliases. Preserve question context and published question-quality flags.
* MISeD: semi-automatically generated, human-verified information-seeking dialogues with attributed transcript segments. Preserve history and attribution; never score its summaries as exact short-answer labels. The fully manual WOZ subset is a separate follow-up benchmark, not training material.

These datasets share AMI/ICSI meetings. Published split names across different repositories do not guarantee disjointness. Build canonical meeting/family identities and prevent a meeting family used in development from entering successor qualification in another dataset. Initial schema inspection and development use training files only. No held-out labels are opened for candidate tuning.

## Initial representation and metrics, before model calls

The initial successor compiler classifies transcript framing, speaker identifiers and utterance bodies. It preserves **all** utterance words, order and turn multiplicity, rather than selecting five fact kinds. It may shorten structural speaker labels and inherit a speaker across consecutive turns. No model extraction, grammar legend, retrieval or question-dependent selection is used. This extends the original grammar approach to broad transcript content. Exact reconstruction and token non-expansion are independent integrity checks, not comprehension claims.

The extractive tracks use one identical quote-extraction instruction and structured span-answer schema for raw and encoded inputs. This is a new native task, not a rerun of the old short-answer benchmark. Report answerability precision/recall/balanced accuracy; macro question and macro meeting text-overlap F1; exact match; ungrounded output; per-dataset and answerable/unanswerable strata; paired changes; and actual input/output/cost. F1 is not used as a substitute for exact integrity or answerability. The initial metric is labeled as our Unicode token-overlap implementation, not an exact reproduction of published leaderboard scores.

First pilot: deterministic, label-stratified training sample of 16 questions per extractive dataset (eight answerable and eight unanswerable), with a stable hash ordering and at most two questions per meeting. No selection based on model outcomes. Retain exclusions and sample IDs. Compare raw and one fixed transcript grammar at temperature zero. All API calls share the existing ledger and remaining limits; replays preserve historical usage and do not count as new spending. The development candidate calls count toward discovery. MISeD is initially integrated and screened locally; its dialogue generation and attribution benchmark is separate.

Pilot advancement requires exact reconstruction, lower aggregate actual input tokens, no decrease in macro F1 or balanced answerability, and no increase in ungrounded answers. These are advancement criteria, not a general-purpose qualification certificate. An eventual qualified release requires a frozen successor and a sufficiently sized fresh, source-disjoint evaluation with uncertainty and explicit workload scope. Do not silently apply the old 119/120 short-field threshold to a different human multi-span extraction task, or loosen criteria after seeing results.

## Development iteration 2

The first `@N:` candidate failed the predeclared quality gates. Its compact symbol removed an explicit speaker cue, and native MeeQA questions sometimes themselves contain original speaker labels. Candidate 2 retains the familiar word `Speaker`, shortening only MeeQA's verbose `& SPEAKER_N:` framing to `Speaker N:`; MeetingQA stays raw. The questions, reference labels, sample, instruction, output limit and metrics stay unchanged. Successful baseline calls are replayed. The one baseline parsing failure may be retried under the existing ledger; its original unknown usage remains recorded. This development iteration is not independent confirmation.

## Development iteration 3

The readable-label revision also failed. The third and final initial development candidate removes only the decorative `& ` before MeeQA speaker labels, preserving their exact `SPEAKER_N:` spelling. Other framing remains raw. A raw response repeatedly exceeded the 512-token answer cap; this iteration increases **both** arms to 2048 and obtains new matched baseline calls, rather than granting the candidate extra output capacity. This changes the experiment configuration and must not be presented as a direct single-variable comparison against previous pilots. Baselines finish before candidate calls so unchanged contexts share their exact response. All quality gates remain unchanged.

## Separate dialogue baseline

Before calls, fix an eight-turn MISeD **raw-only** training baseline: four first questions and four questions with preceding reference dialogue, stable hash order, one turn per meeting family. Supply every transcript segment with its original speaker name and zero-based index, without retrieval or truncation. Score Unicode response token overlap and citation-set precision/recall/F1/IoU against the published attribution; interpret start/end indices inclusively (the dataset contains singleton ranges with equal start/end). Malformed or omitted bounds stay quarantined. Lexical overlap is not a faithfulness judgment, and citation overlap does not prove entailment. This separate baseline develops the dialogue evaluation, not a compressed-candidate qualification. No WOZ or held-out labels are inspected.

## Parallel local-search branch

A separate, independently implemented grammar keeps the first/changed speaker label verbatim and writes `↳` for a consecutive turn by the same speaker. Local screening found 7.26% MeeQA context savings with exact reconstruction. Before model calls, fix a four-question development diagnostic to the first four cases in the existing deterministic MeeQA pilot ordering (two answerable, two unanswerable); reuse matching successful raw calls with the same 2048 output cap. This consumes the remaining four discovery calls. It is explicitly smaller than the initial pilot and cannot satisfy its advancement or release requirements. Full quoted spans containing inherited labels must be aligned to the full encoded context to restore the preceding speaker; ambiguous or ungrounded quotations fail instead of selecting a gold-favorable interpretation. No grammar edits follow these four answers within this branch.

## User-directed exploratory quality tradeoffs

The user subsequently clarified that a slight quality decrease is acceptable and that curiosity and experimentation are the objective. Keep the original zero-loss gate results historical, but do not use them to suppress new experiments. Display paired F1/answerability loss, regressions, grounding and uncertainty beside savings. A configurable UI loss tolerance is an exploration preference, not a claim of factual equivalence or safety certification.

The next fixed block candidate uses familiar indented list structure with unchanged speaker headers. Evaluate the same 32 training questions, prompt/schema and 2048 output cap. Correct quotations already present in the original source are accepted without further expansion; otherwise map exact encoded quotations back to the full source context. This source-only restoration never consults gold labels. Continue displaying original native lexical scores rather than replacing them with the agent's qualitative error taxonomy.

The original plan explicitly describes its call ceilings as implementation defaults. Record one immutable successor discovery allocation of up to 48 calls from the remaining 179 overall calls, leaving at least 128 for paired final evaluation. Preserve the existing 3,500-call/$10 global limits, original 1,500-call discovery counter, old manifests, calls and costs. The new study has an explicit separately capped purpose; restarting cannot enlarge it. This is a disclosed allocation following the user's instruction to continue exploring, not a reset of spent budget.

## Frozen exploratory test after the user's clarification

Select the minimal separator format using development results before opening the test file: its measured F1 loss is slightly smaller than blocks, it has no invalid structured answers, and its total workflow cost reduction is larger. Freeze compiler/scorer/adapter/runner-entry hashes, model, prompt/schema and output cap. Draw 64 MeeQA test logical questions (32 answerable, 32 unanswerable), stable hash order with at most four per meeting, excluding every meeting family loaded in any training dataset. If that sample cannot be formed, report the shortfall without substituting an easier sample. Use original test references and the existing native scorer.

Predeclare a five-percentage-point loss tolerance for aggregate F1 and balanced answerability as an exploratory default reflecting the user's allowance for slight degradation; show actual losses, grounding, invalid outputs, paired changes and family-bootstrap uncertainty regardless. This is a new user-directed tradeoff criterion, not a retroactive claim that old zero-loss gates passed. A point estimate within tolerance is not statistical proof of equivalence. No test failures may drive edits to this frozen candidate, and no other candidate may be selected using this test. The 128 paired requests remain within the original global limits.
