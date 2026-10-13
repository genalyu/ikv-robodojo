# RoboDojo grounded Qwen IKV

The policy checkpoints remain unchanged. DM05 uses 320 historical KV positions plus its dense current three-camera prefix; no latest-history patches are forcibly protected. PI05 uses the baseline current visual count (768 for this checkpoint). OpenWAM uses its clean observed-prefix count (120 in this deployment). Motion filtering is disabled.

All real native policy K/V rows are archived on CPU, with duplicate current-window IDs refreshed. Qwen selects the fixed GPU read set. Unselected CPU rows can re-enter. This differs from a permanently evicting recursive bank. Only selected rows absent from the prior GPU read set transfer again. Scoring/action failures do not commit a partially updated archive; the default per-stream CPU limit is 8 GiB.

## Scoring protocol: grounded_independent_v1

1. Encode each new RGB image once into immutable native Qwen vision merger features. Historical requests reuse those features, not historical RGB decoding or re-encoding.
2. Qwen localizes objects, markings, and background from the cached features. Validated label/box records are cached with a model/prompt signature. Greedy generation uses repetition penalty 1.1. Failed localization gets a frame-seeded sampling retry (temperature 0.7, top-p 0.8, top-k 20), then one format-constrained retry. Strict validation rejects repeated identical boxes, invalid coordinates, and truncated output; no partial detections are accepted. Version 5 invalidates earlier grounding caches. These guards do not detect every hallucinated object.
3. Deterministic box intersection associates detected evidence with each actual Sys1 patch. Coverage is input metadata, never a formula that substitutes for the Qwen score.
4. Identical frames share one feature packet but retain every observation time. Full historical native features, localized regions, task, and times produce a short task analysis. This helps keep identity bindings consistent between patch questions.
5. Each patch is assessed independently and produces a short evidence description plus an integer 0-100 score. This replaces the long score-vector generation that frequently collapsed to a constant value.
6. The public response retains exact candidate IDs and complete modality coverage. No heuristic score fallback or forced score spread is used.

Active prompts are patch_score_prompt.txt and grounding_prompt.txt. qwen_patch_scorer.py also contains the task-analysis and individual-patch question templates. The old sys2_prompt.txt / compact-vector implementation remains for comparison; the service entry point selects the new scorer.

Qwen sees task and visual evidence, not Sys1 K/V values or action-expert attention. Its score estimates semantic utility; it is not a measured causal action improvement. Localization can miss objects or shift boundaries, and the resulting patch association can be wrong. A high score is not a success probability.

## Computation and state

Inside one score request, the full Qwen prefix is computed once after task analysis. Each patch branches from it: dynamic attention prefix tensors stay immutable, and mutable linear-attention convolution/recurrent states are cloned. Exact duplicate patch queries reuse their validated result within that request. No Qwen decoder cache survives the request. Native vision features and immutable localization records persist.

Current clients batch up to 128 candidate positions but provide the same complete historical context to each batch. Candidate order is canonicalized; other patch answers are never included in a patch's prompt. All-fit policy banks retain every row and skip unused scoring.

The last-token vocabulary projection (logits_to_keep=1) and tokenizer-aligned EOS are retained. Context overflow raises an error rather than dropping old evidence. BF16 full-prefill and branched-prefill computations are not guaranteed bitwise identical; an earlier controlled probe observed a 10-point score difference; the latest four-patch probe matched exactly. Cache-state isolation is tested independently.

Validated runtime: PyTorch 2.7.1 CUDA 12.6, Transformers 5.18.0, and requirements-sys2-format.txt. /health exposes the protocol and scorer revision. patch_scores_grounded_v1.jsonl records task analysis, per-patch evidence/score, context IDs, revision, timings, and unique-query counts.

## Evidence and limits

Controlled two-frame occlusion checks cover 4x4, 16x16, and 12x10 grids. They require retention of the historical target, preference for its current associated cover, and low background scores. Input reversal/partition checks and native hybrid-cache isolation are separate checks. A stricter historical target-vs-distractor margin is recorded separately: the final controlled cases score the historical target 95, distractors 10-15, and tested backgrounds 0. Earlier revisions gave a coarse-grid distractor 80 or even tied both colors at 95. These are limited fixtures, not calibrated object-importance accuracy.

These checks are not RoboDojo success rates. Real long-history rollouts and comparisons must use a new evaluation label and directory. Do not combine results produced by the prior vector scorer with this protocol. Synchronous scoring and first-use localization add latency.

## Validation snapshot (2026-10-13)

On the 4090 Qwen runtime, nine cache/grounding/timeline regression tests passed. The final runtime revision `abab87db7e7be7a911d97f8467ffd386840cad542c01ef59494a41867adcdd9f` passed an 80-frame, four-candidate complete-context stress request in 21.47 seconds with localization already cached. Initial cache construction on the preceding revision took 701.72 seconds. The 80-frame fixture mixes archived scenes and tests execution, not long-horizon semantic correctness. The real 640x480 cover-blocks fixture was visually checked: zero-based 4x4 object patches are 9 and 10, not 5 and 6. This corrected a prior test annotation. Final controlled patch tests passed; no formal RoboDojo success improvement has yet been established.
