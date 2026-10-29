# The Ghost Hunter's Lab — Scientific Evidence Report

**Evidence snapshot:** 29 October 2026  
**Research branch:** `feat/steering-validation`  
**Last reported research commit:** `5a766c4` (`feat: export verified steering evidence for publication`)  
**Status:** Activation-steering investigation completed at strength +16; general-capability preservation and weight editing not established.  
**Source policy:** This report documents results supplied in the project terminal transcript and the exporter contract. It is not an independent examination of the raw JSON files, which remain on the developer's machine.

## 1. Research question and scope

We opened a locally runnable, open-weight Gemma model and asked: **Can we intervene in an internal direction associated with ghost-related text, measure the intended effect against random controls, and observe what else changes?**

We tested **activation steering**, not a permanent rewrite of model weights. A direction extracted from a published sparse autoencoder (SAE) was added to the model's residual-stream activity during inference. The underlying model weights remained unchanged. This is neither evidence of artificial consciousness nor proof that a single feature contains an entire concept or thought.

## 2. Experimental components

| Component | Recorded configuration |
| --- | --- |
| Base model | Gemma 3 4B IT; Q4_K_M GGUF |
| Inference | Local `llama.cpp` server; reported Git revision `fae9ac561` in earlier project notes (confirm against exported runtime provenance before publication) |
| Interpretability resource | Gemma Scope 2 SAE, `resid_post/layer_22_width_65k_l0_medium` |
| SAE feature | 23871, published ghost/spirits-related description |
| Intervention | Decoder direction, 2,560 float32 components; GGUF tensor `direction.22` |
| Layer | 22 |
| Strength | +16 for frozen intervention evaluations; baseline 0 |
| Controls | Five independently seeded, norm-matched random directions, seeds 2026–2030 |
| Inference request | `n_predict=1`, `n_probs=1000`, `temperature=0`, seed 2026, `cache_prompt=false`, `post_sampling_probs=false` |
| Target token | Exact token `" ghost"` (leading space), ID 25556 |
| Comparator | Exact token `" man"` (leading space), ID 880 |

**Runtime mapping caveat:** The extracted SAE decoder direction was serialized for a llama.cpp intervention at layer 22. Numerical identity between the original SAE hook and this llama.cpp injection site has not been independently validated. We can describe the intervention actually executed without asserting perfect cross-runtime equivalence.

## 3. Pilot demonstration — not held-out evidence

Exploratory prompt: `The mysterious figure was a`.

| Condition | Generated text | P(exact token `" ghost"`) | Ghost-versus-man log preference |
| --- | --- | ---: | ---: |
| Baseline | `" man"` | 5.8628002% (later preflight) | −0.29022884 |
| Ghost SAE, +16 | `" man"` | 6.1314686% (later preflight) | −0.17979813 |
| Random, +16 | `" man"` | approximately 6.30% (earlier pilot) | approximately −0.2349 |

The generated next token remained `" man"`. The random intervention produced a **higher absolute probability** of `" ghost"` than the SAE intervention in this pilot. The SAE intervention produced the larger change in the **two-token relative preference**. These metrics answer different questions. No claim of ghost-specific success should be derived from the first row alone.

The six-prompt pilot and its 24/24 reproducibility regression were exploratory; neither should be merged numerically with the separately frozen dataset.

## 4. Held-out steering evaluation

**Dataset:** `eval/heldout_prompts.json` (20 unique prompts: eight neutral, six ambiguous, six supernatural). Created and committed before evaluation, but designed with prior knowledge of the candidate feature and exploratory results; it is **not** an independently sampled representative benchmark.  
**Dataset commit:** `e8b4f91`.  
**Dataset SHA-256:** `3fdc774a6274f77512694bb983ea30b7445179a4e31617e88fd5e7d448e2de73`.  
**Raw-result file:** `eval/results/heldout_20261029T100108969558Z.json` (local; Git-ignored).  
**Raw-result SHA-256:** `a1c7fce2d532e444236930adf0d2c738bea85dc476eec1f57cf506c56fe28910`.  
**Coverage:** 140/140 unique prompt-condition pairs, status `completed`; no missing target or comparator values in reported verification.

### Primary metric

For each matched prompt, **Δ log P(`" ghost"`) = intervention log probability − baseline log probability**. Positive means the exact next-token probability increased.

| Direction | Mean paired Δ log P | Prompts positive / 20 |
| --- | ---: | ---: |
| Ghost SAE +16 | **+0.016386944** | **13** |
| Random seed 2026 | −0.049365777 | 7 |
| Random seed 2027 | −0.006433243 | 8 |
| Random seed 2028 | −0.033734083 | 8 |
| Random seed 2029 | −0.030156749 | 6 |
| Random seed 2030 | −0.036793494 | 6 |

The SAE direction's **mean** exceeded all five sampled random-direction means. On only **6 of 20 individual prompts** did its primary-metric shift exceed all five random-control shifts. Its sample standard deviation across prompts was 0.121368, substantially larger than its mean 0.016387. The effect was small and variable.

### Secondary metric

Define **S = log P(`" ghost"`) − log P(`" man"`)**, then **ΔS = S(intervention) − S(baseline)**. This is a comparison of two exact tokens, not a general ghost-concept score.

| Direction | Mean paired ΔS | Prompts positive / 20 |
| --- | ---: | ---: |
| Ghost SAE +16 | **+0.042218480** | **11** |
| Random seed 2026 | −0.008196467 | 8 |
| Random seed 2027 | +0.025101516 | 9 |
| Random seed 2028 | −0.043899137 | 8 |
| Random seed 2029 | −0.010002196 | 8 |
| Random seed 2030 | −0.035859296 | 9 |

The SAE mean exceeded all five random means on this metric, too. On only **5 of 20 prompts** did its secondary-metric shift exceed every sampled control. Its standard deviation was 0.170347: again, appreciable variability. The two measures must remain visually separate.

### Descriptive category breakdown — primary metric only

| Category | Number of prompts | SAE mean Δ log P | SAE positive |
| --- | ---: | ---: | ---: |
| Neutral | 8 | +0.005712688 | 5 |
| Ambiguous | 6 | +0.040823877 | 4 |
| Supernatural | 6 | +0.006182353 | 4 |

These are small hand-selected subgroups, not evidence of a reliable category interaction. Do **not** claim that ambiguity causes stronger steering.

## 5. Unrelated next-token side-effect probes

**Dataset:** `eval/side_effect_prompts.json`, 12 prompts spanning geography, colors, language and everyday knowledge. Frozen prior to evaluation.  
**Dataset commit:** `325112f`.  
**Dataset SHA-256:** `0098d695adf1532ca0c4575420d3b845c26f4e02e7f6b74f30cbce874ca0791d`.  
**Raw-result file:** `eval/results/side_effects_20261029T105547813287Z.json` (local; Git-ignored).  
**Raw-result SHA-256:** `dd9f08c28b3a06a975aec5a1f48a72a9606e02ae58fb17574110004e01e8b6ec`.  
**Coverage:** 36/36 unique pairs, zero missing target scores; tokenizer IDs checked against the running Gemma tokenizer.

Measured **change in log probability of a preselected exact next token**; this is **not** accuracy, perplexity, arithmetic competence, or a broad benchmark.

| Measure across 12 probes | Ghost +16 | Random 2026 +16 |
| --- | ---: | ---: |
| Mean paired Δ log P(expected exact token) | −0.011952 | −0.019604 |
| Mean absolute paired change | **0.049951** | **0.079324** |
| Positive shifts | 5 | 7 |
| Negative shifts | 7 | 5 |

Category mean shifts:

| Category | Ghost +16 | Random 2026 +16 |
| --- | ---: | ---: |
| Geography | −0.022675 | −0.009384 |
| Colors | −0.071523 | −0.180248 |
| Language | +0.033211 | +0.013518 |
| Everyday | +0.013179 | +0.097696 |

The SAE direction changed unrelated predictions. In this specific probe set, the mean absolute change was smaller than for one sampled random direction. This does **not** establish that steering preserves general model performance or is globally more selective/safe. Some chosen exact answer tokens may be less natural than alternative valid completions (e.g., numerals versus number words).

## 6. Publication evidence contract and integrity

**Generated artifact:** `eval/summary.json`, schema version 1, about 28 KB; exported by `python -m ghost_lab.export_evidence` from the two local raw-result files.  
**Exporter commit:** `5a766c4` on `feat/steering-validation`.  
**Summary SHA-256:** `4a900d168f03c95a867fd9b88f3ee64abc16f14ce58eb00ee2098a0670fcd55d`.  
**Repeatability check:** The same SHA-256 was observed before and after a second export.  
**Quality gates, last reported:** 60 passing tests; Ruff lint and format clean; strict mypy clean across 13 source files.

The summary contains: analysis for primary and secondary metrics; all 20 held-out prompt deltas; 12 side-effect prompt deltas; experiment and dataset fingerprints; model/vector provenance; and statuses marking weight-level exorcism, general-capability preservation, and selective-concept control as unestablished. Reported scan found no `/home/masse`, `/mnt/data`, `localhost`, or `127.0.0.1` strings.

**Important distinction:** The compact summary is *derived evidence*, not a complete public raw-results archive. Full JSON responses and per-condition server logs remain local and Git-ignored. A third party cannot fully rerun or audit every measurement using only the summary; they need the model, extracted vector, runtime, raw records or rerun instructions.

## 7. Claim ledger — text and UI must conform

| Status | Allowed formulation | Prohibited overstatement |
| --- | --- | --- |
| Observed | We extracted a published SAE decoder direction and applied a layer-22 control vector during local Gemma inference. | We found or replaced a literal thought in the weights. |
| Observed | The pilot response stayed `" man"` under all three conditions while exact next-token probabilities differed. | Gemma started producing ghost stories or was possessed. |
| Observed | On 20 frozen prompts, SAE's mean effect exceeded five sampled random-direction means on both measured metrics. | Ghost steering conclusively outperforms random interventions or generalizes to new tasks. |
| Observed | 13/20 primary shifts and 11/20 secondary shifts were positive. | Every prompt moved toward ghosts. |
| Observed | Unrelated exact-token probabilities also changed in 12 probes. | No unrelated functionality was affected, or the model was unharmed. |
| Not tested | A targeted weight edit could be investigated in future. | We erased the ghost feature, changed weights, exorcised the model, or restored it afterward. |
| Not established | No full model accuracy or perplexity benchmark was run. | We measured intelligence, fluency, safety, or general accuracy. |
| Not established | Runtime hook equivalence merits separate verification. | Our SAE and llama.cpp insertion points are proven numerically identical. |

### Approved short conclusion

> We used an SAE-derived direction to intervene in Gemma 3 4B's internal activations. Across 20 previously untested, precommitted prompts, the direction produced a small positive mean shift for an exact ghost-related token and exceeded five sampled norm-matched random-control means. Individual responses varied, and unrelated token predictions changed too. We demonstrated a measurable intervention—not selective control of an entire concept, a permanent weight edit, or preservation of general capability.

## 8. Reproduction checklist

1. Obtain the compatible Gemma 3 4B IT Q4_K_M GGUF and the Gemma Scope 2 SAE checkpoint under their respective licenses/terms; obtain or regenerate the decoder direction for feature 23871.
2. Build a compatible llama.cpp revision with support for layer-specific `--control-vector-scaled` and `--control-vector-layer-range`; verify flags and GGUF tensor `direction.22`.
3. Follow repository setup (`uv sync`, project Python version, GGUF dependency) and inspect the actual configured model paths; current scripts use paths under `~/models/ghost-hunters-lab` and `~/src/llama.cpp`.
4. Verify token IDs for the exact strings, including leading spaces, with `/tokenize`; note `" 4"` is multiple tokens in the recorded tokenizer preflight.
5. Confirm dataset hashes before running; preserve identical `temperature=0`, `n_predict=1`, `n_probs=1000`, seed and request settings across conditions.
6. Run the held-out evaluator and side-effect evaluator; inspect server logs, runtime revision, vector hashes and status. Server health alone does not prove numerical hook equivalence.
7. Re-run `ghost_lab.analyze_heldout` and `ghost_lab.export_evidence` against the designated raw results, then verify output hashes. The exporter currently pins the recorded source-file hashes, so regenerated raw files will not automatically match the original publication snapshot.
8. Run `uv run ruff check .`, `uv run mypy`, `uv run pytest -q`, and verify that all results are complete and unique.

## 9. Remaining questions and publication priority

**Required before publication:** editorial UI with zero invented data; strong README methodology and limitations; DEV write-up; correct provenance and license notices; visible distinction between recorded pilot and frozen results; independent review and accessible mobile presentation.

**Optional future research:** strength sweeps, more random directions, larger independent prompt sets, full capability/perplexity evaluation, direct hook equivalence verification, weight-level exorcism and reinjection. None of these is a completed result in this snapshot.

**Scientific framing:** This project borrows Gilbert Ryle's “ghost in the machine” metaphor from *The Concept of Mind* (1949); it is not a test of whether the model has a mind.
