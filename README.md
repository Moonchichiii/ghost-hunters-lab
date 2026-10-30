# The Ghost Hunter's Lab 👻

**A local, open-weight interpretability experiment built for the 2026 Hacktoberfest DEV Challenge, Week 4: Ghost in the Machine.**

I went looking for a ghost inside an AI model. Instead of asking a model to write a ghost story, I extracted a published ghost-related sparse-autoencoder (SAE) feature direction, injected it into **Gemma 3 4B IT** during local inference, and measured what changed against random controls.

**The finding is deliberately modest:** the intervention caused a small, variable shift in next-token scores. It also changed unrelated predictions. I did **not** edit the model weights, erase a concept, or establish that the model's general capabilities were preserved.

## See the experiment

- **Source and evidence:** this repository
- **Read the field notes:** [Scientific evidence report](docs/research/EVIDENCE_REPORT.md)
- **Inspect the compact recorded results:** [eval/summary.json](eval/summary.json)
- **Recorded demonstration:** [Add public demo/video URL before submitting to DEV]

The FastAPI site is an **editorial explorer of recorded results**, not a live model-inference endpoint. Its 34-layer illustration is schematic, not an activation scan.

## What I built

- Extracted a 2,560-dimensional decoder direction for **feature #23871**, labeled with ghost/spirit-related text in Gemma Scope 2, from the SAE for layer 22.
- Packaged the vector as a llama.cpp GGUF control vector (`direction.22`), leaving original Gemma weights unchanged.
- Generated **five reproducible norm-matched random controls** (seeds 2026–2030).
- Froze **20 prompts** before testing and collected **140 condition–prompt observations**: unmodified Gemma, ghost +16 and five random +16 directions.
- Froze **12 unrelated next-token probes** (geography, colors, language, everyday knowledge) and collected **36 further observations**: baseline, ghost +16 and random +16.
- Built a small **FastAPI + Jinja2 + native CSS** editorial site with self-hosted Poppins WOFF2 and GSAP/ScrollTrigger, showing the recorded evidence.

## Measured results

| Metric | Ghost SAE +16 | Interpretation |
| --- | ---: | --- |
| Mean change in log P(exact token `" ghost"`) across 20 prompts | **+0.016386944** | 13/20 positive; higher mean than each of five sampled random directions |
| Mean change in log preference `" ghost"` vs `" man"` | **+0.042218480** | 11/20 positive; higher mean than each sampled random direction |
| Mean absolute change in log P(expected unrelated token), 12 probes | **0.049950896** | Predictions outside the ghost context also changed |
| Same unrelated-token measure under random 2026 +16 | **0.079324351** | One control only; not evidence of generally safer steering |

The pilot prompt `The mysterious figure was a` generated the next token `" man"` under baseline, ghost +16 and the original random +16 control, even though the probability of the exact alternative token `" ghost"` changed (approximately 5.86%, 6.13% and 6.30% respectively). **Random was highest on that absolute probability metric.** This pilot was exploratory and separate from the 20-prompt evaluation.

These are **exact single-token probability probes**, not a benchmark of creativity, reasoning, accuracy, or the complete semantic concept of ghosts. The prompt set was precommitted before its own evaluation but constructed with knowledge of the feature and earlier pilot.

## Stack

- Python 3.14, `uv`, NumPy, safetensors, GGUF, pytest, Ruff and strict mypy
- Gemma 3 4B IT Q4_K_M, Gemma Scope 2 and a compatible locally built llama.cpp
- FastAPI, Jinja2, native CSS, locally bundled Poppins WOFF2, GSAP and ScrollTrigger for the site
- No database, cloud inference API or user accounts

## Run the editorial site

From the repository root:

```bash
uv sync --group dev
uv run uvicorn ghost_lab.web.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**. The website reads the committed `eval/summary.json`; **it does not need model weights or a running llama-server**.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

## Reproduce the research

The **full-precision raw records and server logs are intentionally ignored by Git**; this is a compact publication snapshot, not a complete downloadable raw archive. To rerun the experiments, first supply compatible weights and the SAE decoder direction under their respective license terms.

1. Obtain a compatible **Gemma 3 4B IT Q4_K_M GGUF** and the Gemma Scope 2 checkpoint `resid_post/layer_22_width_65k_l0_medium`.
2. Extract feature **23871** into `~/models/ghost-hunters-lab/ghost-23871.f32`, using `ghost_lab.extract_feature`, and create `ghost-23871.gguf` as in the project scripts. Inspect the scripts' expected filenames before running; the current control generator produces five seeded control files and refuses to overwrite existing ones.
3. Build a compatible `llama.cpp` with control-vector support under `~/src/llama.cpp`. The recorded experiments used revision `fae9ac561`. Verify the flags `--control-vector-scaled` and `--control-vector-layer-range` against your build.
4. Confirm the frozen datasets `eval/heldout_prompts.json` and `eval/side_effect_prompts.json`, plus tokenizer IDs and model/vector hashes.
5. Run `uv run python -m ghost_lab.evaluate_heldout` and `uv run python -m ghost_lab.evaluate_side_effects` locally. Both scripts start sequential `llama-server` configurations on loopback port 8081 and save ignored raw results under `eval/results/`.
6. Review `uv run python -m ghost_lab.analyze_heldout`. The publication exporter `uv run python -m ghost_lab.export_evidence` pins the **original raw-file SHA-256 values**, so a fresh rerun will not automatically regenerate the identical publication file. Do not overwrite the committed snapshot without deliberate provenance review.

For exact hashes, runtime settings, limitations, and a detailed claim ledger, read [the scientific evidence report](docs/research/EVIDENCE_REPORT.md).

## Why open weights mattered

A standard closed text-generation API would let me prompt for a ghost story. It would not ordinarily expose this model's internal residual stream, let me insert an SAE decoder direction at a chosen layer, or edit its weights. Access to the locally runnable model, the published SAE checkpoint and llama.cpp's control-vector interface made this particular experiment possible. The local model inference used no hosted API, though hardware and electricity still have costs.

## Limitations and credits

- This is **activation steering**, not weight-level exorcism; the model weights remained unchanged.
- No proof of a single stored thought, consciousness, general selectivity, or preservation of broad capabilities.
- Five sampled random vectors and 20 deliberately chosen prompts are not a comprehensive statistical evaluation.
- Equivalence between the original SAE hook and the llama.cpp injection point was not independently established.
- The figure of 34 concentric rings represents **depth schematically**, not the geometry of features or measured activations.
- Built using **Google Gemma 3 4B IT**, **Gemma Scope 2** research resources, **llama.cpp**, and the open-source Python tooling above. Respect each upstream license. Third-party CSS font and GSAP distribution files retain their own licenses.

**Built during 26–30 October 2026.** The initial repository commit is dated 27 October. Future changes made after the contest deadline will be noted here, as required by the challenge rules.

> Gilbert Ryle's *The Concept of Mind* (1949) supplies the “ghost in the machine” metaphor. This project does not investigate whether AI has a mind.
