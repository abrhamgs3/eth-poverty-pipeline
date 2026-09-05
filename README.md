# AI-Screened Poverty Annotation → Econometric Bias Correction

[![Tests](https://github.com/abrhamgs3/eth-poverty-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/abrhamgs3/eth-poverty-pipeline/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A staged, tested pipeline demonstrating a real problem in applied
development economics: using an AI-screened poverty proxy as a regressor
instead of the real (expensive) measurement introduces attenuation bias —
and how to correct for it — run on **real Ethiopian household survey
data**, not a synthetic toy.

This is a companion methods piece to the
[Ethiopia Development & Health Economics](../DHS) portfolio project: it
reuses that project's real World Bank ESS/ERSS household panel (2011–2015)
and its child-health (HAZ) outcome variable, adding a new layer — what
happens to a poverty→child-health estimate if poverty status comes from a
cheap AI screen instead of the full consumption module.

## The question

Administering a full household consumption survey — the standard basis
for a poverty line — is expensive. A tempting shortcut: have a
lightweight screen (a brief field note, scored by an LLM) classify
households as poor/non-poor instead. But that screen has a real,
measurable error rate, and using its output directly as a regressor
introduces **classical measurement error** — attenuating the estimated
effect of poverty toward zero, exactly the mechanism demonstrated in the
sibling [`llm-sentiment-pipeline`](https://github.com/abrhamgs3/llm-sentiment-pipeline)
and [`llm-econ-pipeline`](https://github.com/abrhamgs3/llm-econ-pipeline)
repos, applied here to a real development-economics use case.

This pipeline:

1. Loads the real ESS/ERSS household-wave panel (consumption + child HAZ).
2. Defines a **true** poverty indicator from real per-adult-equivalent
   consumption (bottom third of each wave — see Design decisions below).
3. Generates a short field-note text per household from covariates already
   in the panel (household size, distance to market, elevation, rainfall
   anomaly, weather shocks) — deliberately excluding consumption itself,
   so the classification task isn't circular.
4. Has an LLM (offline mock by default, real API optional) classify each
   note as poor/non-poor.
5. Evaluates that screen against the true indicator on a validation split.
6. Regresses child HAZ on true poverty (benchmark), AI-screened poverty
   (naive, attenuated), and a Rogan–Gladen-corrected estimate.

Real-data result (`sentences_allagree`-equivalent panel, seed=42, n≈4,850
household-waves): poverty is associated with **−0.36 mean HAZ**
(benchmark); the naive AI-screened estimate attenuates that to **−0.20**
(43% attenuation); the correction recovers **−0.37**, closing 94% of the
gap. Numbers will differ slightly by seed and by whether the real panel or
the synthetic fallback is in use — see `output/report.md` after a run.

## Pipeline stages

```
data_extraction → annotation → encoding → evaluation
                                              │
                                              ▼
                              econometrics (SimpleOLS, bias correction)
                                              │
                                              ▼
                                    performance → reports
```

| Stage | Module | What it does |
|---|---|---|
| 0. Data extraction | `data_extraction` | Loads the real ESS panel from `../DHS/data/processed/`, or a schema-matching synthetic panel if unavailable; derives the within-wave relative poverty indicator. |
| 1. Annotation | `annotation` | Builds a covariate-derived field note per household-wave; a pluggable `LLMClient` classifies it, validated against a Pydantic schema. |
| 2. Encoding | `encoding` | Turns the predicted poverty label into the binary `D_pred` regressor. |
| 3. Evaluation | `evaluation` | Accuracy, Cohen's Kappa, F1, and confusion-matrix TPR/TNR on a held-out validation split. |
| 4. Econometrics | `econometrics` | Dependency-free `SimpleOLS`; Rogan-Gladen bias correction with bootstrapped inference. |
| 5. Performance | `performance` | Stage timings; how much of the naive model's gap to the benchmark the correction actually closed. |
| 6. Reports | `reports` | A comparison chart (PNG) + a single Markdown run report. |

## Directory structure

```
.
├── eth_poverty_pipeline/     # the installable package
│   ├── config.py             #   PipelineConfig — real-data path, poverty quantile, seed, etc.
│   ├── data_extraction/      #   loader.py — real-panel loader + synthetic fallback
│   ├── annotation/           #   schema.py, clients.py (Mock + OpenAI), annotator.py (note builder)
│   ├── encoding/
│   ├── evaluation/
│   ├── econometrics/         #   ols.py, bias_correction.py
│   ├── performance/
│   ├── reports/               #   visualizations.py, report_builder.py
│   └── pipeline.py            #   orchestrator + CLI entry point
├── tests/                     # pytest suite (20 tests) — never touches the real ESS data
├── legacy/                    # original prototype scripts — wrong domain entirely, kept for the record
├── output/                    # generated at runtime: chart + report.md (gitignored)
├── main.py
├── requirements.txt / requirements-dev.txt
└── pyproject.toml             # `pip install -e .` → `eth-poverty-pipeline` console script
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -e .
```

## Getting the real data (optional — a synthetic fallback runs without it)

This pipeline reads `../DHS/data/processed/household_wave_panel_i.parquet`
by default — the processed panel built by the sibling `DHS` repo from the
World Bank Ethiopia Socioeconomic Survey (LSMS-ISA). That data is
public-use but requires a free account and is never committed to either
repo — see [`../DHS/docs/wb_lsms_data_checklist.md`](../DHS/docs/wb_lsms_data_checklist.md)
for how to get it, then run `../DHS`'s own build step to produce the
processed panel. Without it, this pipeline automatically falls back to a
synthetic panel with a matching schema and roughly matching distributions
(see `data_extraction/loader.py`) — every command below works either way.

## Quick start

```bash
python main.py                              # real panel if present, else synthetic
python main.py -v                           # verbose stage logging
python main.py --data-path /path/to/panel.parquet   # point at a different panel
python main.py --poverty-quantile 0.4       # bottom 40% instead of the default third
python main.py --client openai --api-key sk-...     # real LLM instead of the offline mock
python main.py --seed 123                   # reproducible re-run
```

Every run writes to `output/`: `poverty_haz_comparison.png` (benchmark vs.
naive vs. corrected, with 95% CIs) and `report.md` (full metrics + table).

## Design decisions (read before citing the numbers)

- **Poverty is defined relatively, within each survey wave**: the bottom
  third of `nom_totcons_aeq` (nominal consumption per adult equivalent) in
  that wave. This is nominal birr, not inflation-adjusted across
  2011/2013/2015, so a fixed cross-wave birr threshold would silently
  conflate "poor in 2011 terms" with "poor in 2015 terms" — a modeling
  choice, not an official Ethiopian poverty line.
- **The field notes are constructed, not real enumerator text** — the real
  ESS survey has no free-text field. They're built from real covariates
  already in the panel (household size, distance to market, elevation,
  rainfall anomaly, weather shocks), explicitly excluding consumption
  itself so the classification task isn't circular. This is a plausible
  stand-in for a lightweight AI-screening deployment, not a claim that
  these specific notes were collected.
- **The "Benchmark" is a real-data estimate, not an oracle.** Unlike the
  two sibling repos, `D_true` here is real, measured data with its own
  sampling error — the "true" effect is genuinely unknown. Success is
  "the corrected estimate lands close to the benchmark, not zero," not
  "recovers a known planted constant."
- **`MockLLMClient` is a controlled-accuracy label permutation**, same
  design as the sibling repos: it's handed the true label and flips it at
  a target rate. It never reads the generated note. `OpenAIClient` (the
  real path) only ever sees the note.

## Validation

- `SimpleOLS`: exact recovery on noiseless data, an explicit test forcing
  the near-singular-design clip path, approximate recovery under noise.
- Bias correction: Rogan-Gladen prevalence recovery and covariance-scaling
  β recovery against known simulated parameters, including a **negative**
  effect size (the real HAZ use case) — the two sibling repos only ever
  tested positive effects, which is exactly what let a chart bug (labels
  overflowing past the axis) through undetected there; see below.
- Data loader: relative-poverty assignment tested to be computed
  separately per wave, not pooled across waves.
- End-to-end: a pipeline smoke test using the synthetic fallback (CI has
  no access to the real panel) asserts every artifact exists, checks the
  report's Markdown tables are well-formed, and checks seed-reproducibility.

Run the suite: `pytest -q` (never touches the real ESS data or the network).

## A chart bug worth naming

The comparison chart's value-label placement (labels sit below each
point's CI, learned from a table-corruption bug in the sibling repos)
assumed a positive effect size — appropriate there, wrong here, where
poverty's effect on HAZ is negative. Running against the real data
surfaced it immediately: labels overflowed past the axis into the x-tick
labels. Fixed by computing an explicit `ylim` that reserves room for the
label text regardless of sign, and moving the observations textbox to
below the axes (figure coordinates) instead of a data-relative corner —
both are sign- and scale-agnostic now. Concretely, this is why running
against *real* data (which doesn't reliably hand you convenient positive
numbers) matters even for a "demo" pipeline.

## Limitations

- **Small validation split at low sample sizes.** `evaluation.metrics`
  warns below 30 examples; the default 20% split against the real panel's
  ~4,850 rows is comfortably large, but a custom `--data-path` with a
  small n could produce an unreliable TPR/TNR estimate.
- **TPR/TNR are treated as known constants** through the bootstrap, which
  resamples the study set but not the validation set that produced them —
  total uncertainty is understated, more so the smaller the validation split.
- **Non-differential misclassification is assumed** — that the AI
  screen's error rate doesn't depend on the true HAZ outcome. Plausible
  here (the note never sees HAZ), but not something to assume blindly for
  a different construction.
- **The correction doesn't address rainfall/geography confounding the
  poverty→HAZ relationship itself** — that's a different, real identification
  problem the sibling `DHS` project's Project 1 (rainfall shocks) already
  investigated (null result, triangulated 5 ways) and Project 2 (DHS
  malaria) is designed around. This pipeline is about correcting
  *measurement* error in the poverty variable, not about causally
  identifying poverty's effect on child health from observational data.

## References

- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014) — LSMS-ISA/ESS survey design lineage referenced in the sibling repos.
- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). *ChatGPT outperforms crowd workers for text-annotation tasks.* PNAS, 120(30).
- Rogan, W. J., & Gladen, B. (1978). *Estimating prevalence from the results of a screening test.* American Journal of Epidemiology, 107(1).
- World Bank LSMS-ISA, Ethiopia Socioeconomic Survey (ESS/ERSS) — see `../DHS/docs/wb_lsms_data_checklist.md`.

## License

MIT — see [LICENSE](LICENSE).
