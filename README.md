# Dialect-Aware Synthesis and Bias-Corrected Evaluation for Hindi Text-to-Speech

**M.Tech. Project (MTP-2), Data Engineering, IIT Jodhpur**
**Author:** Narendra Singh (M25DE2052)
**Supervisor:** Dr. Sidharth Ranjan, Department of Computer Science & Engineering, IIT Jodhpur

---

## What this project is about

Hindi TTS voices are trained on standard (Khadi Boli based) Hindi, but a large share of Hindi speakers grow up with Haryanvi, Chhattisgarhi, Braj, Bundeli, Awadhi or a Bihari variety. This project looks at two connected problems:

1. **Dialect-aware synthesis.** Can a standard Hindi neural TTS model be made to sound regional, without retraining, by rewriting the input text with dialect rules and a word dictionary?
2. **Bias-corrected evaluation.** Listeners from different dialect regions use the MOS scale differently. How do we correct for that before comparing TTS systems?

The starting point is the paper *"Evaluating user preferences in Hindi Text-to-Speech"* (B. Gupta, MeitY). It lists character- and word-level dialect patterns and proposes the normalisation `y = a·x + b` (dialect-group score `x`, Khadi Boli score `y`). I implemented both ideas, tested them, and found and fixed a statistical problem in the normalisation.

---

## What's in this submission

| Item | What it is |
|---|---|
| `Hindi_Dialect_TTS_Colab_MTP2_Final.ipynb` | Main Google Colab notebook. Runs the whole pipeline on a T4 GPU: dialect front-end, Indic-TTS synthesis, stimulus generation, feature extraction, ratings analysis and bias correction. |
| `MTP_Hindi_Dialect_Dataset.zip` | Project dataset: sentences, lexicons, model configs, stimuli, blinded rating sheets, evaluator panel, phonetic features, reference-paper numbers, planned corpus layout. |
| `hindi-dialect-tts.zip` | The same toolkit packaged as an application (FastAPI + SQLite/PostgreSQL), with a listening-test web page, command-line scripts, a paper-replication script and a pytest suite. |

---

## 1. Google Colab notebook

### How to run

1. Open the notebook in Colab, then choose **Runtime → Change runtime type → T4 GPU**.
2. Run **Section 1** (settings) and **Section 2** (writes the toolkit to `/content/hindi_dialect`).
3. Run **Section 2b** and upload `MTP_Hindi_Dialect_Dataset.zip` when asked. You can also put it in Google Drive and set `DATASET_ZIP`.
4. Run **4a–4b** to install `coqui-tts` and load AI4Bharat Indic-TTS. If Colab asks you to restart, restart and run again.
5. Run Sections **5 → 10** in order.

To also build models M2 and M5, set `USE_PARLER = True` in Section 1. This needs a GPU and about a 4 GB download, and the Hugging Face licence for Indic Parler-TTS must be accepted first.

### Notebook sections

| Section | What it does |
|---|---|
| 1 | Settings: engine, speaker, dialect, mode, rule level |
| 2 | Writes the toolkit modules (`app.dialects`, `app.features`, `app.synthesis`, `app.evaluation`, `app.data`) |
| 2b | Loads the dataset; dataset lexicons replace the built-in ones |
| 3 | Text-only dialect transformation and SSML preview |
| 4a–4d | Engines: Indic-TTS (FastPitch + HiFi-GAN), Indic Parler-TTS, gTTS |
| 5a / 5b | Standard vs dialect side-by-side; batch of 20 sentences × 3 conditions (100 WAVs + manifest) |
| 6 | Gradio web UI with a shareable link |
| 7a / 7b | MTP stimuli (20 sentences × 5 models) and the blinded listening kit (ZIP) |
| 8 | Upload natural reference recordings `REF_01…REF_20.wav` (for DMOS and MCD) |
| 9a / 9b | Phonetic/lexical features; prosodic features and dialect prosody profiles from natural speech |
| 10a–10c | Load ratings, MOS/DMOS with CIs, `y = a·x + b` correction, impact metrics, paired tests, graphs |

### Main settings

```python
ENGINE  = "indic_tts"      # indic_tts | indic_parler | gtts
SPEAKER = "male"           # male | female
DIALECT = "haryanvi"       # haryanvi | chhattisgarhi | bundeli | brajbhasha | awadhi | bihari | khadiboli
MODE    = "combined"       # standard | char | lexicon | combined
LEVEL   = "conservative"   # conservative | full
```

- `char`: rules only (the paper's "character variations").
- `lexicon`: dictionary only (the paper's "word variations").
- `combined`: dictionary first, then rules for unknown words. This mode is my addition.
- `conservative`: only regular sound changes. `full`: every pattern in the paper, which can over-apply.

---

## 2. Dataset (`MTP_Hindi_Dialect_Dataset.zip`)

| Item | Value |
|---|---|
| Sentences | 20 (5 each for Haryanvi, Chhattisgarhi, Bundeli, Braj Bhasha); 10 male-voice, 10 female-voice |
| TTS models | M1–M5 (see below) |
| Stimuli | 100 = 20 sentences × 5 models, each with a blinded code |
| Evaluators | 10 slots = 2 per listener group × 5 groups (Khadi Boli reference + 4 dialects) |
| Ratings planned | 2,000 (MOS + DMOS) + 4 repeated items per evaluator and test |
| Lexicon | 125 entries across 7 dialects (+ W3C PLS exports) |
| Planned corpus | 400 utterances = 5 dialects × 2 speakers × 40 prompts |

| Model | Engine | Text front-end | Prosody |
|---|---|---|---|
| M1 | Indic-TTS | Standard Hindi | – |
| M2 | Indic Parler-TTS | Standard Hindi | – |
| M3 | Indic-TTS | Dialect (combined) | – |
| M4 | Indic-TTS | Dialect (combined) | SSML prosody profile |
| M5 | Indic Parler-TTS (interim for a future VITS model) | Dialect (combined) | SSML prosody profile |

```
sentences/        eval_sentences.csv – text, target dialect, voice, style, dialect rendering
lexicon/          <dialect>.csv (hindi, dialect_form, pos, source, verified, notes) + pls/*.pls
models/           tts_models.csv
stimuli/          stimuli.csv, rating_sheets/E01_MOS.csv …, playlists_KEY_researcher_only.csv
evaluators/       evaluator_panel.csv
features/         phonetic_change_inventory.csv, phonetic_category_profile.csv,
                  lexical_substitutions_eval_sentences.csv, prosody_profiles.csv
reference_study/  table5_evaluator_gradings.csv, regression_digitized_points.csv
corpus/           recording_prompts.csv, manifest_PLANNED.csv, natural_references_PLANNED.csv
ratings/          ratings_TEMPLATE.csv, ratings_SIMULATED_demo_DO_NOT_REPORT.csv
```

**Notes:**

- Each lexicon row has a `source` tag:
  - `paper_table4`: taken from the reference paper (101 rows).
  - `derived_paper_rule`: derived from a documented rule (1 row).
  - `community_seed`: common dialect forms that still need checking by native speakers (23 rows, including all Bundeli entries).
- `ratings_SIMULATED_demo_DO_NOT_REPORT.csv` contains **simulated** ratings with a planted dialect bias. It is only for testing the analysis code. Never report numbers from it as results.
- `playlists_KEY_researcher_only.csv` decodes the blinded rating sheets. Do not share it with evaluators.

---

## 3. Application code (`hindi-dialect-tts.zip`)

The same toolkit as a standalone app. It has an API for transforming and synthesising text, a browser listening test with evaluator metadata, a database for ratings, and corrected results.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# without TTS models (mock audio) – everything runs end to end
TTS_ENGINE=mock python scripts/manage.py init
TTS_ENGINE=mock python scripts/manage.py generate
TTS_ENGINE=mock uvicorn app.main:app --reload      # http://127.0.0.1:8000  (API docs at /docs)

# with AI4Bharat Indic-TTS (Python 3.10/3.11)
pip install -r requirements-tts.txt
python scripts/download_indic_tts.py
python scripts/manage.py init --speakers female,male
python scripts/manage.py generate
```

Other useful commands:

```bash
pytest                                # test suite (mock engine, temporary SQLite)
python scripts/replicate_paper.py     # writes reports/replication_report.md + figures
python scripts/power_analysis.py --effect 0.2 --raters 5 10 20 30
docker compose up --build             # API + PostgreSQL
```

The core test suite runs in about 2 seconds: 156 passed. The API tests are skipped if `fastapi` is not installed.

---

## Results so far

**Dialect front-end**

- The rules reproduce **55 of the 57** example word pairs in the reference paper. Both failures come from two correct rules clashing on the same word:
  - गाड़ी → गारी instead of गाड्डी.
  - लक्ष्मण → लक्स्मन instead of लक्ष्मन.
- Rules alone regenerate 92 % of the Awadhi dictionary but only 27 % of the Haryanvi one, so the dictionary is still needed. This is why `combined` mode exists.
- In `combined` mode, about 30 % of words change for Haryanvi with conservative rules and 38 % with the full set.

**Synthesis (Colab, T4 GPU)**

- Indic-TTS Hindi model loaded at 22,050 Hz.
- 100-file batch for the paper's three conditions (standard / character / dictionary).
- 60 MTP stimuli (M1, M3, M4). M2 and M5 were not built because Parler-TTS was switched off.
- Blinded listening kit and Gradio demo produced.

**Bias correction**

- The paper's least-squares fit of `y = a·x + b` is pulled towards a slope of zero because the dialect means are noisy (regression dilution). In my simulation it gave slopes of **0.62–0.71** when the true slopes were 0.8–1.25.
- I replaced it with **mean–sigma equating**: `a = sd(y)/sd(x)`, `b = mean(y) − a·mean(x)`. A bootstrap check falls back to an offset-only correction when the slope is not clearly different from 1. Equating gave 0.90–1.17 in the same simulation.
- Re-testing the paper's own Table 5 scores: all four dialect-adapted conditions were preferred significantly over standard Hindi (Holm-adjusted p ≤ 0.022).
- On the *simulated* rating file only:
  - The new correction closed the gap between listener groups (0.84 → 0.002 MOS points) and kept the spread between systems.
  - The paper's least-squares mapping removed about 73 % of that spread.

---

## Known limitations

- **No real listener ratings yet.** Every MOS/DMOS number produced in Section 10 comes from the simulated demo file.
- No natural reference recordings yet, so DMOS, MCD and measured prosody profiles are pending. With neutral profiles, M4 sounds the same as M3.
- The lexicon is small (125 entries), and the `community_seed` rows are unverified.
- The rules work word by word: no morphology or syntax, and some over-application (e.g. वन → वण in Haryanvi).
- The G2P only deletes the word-final schwa.

## Troubleshooting

| Problem | Fix |
|---|---|
| `WeightsUnpickler` / `weights_only` error | Re-run cell 4b; it patches `torch.load` for PyTorch ≥ 2.6 |
| `KeyError: 'female'` / speaker error | Re-run 4a; it fixes the speaker-file path in the released configs |
| "Restart session" after installing | Restart, then *Run all* again (downloads are kept) |
| Section 7a shows `engines available: []` | Run 4a/4b; for M2/M5 set `USE_PARLER = True` and run 4c |
| `AssertionError: Load the dataset first` | Run 2b and upload the dataset zip |
| Results tagged `[SIMULATED DEMO]` | Set `USE_DEMO_RATINGS = False` and upload real rating sheets |

## Next steps

1. Run the listening test with real evaluators from all five groups.
2. Record the 20 natural references and the 400-utterance corpus.
3. Turn on Parler-TTS for M2/M5.
4. Grow and verify the lexicons.
5. Fine-tune a dialect-aware VITS model for M5.

## Acknowledgements

- My supervisor, Dr. Sidharth Ranjan.
- B. Gupta (MeitY), for the reference study.
- AI4Bharat, for the Indic-TTS and Indic Parler-TTS models.
- The Coqui TTS, PyTorch, NumPy, SciPy, pandas, Matplotlib and Gradio projects.
