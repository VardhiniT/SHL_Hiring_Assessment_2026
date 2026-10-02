# Grammar Scoring for Spoken Audio (SHL Hiring Assessment 2026)

Predict a 0-5 grammar score for 45-60 s spoken clips (769 training clips, 216 test clips).

The pipeline turns each audio clip into a transcript with word-level confidence, extracts language-quality features from it, and predicts the score with a small, regularised ensemble.

## Approach

1. **Speech-to-text.** Whisper `medium.en` (via `faster-whisper`) transcribes every clip with word timestamps and word-level confidence. Voice-activity filtering is on, and a disfluency-friendly prompt keeps fillers ("um", "uh") and repeats, because grammar scoring depends on what the speaker actually said.
2. **Features** (from the transcript only, so no label leakage):
   - **Fluency / structure (27):** speech rate, articulation rate, pauses, sentence count and length, type-token ratio, fillers, repeated words and bigrams, contractions, and Whisper confidence (mean word probability, share of low-confidence words).
   - **GPT-2 surprise (4):** average negative log-likelihood per token for the whole transcript and for its sentences.
   - **CoLA acceptability (4):** probability that each sentence is grammatically acceptable (`textattack/roberta-base-CoLA`), summarised as mean, minimum, standard deviation and share acceptable.
   - **Sentence embedding (768-d):** `all-mpnet-base-v2`.
3. **Models.** Ridge (alpha chosen by internal CV) and HistGradientBoosting on the 35 linguistic features, and Ridge on the embedding. The final score is the plain average of the three, clipped to [0, 5].
4. **Validation.** 3 x repeated stratified 5-fold CV on the training set. Imputation and scaling are fitted inside each fold.

## Results

| Model | RMSE | Pearson | MAE | Within 0.5 |
|---|---|---|---|---|
| Baseline (predict the mean) | 1.2383 | -0.016 | 0.996 | 0.310 |
| Ridge (linguistic) | 0.7522 | 0.794 | 0.571 | 0.530 |
| HistGradientBoosting (linguistic) | 0.7670 | 0.785 | 0.580 | 0.532 |
| Ridge (embedding) | 0.9960 | 0.594 | 0.751 | 0.432 |
| **Ensemble (average), final model** | **0.7465** | **0.809** | 0.581 | 0.521 |
| Zero-aware two-stage model (not used) | 0.7488 | 0.797 | 0.557 | 0.551 |

- **Cross-validated (out-of-fold) RMSE: 0.7465.** This is the honest estimate of performance on unseen clips, about 40% below the baseline.
- **Training RMSE: 0.5873** (Pearson 0.897). This is the model fitted on all 769 clips and scored on the same clips, so it is optimistic.

The 37 clips labelled 0 are the largest source of error (RMSE 1.74 on those clips against 0.65 on the rest). A two-stage "zero-aware" model was tested: a classifier detects zero clips well (out-of-fold ROC-AUC 0.956) and lowers the error on them, but it raises the error elsewhere, so overall RMSE did not improve by the required margin (0.01) and the plain ensemble was kept.

## Files

| File | Description |
|---|---|
| `SHL_Grammar_Scoring.ipynb` | Full pipeline with outputs, plots and a discussion after each step |
| `submission.csv` | Predictions for the 216 test clips (`filename,label`) |
| `transcripts_medium.en.json` | Cached Whisper output, so the ~36 minute transcription can be skipped |
| `requirements.txt` | Python dependencies |

## How to run

The notebook was run on Kaggle (GPU T4, Internet on) with the competition data attached.

1. Open the notebook on Kaggle and attach the competition dataset.
2. Turn on a GPU and Internet. The notebook downloads Whisper, GPT-2, CoLA and the embedding model.
3. Run all cells. Transcription takes about 36 minutes on a GPU and is cached to `/kaggle/working/transcripts_medium.en.json`, so an interrupted run resumes where it stopped.
4. To skip transcription, copy `transcripts_medium.en.json` from this repo into `/kaggle/working/` before running the transcription cell. The remaining cells then take a few minutes.

Locally: `pip install -r requirements.txt`. A GPU is strongly recommended. Without one the notebook falls back to Whisper `small.en`, which is slower and less accurate.

Audio files are named identically in `train/` and `test/` but are different recordings, so every clip is keyed by `split/filename`.

## Limitations

- **Whisper confidence is the strongest feature, but it mixes audio clarity with grammar.** Unclear recordings get low confidence and also tend to get low scores, so part of the signal is recording quality.
- **ASR smoothing.** Whisper can partly correct ungrammatical speech and drop fillers, which hides some grammar errors.
- **Zero-labelled clips** are predicted poorly. Their transcripts are often fluent-sounding but incoherent or repetitive, and some may contain little real speech. This was not confirmed by listening.
- **Shrinkage toward the middle.** Extreme scores (near 0 and 5) are under-predicted.
- **Train/test shift.** Test clips are shorter on average (48.8 s vs 55.8 s), which may slightly affect duration-based features.
- Individual Ridge coefficients should not be interpreted causally, because the features are correlated.

## Possible improvements

- Listen to a sample of the zero-labelled clips and handle them explicitly (for example a softer blend using the zero detector).
- Use a more verbatim ASR, or add rule-based grammar-error counts.
- Fine-tune a small transformer on the transcripts, or add audio embeddings.
