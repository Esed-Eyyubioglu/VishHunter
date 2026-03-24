# VishHunter

Thesis-aligned implementation of `VISHHUNTER: Voice Phishing (Vishing) Attack Detector`.

## What is implemented

- `FastAPI` backend
- `PostgreSQL-ready` SQLAlchemy schema
- `JWT` authentication
- Encrypted transcript storage aligned to thesis security handling
- `HTML + CSS + JavaScript + Bootstrap` frontend
- `Chart.js` dashboards
- `Wavesurfer.js` waveform view
- `Faster-Whisper` transcription
- `Librosa` audio features
- `DistilBERT` text embeddings
- `XGBoost + SVM + Logistic Regression` hybrid scoring
- Audit logging
- First-run admin setup instead of seeded demo accounts
- Clean delivery state with no pre-created users, cases, or review records

## Public datasets used for model training

The current training workflow uses real public datasets:

- Fraud / scam call positives: `Robocall Audio Dataset`
- Benign bank-call negatives: `Gridspace-Stanford Harper Valley`

These are not committed to the repo and should be downloaded locally into `external-data/`.

This pairing is useful for a public proof-of-build, but it does not guarantee universal performance across every call category. It covers scam/robocall behavior plus legitimate service-style calls, not the full real-world distribution of all phone conversations.

The old mock Next.js prototype has been removed from the runnable deliverable so the workspace now reflects only the production FastAPI application.

## First-time setup

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Download the public datasets into the repo workspace:

```powershell
git clone --depth 1 https://github.com/wspr-ncsu/robocall-audio-dataset.git external-data/robocall-audio-dataset
git clone --depth 1 https://github.com/cricketclub/gridspace-stanford-harper-valley.git external-data/harper-valley
```

If those folders already exist, do not clone again. Either keep using the existing copies or refresh them with:

```powershell
git -C external-data/robocall-audio-dataset pull
git -C external-data/harper-valley pull
```

3. Start the app:

```powershell
python -m uvicorn vishhunter.main:app --reload
```

4. Open `http://127.0.0.1:8000`.

5. Complete the initial setup screen to create the first administrator account.

## How to run and test manually

1. Sign in with the administrator account you created during setup.
2. Open `System Settings`.
3. Confirm the fraud dataset path and benign dataset path point to:
   - `external-data/robocall-audio-dataset`
   - `external-data/harper-valley`
4. Click `Retrain From Public Datasets` once if you want to rebuild the model artifacts.
5. Review the validation metrics shown in `System Settings` after retraining. The app now records cross-validated fusion metrics and recommended thresholds from the public training data.
6. Create one technician user and one analyst user in `User Management`.
7. Sign out and sign in as the technician.
8. Open `Upload & Analyze`.
9. Upload [thesis-smoke.wav](C:/Users/Prog/Desktop/FYP-2/sample-data/thesis-smoke.wav) or your own WAV/MP3 call recording.
10. Wait for the case detail page.
11. Verify:
   - transcript is visible
   - waveform area loads
   - audio score, text score, confidence, and risk level are visible
   - PDF export works
12. Sign out and sign in as the analyst.
13. Open `Review & Validation`, open the case, and save a review decision.
14. Sign back in as administrator.
15. Confirm the result in `Dashboard`, `Case History`, `Reports & Analytics`, and `Audit Logs`.

## Notes

- The first model build can take longer because it may download `Faster-Whisper` and `DistilBERT` weights.
- By default the app uses local SQLite for immediate use.
- To use PostgreSQL, set `DATABASE_URL` before starting the app.

Example:

```powershell
set DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/vishhunter
python -m uvicorn vishhunter.main:app --reload
```
