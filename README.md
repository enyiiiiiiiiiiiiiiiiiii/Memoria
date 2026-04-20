# AI CareBridge SG

A full-stack MVP for the NAISC 2026 healthcare track: a Singapore-focused dementia support prototype for **patients, caregivers, volunteers, and clinicians**.



Warning: Due to GitHub storage constraints, raw EEG/MRI datasets are excluded from the repository and loaded locally or via external sources.

## What this project includes

This codebase integrates the ideas discussed into a single, coherent system:

* **Role-based access control** for doctor / caregiver / volunteer / patient
* **Caregiver education** with video tutorials and quiz-based screening
* **AI dementia-risk screening** using tabular datasets or a built-in fallback rules engine
* **Volunteer call / chat analysis** for memory, mood, and coherence signals
* **Community and peer activity matching** for elderly users
* **Mahjong-inspired cognitive game** with points and community-style voucher rewards
* **Confidential notes** visible to clinicians and caregivers, but not patients
* **SQLite persistence** for interactions, alerts, quiz attempts, activities, rewards, and predictions

## Intended NAISC narrative

The prototype is designed to tell one clear story:

> a central AI engine continuously updates dementia concern signals from structured screening, caregiver quiz inputs, and volunteer call/chat observations, then turns them into actionable support for the people around the elderly user.

That aligns with the challenge emphasis on:

* dementia support in the Singapore context
* family-based caregiving
* multilingual / accessible design
* dataset mining and predictive analytics

## Architecture

```text
app.py                        Streamlit front-end
carebridge/
  auth.py                     Password hashing and login helpers
  community.py                Activity and peer matching logic
  config.py                   Paths, demo users, environment flags
  db.py                       SQLite schema and helpers
  games.py                    Mahjong memory + Kopitiam Mix Master game logic
  i18n.py                     Lightweight multilingual labels
  resources.py                Singapore care-resource directory
  risk\_engine.py              Model training + inference + fallback heuristic
  seed.py                     Demo data for instant use
  transcript\_analysis.py      Local AI / optional Gemini transcript analysis
scripts/
  init\_db.py                  Initialise database and seed demo users
  train\_model.py              Train model from a CSV dataset
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/init\_db.py
streamlit run app.py
```

## Demo accounts

* `doctor\_demo / Doctor@123`
* `caregiver\_demo / Caregiver@123`
* `volunteer\_demo / Volunteer@123`
* `patient\_demo / Patient@123`

## Training the AI model from your dataset

The app works immediately with a built-in heuristic screener, but you can replace that with a trained model using one of the provided tabular datasets.

### Supported best path

The cleanest route is to use one of the **tabular Kaggle datasets** first.

```bash
python scripts/train\_model.py /path/to/alzheimers\_disease\_data.csv
```

The trainer:

* auto-detects a binary target column such as `Diagnosis`
* drops obvious ID columns
* builds a preprocessing + model pipeline
* compares logistic regression and random forest
* saves the best pipeline to `data/models/dementia\_risk\_pipeline.joblib`



## Dataset placement

This ZIP excludes the datasets and database.
If you want model training, place your CSVs here before running the trainer:

```text
data/raw/tabular/alzheimers\_disease\_data.csv
data/raw/tabular/dementia\_dataset.csv
```

The folders will be created automatically on first run if missing.

## Optional Gemini enhancement

You can enable Gemini-based transcript analysis if you want a more polished call-analysis demo.

1. Copy `.env.example` to `.env`
2. Set:

```env
ENABLE\_GEMINI=1
GEMINI\_API\_KEY=your\_key\_here
```

This is optional. The app already includes a deterministic local transcript analyser.

## What is production-ready vs MVP

### Implemented now

* working role-separated dashboards
* database persistence
* risk scoring workflow
* call/chat analysis workflow
* education + quiz workflow
* activities + registration
* reward points + voucher issuance

### Deliberately MVP-level

* the game is **mahjong-inspired**, not full multiplayer online mahjong
* transcript analysis is **supportive** and not diagnostic
* caregiver resources are example directory entries, not live integrations
* no deployment secrets / email / telephony integration are included

## Best way to use this for NAISC

1. Train the risk model on the Rabie El Kharoua dataset.
2. Keep the volunteer transcript analysis as a second AI signal.
3. Use the caregiver quiz and education module as your human-centred entry point.
4. Demo the role-lock to show ethical handling of confidential information.
5. Frame the game and activities as **engagement + passive monitoring**, not entertainment alone.

## Suggested 3-minute demo flow

1. Log in as caregiver and run AI screening.
2. Show doctor-only confidential notes.
3. Switch to volunteer and upload / paste a call transcript.
4. Show the alert generated from the transcript.
5. Switch to patient and show the game + activity recommendations + reward voucher.

## Notes on the challenge fit

This MVP was built to match the problem spaces discussed in the challenge slides:

* early detection
* caregiver burden
* safer daily living
* meaningful cognitive engagement

## License

For demo and educational use.

