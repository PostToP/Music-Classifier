# AI_Classifier

Backend service for PostTop music classification. Includes dataset preparation, feature pipelines, PyTorch model training/packaging, and a production Flask inference API.

## Stack

- Python 3.11
- Flask + Waitress
- PyTorch
- NumPy + pandas + scikit-learn
- PostgreSQL (`psycopg2`) for dataset fetch
- Docker

## Requirements

- Python 3.11+
- pip
- PostgreSQL (only required for `fetch`)

## Environment Variables

Create a `.env` file in the project root.

- `POSTGRES_DB` - PostgreSQL database name
- `POSTGRES_USER` - PostgreSQL user
- `POSTGRES_PASSWORD` - PostgreSQL password
- `POSTGRES_HOST` - PostgreSQL host
- `POSTGRES_PORT` - PostgreSQL port

## Run API

```bash
pip install -r requirements-prod.txt
python src/prod.py
```

Server starts on http://localhost:5000.

## Pipeline Commands

- `python src/cli.py fetch` - fetch dataset rows from PostgreSQL and save `dataset/videos.json`
- `python src/cli.py split` - split dataset
- `python src/cli.py preprocess` - clean and normalize text/labels
- `python src/cli.py feature` - build feature datasets and pipelines
- `python src/cli.py optuna` - run 100 Optuna trials for the PyTorch model, scored on the videos the prod model predicted (adds to the study in `optuna_study.db`)
- `python src/cli.py train` - train the model with the best Optuna trial and save `model/final_model.pt` plus `model/final_model_config.json`
- `python src/cli.py labelerrors` - rank likely mislabelled videos by out-of-fold predictions (trains 5 models) and save `reports/label_error_candidates.csv`
- `python src/cli.py compile` - build `model/model_wrapper.npz`, the single-file numpy model `src/prod.py` serves
- `python src/cli.py fetch preprocess split feature optuna train compile` - run end-to-end pipeline

## API

Endpoint:

- `POST /predict`

Request body:

```json
{
  "title": "Song title",
  "description": "Video description",
  "categories": ["Music"],
  "duration": 300
}
```

Response:

```json
{
  "prediction": 0.97,
  "version": "v2.0.0"
}
```

## Docker

```bash
docker build -t ai-classifier .
docker run --rm -p 5000:5000 ai-classifier
```

Or pull and run the published image:

```bash
docker pull ghcr.io/posttop/ai-music:latest
docker run --rm -p 5000:5000 ghcr.io/posttop/ai-music:latest
```