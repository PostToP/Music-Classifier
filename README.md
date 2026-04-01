# AI_Classifier

Backend service for PostTop music classification. Includes dataset preparation, feature pipelines, model training/compilation, and a production Flask inference API.

## Stack

- Python 3.11
- Flask + Waitress
- TensorFlow / Keras
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
- `python src/cli.py train` - train the model and save `model/final_model.keras`
- `python src/cli.py compile` - build `model/model_wrapper.tar.gz` for runtime inference
- `python src/cli.py optuna` - run hyperparameter tuning
- `python src/cli.py fetch split preprocess feature train compile` - run end-to-end pipeline

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
  "prediction": [0.97],
  "version": "v1.0.0"
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