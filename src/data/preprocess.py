import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer

from data.text_cleaning import DatasetPreprocessor, extract_artist_names, extract_artist_tokens, generate_new_pipeline

load_dotenv()

logger = logging.getLogger("experiment")

ARTIST_TOKENS_PATH = "dataset/artist_tokens.json"


def preprocess_dataset() -> pd.DataFrame:
    dataset = pd.read_json("dataset/videos.json")

    dataset = dataset.dropna(subset=["Is Music"])

    dataset["Title"] = dataset["Title"].fillna("")
    dataset["Description"] = dataset["Description"].fillna("")
    artist_names = extract_artist_names(dataset["Description"], dataset["Channel Name"])
    artist_tokens = sorted(extract_artist_tokens(artist_names))
    Path(ARTIST_TOKENS_PATH).write_text(json.dumps(artist_tokens))
    logger.info(f"Saved {len(artist_tokens)} artist tokens to {ARTIST_TOKENS_PATH}.")

    text_pipeline = generate_new_pipeline(verbose=True)
    transformer = ColumnTransformer(
        [
            ("title", text_pipeline, "Title"),
            ("description", text_pipeline, "Description"),
        ],
        verbose_feature_names_out=False,
        remainder="passthrough",
        verbose=True,
    ).set_output(transform="pandas")
    processed_dataset = transformer.fit_transform(dataset)

    len_before = len(processed_dataset)
    processed_dataset = DatasetPreprocessor.remove_similar_rows(processed_dataset)
    len_after = len(processed_dataset)
    logger.info(f"Removed {len_before - len_after} similar rows.")

    processed_dataset.to_json("dataset/p2_dataset.json", index=False)

    return processed_dataset
