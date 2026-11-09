"""Write src/data/nltk_tables.json, the WordNet noun data and stopwords the text cleaner needs without nltk.

Rerun after upgrading nltk or its corpora: `python src/data/build_nltk_tables.py` from the repo root.
"""

import json
import re
from pathlib import Path

from nltk import download as nltk_download
from nltk.corpus import stopwords, wordnet

TABLES_PATH = Path("src/data/nltk_tables.json")
LEMMATIZABLE_FORM = re.compile(r"[a-z]+")


def main() -> None:
    nltk_download("wordnet", quiet=True)
    nltk_download("stopwords", quiet=True)

    noun_lemmas = set(wordnet.all_lemma_names(pos="n"))
    noun_exceptions = {
        form: lemmas for form, lemmas in wordnet._exception_map["n"].items() if LEMMATIZABLE_FORM.fullmatch(form)
    }
    exception_lemmas = {lemma for lemmas in noun_exceptions.values() for lemma in lemmas if lemma in noun_lemmas}
    reachable_noun_lemmas = {lemma for lemma in noun_lemmas if LEMMATIZABLE_FORM.fullmatch(lemma)} | exception_lemmas

    tables = {
        "noun_lemmas": sorted(reachable_noun_lemmas),
        "noun_exceptions": dict(sorted(noun_exceptions.items())),
        "english_stopwords": stopwords.words("english"),
    }
    TABLES_PATH.write_text(json.dumps(tables, separators=(",", ":")))


if __name__ == "__main__":
    main()
