"""Per-text cleaning steps shared by training and prod; no nltk, sklearn or pandas so prod stays small."""

import json
import re
from itertools import cycle
from pathlib import Path
from unicodedata import combining, normalize

import contractions
from ftfy import fix_text
from tldextract import extract
from unidecode import unidecode

from config.config import custom_contractions, custom_stopwords

# Built by src/data/build_nltk_tables.py from nltk's WordNet noun index, noun exceptions and English stopwords.
NLTK_TABLES = json.loads((Path(__file__).parent / "nltk_tables.json").read_text())
NOUN_LEMMAS = frozenset(NLTK_TABLES["noun_lemmas"])
NOUN_EXCEPTIONS = NLTK_TABLES["noun_exceptions"]
# nltk.corpus.reader.wordnet.WordNetCorpusReader.MORPHOLOGICAL_SUBSTITUTIONS["n"], in nltk's order.
NOUN_SUFFIX_SUBSTITUTIONS = (
    ("s", ""),
    ("ses", "s"),
    ("ves", "f"),
    ("xes", "x"),
    ("zes", "z"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("men", "man"),
    ("ies", "y"),
)
# Kana, CJK ideographs and Hangul stay as written: romanizing them with unidecode produced unrelated Pinyin.
CJK_RUN = re.compile(r"([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]+)")
# nltk's MacIntyre CONTRACTIONS2 that can match letters-only text; each splits after its third letter.
TREEBANK_CONTRACTIONS = frozenset({"cannot", "gimme", "gonna", "gotta", "lemme", "wanna"})

for key, value in custom_contractions.items():
    contractions.add(key, value)

stop_words = set(NLTK_TABLES["english_stopwords"])
stop_words.update(custom_stopwords)


def _split_treebank_contraction(word: str) -> list[str]:
    if word.lower() in TREEBANK_CONTRACTIONS:
        return [word[:3], word[3:]]
    return [word]


def _lemmatize_noun(word: str) -> str:
    """Nltk WordNetLemmatizer.lemmatize(word, "n"): the shortest WordNet noun among the word and its morphy forms."""
    if word in NOUN_EXCEPTIONS:
        forms = NOUN_EXCEPTIONS[word]
    else:
        forms = [word[: -len(old)] + new for old, new in NOUN_SUFFIX_SUBSTITUTIONS if word.endswith(old)]
    lemmas = [form for form in dict.fromkeys([word, *forms]) if form in NOUN_LEMMAS]
    return min(lemmas, key=len) if lemmas else word


class TextPreprocessor:
    """Text preprocessing utilities.

    This class contains static methods for cleaning and processing text data.
    """

    @staticmethod
    def clean_categories(categories: list[str]) -> list[str]:
        """Extract category names from Wikipedia URLs.

        Args:
            categories (list[str]): List of Wikipedia URLs

        Returns:
            list[str]: List of category names

        """
        return [x.replace("https://en.wikipedia.org/wiki/", "").replace("_", " ") for x in categories]

    @staticmethod
    def convert_duration(duration: str) -> int:
        """Convert ISO 8601 duration to seconds.

        Args:
            duration (str): ISO 8601 duration string

        Returns:
            int: Duration in seconds

        """
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not match:
            return 0
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def remove_escaped_characters(text: str) -> str:
        """Remove escaped characters from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with escaped characters removed

        """
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r"\r", " ", text)
        text = re.sub(r"\"", " ", text)
        return text

    @staticmethod
    def normalize_text_to_ascii(text: str) -> str:
        """Convert fancy unicode characters from text to ASCII.

        Args:
            text (str): Input text

        Returns:
            str: Text with fancy unicode characters changed to ASCII

        """
        text = fix_text(text)
        normalized_text = normalize("NFKD", text)
        ascii_text = unidecode(
            "".join([c for c in normalized_text if not combining(c)]),
        )
        return ascii_text

    @staticmethod
    def normalize_text_keeping_cjk(text: str) -> str:
        """Apply normalize_text_to_ascii to everything outside CJK runs."""
        parts = CJK_RUN.split(text)
        return "".join(
            part if is_cjk_run else TextPreprocessor.normalize_text_to_ascii(part)
            for part, is_cjk_run in zip(parts, cycle([False, True]))
        )

    @staticmethod
    def space_cjk_runs(text: str) -> str:
        """Surround each CJK run with spaces so it is its own token next to Latin words."""
        return CJK_RUN.sub(r" \1 ", text)

    @staticmethod
    def __get_second_level_domain(url: str) -> str:
        """Extract the second-level domain from a URL.

        Args:
            url (str): Input URL

        Returns:
            str: Second-level domain

        """
        extracted = extract(url)
        return extracted.domain

    @staticmethod
    def replace_links_with_sld(text: str) -> str:
        """Replace links in text with their second-level domain.

        Args:
            text (str): Input text

        Returns:
            str: Text with links replaced by second-level domains

        """
        urls = re.findall(
            r"(https?:[ |\n]?\/\/[ |\n]?[^\s]+)",
            text,
            flags=re.IGNORECASE,
        )
        urls_starting_with_www = re.findall(r"www\.[^\s]+", text, flags=re.IGNORECASE)
        urls = urls + urls_starting_with_www
        urls = [(x, "".join(x.split(" "))) for x in urls]
        for intext, url in urls:
            sld = TextPreprocessor.__get_second_level_domain(url)
            text = text.replace(intext, sld)
        return " " + text + " "

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with email addresses removed

        """
        EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return re.sub(EMAIL_REGEX, " ", text)

    @staticmethod
    def remove_ats(text: str) -> str:
        """Remove mentions from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with mentions removed

        """
        return re.sub(r"@\S+", " ", text)

    @staticmethod
    def lemmatize_text(text: str) -> str:
        """Lemmatize every word as a WordNet noun, matching nltk's word_tokenize + WordNetLemmatizer().lemmatize.

        Equal to nltk for ASCII letters and single spaces; CJK tokens are not WordNet nouns and pass through.
        """
        tokens = [part for word in text.split() for part in _split_treebank_contraction(word)]
        return " ".join(_lemmatize_noun(token) for token in tokens)

    @staticmethod
    def to_lower(text: str) -> str:
        """Convert text to lowercase.

        Args:
            text (str): Input text

        Returns:
            str: Text converted to lowercase

        """
        return text.lower()

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand contractions in text.

        Args:
            text (str): Input text

        Returns:
            str: Text with contractions expanded

        """
        result = contractions.fix(text)
        return str(result) if result is not None else ""

    @staticmethod
    def remove_hyphen(text: str) -> str:
        """Remove hyphens between words.

        Args:
            text (str): Input text

        Returns:
            str: Text with hyphens removed

        """
        return re.sub(r"(?<=\w)-(?=\w)", "", text)

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """Remove non-ASCII characters and numbers from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with special characters removed

        """
        return re.sub(r"[^a-zA-Z\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", " ", text)

    @staticmethod
    def collapse_whitespaces(text: str) -> str:
        """Collapse consecutive whitespaces into a single space.

        Args:
            text (str): Input text

        Returns:
            str: Text with consecutive whitespaces collapsed

        """
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_stopwords(text: str) -> str:
        """Remove stopwords from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with stopwords removed

        """
        word_tokens = text.split(" ")
        filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
        return " ".join(filtered_sentence)

    @staticmethod
    def remove_consecutive_duplicates(text: str) -> str:
        """Remove consecutive duplicate words from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with consecutive duplicate words removed

        """
        words = text.split()
        new_words = []
        for i in range(len(words)):
            if i == 0 or words[i] != words[i - 1]:
                new_words.append(words[i])
        return " ".join(new_words)


TEXT_CLEANING_STEPS = [
    ("remove_escaped_characters", TextPreprocessor.remove_escaped_characters),
    ("normalize_text_keeping_cjk", TextPreprocessor.normalize_text_keeping_cjk),
    ("space_cjk_runs", TextPreprocessor.space_cjk_runs),
    ("replace_links_with_sld", TextPreprocessor.replace_links_with_sld),
    ("remove_emails", TextPreprocessor.remove_emails),
    ("remove_ats", TextPreprocessor.remove_ats),
    ("to_lower", TextPreprocessor.to_lower),
    ("expand_contractions", TextPreprocessor.expand_contractions),
    ("remove_hyphen", TextPreprocessor.remove_hyphen),
    ("remove_special_characters", TextPreprocessor.remove_special_characters),
    ("collapse_whitespaces", TextPreprocessor.collapse_whitespaces),
    ("lemmatize_text", TextPreprocessor.lemmatize_text),
    ("remove_stopwords", TextPreprocessor.remove_stopwords),
    ("remove_consecutive_duplicates", TextPreprocessor.remove_consecutive_duplicates),
    ("to_lower2", TextPreprocessor.to_lower),
]


def clean_text(text: str) -> str:
    for _name, step in TEXT_CLEANING_STEPS:
        text = step(text)
    return text
