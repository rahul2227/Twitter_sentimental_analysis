import re
from typing import List


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
_APOSTROPHE_RE = re.compile(r"’")  # normalize curly apostrophe


def _normalize_basic(text: str) -> str:
    """Apply lightweight normalization:

    - Lowercase
    - Normalize curly apostrophes to straight
    - Replace URLs with <URL>
    - Replace @mentions with <USER>
    - Keep hashtag word by stripping leading '#'
    - Collapse 3+ repeated characters to 2 (e.g., cooool -> coool)
    """
    text = text.strip().lower()
    text = _APOSTROPHE_RE.sub("'", text)
    text = _URL_RE.sub(" <url> ", text)
    text = _MENTION_RE.sub(" <user> ", text)

    # Replace hashtags by the word itself (keep signal without '#')
    def _strip_hashtag(m: re.Match) -> str:
        return " " + m.group()[1:] + " "

    text = _HASHTAG_RE.sub(_strip_hashtag, text)

    # Collapse char repeats: 'sooooo' -> 'sooo' (max three repeats preserved)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return text


def preprocess_and_tokenize(text: str) -> List[str]:
    """Normalize and tokenize tweet text with a fast regex-based approach.

    This replaces TextBlob tokenization used in the legacy script to improve
    speed and ensure consistent tokenization without external corpora.

    Rules:
    - Accept words with optional internal apostrophes (don't, i'm)
    - Keep placeholder tokens <URL> and <USER>
    - Drop tokens containing digits or non-letters (besides apostrophes)
    - Remove very short tokens (length < 2) except placeholders
    - English stop words are removed downstream by vectorizer if desired
    """
    text = _normalize_basic(text)

    # Token pattern: words with optional internal apostrophes
    tokens = re.findall(r"<url>|<user>|[a-z]+(?:'[a-z]+)?", text)

    # Filter out single-letter words except placeholders
    filtered = [t for t in tokens if t in {"<url>", "<user>"} or len(t) >= 2]
    return filtered

