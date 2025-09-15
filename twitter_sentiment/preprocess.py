import re
from dataclasses import dataclass
from typing import List, Optional, Dict


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
_APOSTROPHE_RE = re.compile(r"’")  # normalize curly apostrophe


@dataclass
class PreprocessorConfig:
    lowercase: bool = True
    max_repeat_char: int = 3  # 0 disables
    hashtag_policy: str = 'strip'  # strip|keep|segment
    replace_url: bool = True
    replace_user: bool = True
    number_policy: str = 'drop'  # keep|drop|token
    emoji_policy: str = 'ignore'  # ignore|drop|map
    emoticons: bool = True  # map basic emoticons
    contractions: Optional[Dict[str, str]] = None  # default off for back-compat
    negation_scope: Optional[int] = None  # 0/None disables


# Minimal emoji/emoticon/contractions tables (extendable)
EMOJI_MAP = {
    '😀': 'smile', '😃': 'smile', '😄': 'smile', '😁': 'smile', '🙂': 'smile', '😊': 'smile',
    '😂': 'laugh', '🤣': 'laugh',
    '😍': 'love', '😘': 'kiss', '😢': 'sad', '😭': 'sad', '😡': 'angry', '😠': 'angry',
    '👍': 'thumbs_up', '👎': 'thumbs_down',
}

EMOTICON_MAP = {
    ':)': 'smile', ':-)': 'smile', ':D': 'smile', ':-D': 'smile',
    ':(': 'sad', ':-(': 'sad', ':\'(': 'sad',
    ';)': 'wink', ';-)': 'wink', ':P': 'playful', ':-P': 'playful',
}

DEFAULT_CONTRACTIONS = {
    "can't": 'can not', "won't": 'will not', "don't": 'do not',
    "i'm": 'i am', "it's": 'it is', "you're": 'you are', "we're": 'we are',
    "they're": 'they are', "that's": 'that is', "there's": 'there is',
    "i've": 'i have', "i'd": 'i would', "i'll": 'i will',
}


def _normalize_quotes_and_case(text: str, cfg: PreprocessorConfig) -> str:
    text = text.strip()
    text = _APOSTROPHE_RE.sub("'", text)
    if cfg.lowercase:
        text = text.lower()
    return text


def _replace_urls_users(text: str, cfg: PreprocessorConfig) -> str:
    if cfg.replace_url:
        text = _URL_RE.sub(" <url> ", text)
    if cfg.replace_user:
        text = _MENTION_RE.sub(" <user> ", text)
    return text


def _apply_contractions(text: str, cfg: PreprocessorConfig) -> str:
    if not cfg.contractions:
        return text
    # simple word-boundary replacement
    for k, v in cfg.contractions.items():
        # assume keys are lowercase; text already lowercased if cfg.lowercase
        text = re.sub(rf"\b{k}\b", v, text)
    return text


def _map_emoji_emoticons(text: str, cfg: PreprocessorConfig) -> str:
    out = text
    if cfg.emoji_policy == 'map':
        for ch, tok in EMOJI_MAP.items():
            out = out.replace(ch, f" {tok} ")
    elif cfg.emoji_policy == 'drop':
        for ch in EMOJI_MAP.keys():
            out = out.replace(ch, ' ')
    # emoticons are ASCII sequences; process regardless of emoji policy when enabled
    if cfg.emoticons:
        for em, tok in EMOTICON_MAP.items():
            out = out.replace(em, f" {tok} ")
    return out


def _apply_hashtag_policy(text: str, cfg: PreprocessorConfig) -> str:
    def _strip(m: re.Match) -> str:
        return ' ' + m.group()[1:] + ' '

    def _keep(m: re.Match) -> str:
        word = m.group()[1:]
        return ' ' + f'hashtag_{word}' + ' '

    def _segment_word(w: str) -> List[str]:
        # naive heuristic: split on underscores and case changes; fallback to original
        parts = re.split(r'[_]+', w)
        segmented: List[str] = []
        for p in parts:
            if not p:
                continue
            # split CamelCase boundaries
            splits = re.findall(r'[a-z]+|[0-9]+', p)
            if len(splits) > 1:
                segmented.extend(splits)
            else:
                segmented.append(p)
        return segmented or [w]

    def _segment(m: re.Match) -> str:
        word = m.group()[1:]
        seg = _segment_word(word)
        return ' ' + ' '.join(seg) + ' '

    if cfg.hashtag_policy == 'strip':
        return _HASHTAG_RE.sub(_strip, text)
    if cfg.hashtag_policy == 'keep':
        return _HASHTAG_RE.sub(_keep, text)
    if cfg.hashtag_policy == 'segment':
        return _HASHTAG_RE.sub(_segment, text)
    return _HASHTAG_RE.sub(_strip, text)


def _normalize_repeats(text: str, cfg: PreprocessorConfig) -> str:
    if not cfg.max_repeat_char or cfg.max_repeat_char <= 0:
        return text
    n = cfg.max_repeat_char
    return re.sub(rf"(.)\1{{{n},}}", lambda m: m.group(1) * n, text)


def _normalize_numbers(text: str, cfg: PreprocessorConfig) -> str:
    if cfg.number_policy == 'token':
        return re.sub(r"\d+", " <num> ", text)
    # 'keep' or 'drop' handled at tokenization stage; here we do nothing
    return text


def _tokenize(text: str, cfg: PreprocessorConfig) -> List[str]:
    # token patterns depend on number policy
    if cfg.number_policy == 'keep':
        pattern = r"<url>|<user>|<num>|[a-z0-9]+(?:'[a-z0-9]+)?"
    else:
        pattern = r"<url>|<user>|<num>|[a-z]+(?:'[a-z]+)?"
    tokens = re.findall(pattern, text)
    # Short token filter: keep placeholders; keep len>=2 words; keep numbers if policy keep or token
    placeholders = {"<url>", "<user>", "<num>"}
    filtered: List[str] = []
    for t in tokens:
        if t in placeholders:
            filtered.append(t)
        elif cfg.number_policy == 'keep' and any(ch.isdigit() for ch in t):
            filtered.append(t)
        elif len(t) >= 2:
            filtered.append(t)
    return filtered


def _apply_negation_scope(tokens: List[str], cfg: PreprocessorConfig) -> List[str]:
    scope = cfg.negation_scope if cfg.negation_scope else 0
    if scope <= 0:
        return tokens
    neg_cues = {"not", "no", "never", "n't"}
    out: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)
        if tok in neg_cues:
            j = 0
            while j < scope and (i + 1 + j) < len(tokens):
                nxt = tokens[i + 1 + j]
                if nxt not in {"<url>", "<user>", "<num>"}:
                    out.append(f"{nxt}_neg")
                else:
                    out.append(nxt)
                j += 1
            i += scope  # skip over ones we negated
        i += 1
    return out


class Tokenizer:
    """Picklable callable tokenizer that applies configurable preprocessing."""

    def __init__(self, cfg: Optional[PreprocessorConfig] = None):
        self.cfg = cfg or PreprocessorConfig()

    def __call__(self, text: str) -> List[str]:
        cfg = self.cfg
        x = _normalize_quotes_and_case(text, cfg)
        x = _replace_urls_users(x, cfg)
        if cfg.contractions is not None and cfg.contractions is True:  # backward safety if passed True
            cfg.contractions = DEFAULT_CONTRACTIONS
        if cfg.contractions is None:
            # default None keeps back-compat: no expansion
            pass
        elif cfg.contractions == {}:
            pass
        else:
            if cfg.contractions == 'default':  # in case str passed accidentally
                cfg.contractions = DEFAULT_CONTRACTIONS
            x = _apply_contractions(x, cfg)
        x = _map_emoji_emoticons(x, cfg)
        x = _apply_hashtag_policy(x, cfg)
        x = _normalize_repeats(x, cfg)
        x = _normalize_numbers(x, cfg)
        toks = _tokenize(x, cfg)
        toks = _apply_negation_scope(toks, cfg)
        return toks


# Backward-compatible entry point
def preprocess_and_tokenize(text: str, cfg: Optional[PreprocessorConfig] = None) -> List[str]:
    return Tokenizer(cfg)(text)

