from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as exc:  # pragma: no cover - dependency import error
    raise RuntimeError(
        "NLTK is required for the preprocessing service. Install nltk before running the app."
    ) from exc


_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_FALLBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


class TokenizationStage:
    def process(self, text: str) -> list[str]:
        try:
            return word_tokenize(text)
        except (LookupError, OSError):
            return _TOKEN_PATTERN.findall(text)


class LowercasingStage:
    def process(self, tokens: list[str]) -> list[str]:
        return [token.lower() for token in tokens]


class StopWordsRemovalStage:
    def __init__(self, language: str = "english") -> None:
        self._stop_words = self._load_stop_words(language)

    def _load_stop_words(self, language: str) -> set[str]:
        try:
            return set(stopwords.words(language))
        except (LookupError, OSError):
            return _FALLBACK_STOP_WORDS

    def process(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in self._stop_words]


class StemmingStage:
    def __init__(self, language: str = "english") -> None:
        self._stemmer = SnowballStemmer(language)

    def process(self, tokens: list[str]) -> list[str]:
        return [self._stemmer.stem(token) for token in tokens]


class LemmatizationStage:
    def __init__(self) -> None:
        self._lemmatizer = WordNetLemmatizer()

    def process(self, tokens: list[str]) -> list[str]:
        lemmatized_tokens: list[str] = []
        for token in tokens:
            try:
                lemmatized_tokens.append(self._lemmatizer.lemmatize(token))
            except (LookupError, OSError):
                lemmatized_tokens.append(token)
        return lemmatized_tokens


@dataclass(frozen=True)
class PreprocessingPipeline:
    tokenization_stage: TokenizationStage
    lowercasing_stage: LowercasingStage
    stop_words_stage: StopWordsRemovalStage
    stemming_stage: StemmingStage
    lemmatization_stage: LemmatizationStage

    @classmethod
    def default(cls) -> "PreprocessingPipeline":
        return cls(
            tokenization_stage=TokenizationStage(),
            lowercasing_stage=LowercasingStage(),
            stop_words_stage=StopWordsRemovalStage(),
            stemming_stage=StemmingStage(),
            lemmatization_stage=LemmatizationStage(),
        )

    def process(self, text: str) -> str:
        tokens = self.tokenization_stage.process(text)
        tokens = self.lowercasing_stage.process(tokens)
        tokens = self.stop_words_stage.process(tokens)
        tokens = self.stemming_stage.process(tokens)
        tokens = self.lemmatization_stage.process(tokens)
        return " ".join(tokens)
