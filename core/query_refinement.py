from __future__ import annotations

import re
from dataclasses import dataclass

from spellchecker import SpellChecker

try:
    from nltk.corpus import wordnet as wn
except ImportError as exc:  # pragma: no cover - dependency import error
    raise RuntimeError(
        "NLTK is required for the query refinement service. Install nltk before running the app."
    ) from exc


_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class QueryRefinementResult:
    original_query: str
    corrected_query: str
    expanded_query: str
    expanded_terms: list[str]


class QueryRefinementService:
    def __init__(self) -> None:
        self._spell_checker = SpellChecker()

    def tokenize(self, query: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(query)]

    def correct_spelling(self, tokens: list[str]) -> list[str]:
        corrected_tokens: list[str] = []
        for token in tokens:
            if token.isalpha():
                corrected_tokens.append(self._spell_checker.correction(token) or token)
            else:
                corrected_tokens.append(token)
        return corrected_tokens

    def expand_synonyms(self, tokens: list[str]) -> list[str]:
        expanded_terms: list[str] = []
        seen_terms: set[str] = set()

        for token in tokens:
            try:
                synsets = wn.synsets(token)
            except LookupError:
                synsets = []

            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace("_", " ").lower()
                    if synonym != token and synonym not in seen_terms:
                        seen_terms.add(synonym)
                        expanded_terms.append(synonym)

        return expanded_terms

    def refine(self, query: str) -> QueryRefinementResult:
        original_tokens = self.tokenize(query)
        corrected_tokens = self.correct_spelling(original_tokens)
        expanded_terms = self.expand_synonyms(corrected_tokens)

        corrected_query = " ".join(corrected_tokens)
        if expanded_terms:
            expanded_query = f"{corrected_query} {' '.join(expanded_terms)}"
        else:
            expanded_query = corrected_query

        return QueryRefinementResult(
            original_query=query,
            corrected_query=corrected_query,
            expanded_query=expanded_query,
            expanded_terms=expanded_terms,
        )


def build_query_refinement_service() -> QueryRefinementService:
    return QueryRefinementService()
