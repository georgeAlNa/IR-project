from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def normalize_text(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class IndexedDocumentEntity:
    document_id: str
    processed_text: str
    tokens: tuple[str, ...]
    original_text: str | None = None


@dataclass(frozen=True)
class InvertedIndex:
    postings: dict[str, dict[str, int]]
    document_lengths: dict[str, int]
    document_frequencies: dict[str, int]
    vocabulary: tuple[str, ...]
    average_document_length: float

    @property
    def document_count(self) -> int:
        return len(self.document_lengths)


class InvertedIndexBuilder:
    def build(self, documents: list[IndexedDocumentEntity]) -> InvertedIndex:
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        document_lengths: dict[str, int] = {}

        for document in documents:
            document_lengths[document.document_id] = len(document.tokens)
            term_frequencies = Counter(document.tokens)
            for term, frequency in term_frequencies.items():
                postings[term][document.document_id] = frequency

        document_frequencies = {term: len(posting) for term, posting in postings.items()}
        vocabulary = tuple(sorted(postings))
        average_document_length = (
            sum(document_lengths.values()) / len(document_lengths)
            if document_lengths
            else 0.0
        )

        return InvertedIndex(
            postings={term: dict(posting) for term, posting in postings.items()},
            document_lengths=document_lengths,
            document_frequencies=document_frequencies,
            vocabulary=vocabulary,
            average_document_length=average_document_length,
        )

