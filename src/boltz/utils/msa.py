"""Utilities for working with multiple-sequence alignments (MSAs)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
LOG_ALPHA = np.log2(len(AA_ALPHABET) + 1)


def clean_msa_sequence(raw_sequence: str) -> list[str]:
    """Convert an MSA sequence into alignment columns.

    Lowercase characters represent insertions relative to the query and are
    ignored when building the column alignment. Uppercase amino-acid letters
    and gaps are retained.
    """

    columns: list[str] = []
    for char in raw_sequence.strip():
        if char != "-" and char.islower():
            # Insertions are skipped but do not advance the column index.
            continue
        if char == "-":
            columns.append("-")
        else:
            columns.append(char.upper())

    return columns


def extract_query_from_msa(msa_path: Path) -> str:
    """Heuristically recover the query sequence from an MSA CSV file."""

    data = pd.read_csv(msa_path)
    if "sequence" not in data.columns:
        msg = f"MSA file {msa_path} is missing the 'sequence' column"
        raise ValueError(msg)

    for raw_sequence in data["sequence"]:
        if not isinstance(raw_sequence, str):
            continue
        processed = [char for char in raw_sequence.strip() if char.isalpha()]
        if not processed:
            continue
        query = "".join(processed).upper()
        if query:
            return query

    msg = f"MSA file {msa_path} does not contain a valid query sequence"
    raise ValueError(msg)


def _sequence_entropy(counts: Dict[str, int]) -> float:
    """Compute the entropy for the residue distribution at a column."""

    total = sum(counts.values())
    if total == 0:
        return 0.0

    probabilities = np.array(list(counts.values()), dtype=np.float32) / total
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def _residue_conservation(score: float) -> float:
    """Normalise a conservation score to the range [0, 1]."""

    return float((LOG_ALPHA - score) / LOG_ALPHA)


def compute_query_conservation(msa_path: Path, query_sequence: str) -> np.ndarray:
    """Compute per-residue conservation scores from an MSA CSV."""

    data = pd.read_csv(msa_path)
    if "sequence" not in data.columns:
        msg = f"MSA file {msa_path} is missing the 'sequence' column"
        raise ValueError(msg)

    processed_sequences: list[list[str]] = []
    query_index: Optional[int] = None

    for idx, raw_sequence in enumerate(data["sequence"]):
        if not isinstance(raw_sequence, str):
            continue

        aligned = clean_msa_sequence(raw_sequence)
        if not aligned:
            continue
        processed_sequences.append(aligned)

        if query_index is None:
            stripped = "".join(char for char in aligned if char != "-")
            if stripped.upper() == query_sequence.upper():
                query_index = len(processed_sequences) - 1

    if not processed_sequences:
        msg = f"MSA file {msa_path} does not contain valid sequences"
        raise ValueError(msg)

    alignment_lengths = {len(seq) for seq in processed_sequences}
    if len(alignment_lengths) != 1:
        msg = (
            f"MSA file {msa_path} has inconsistent alignment lengths: "
            f"{alignment_lengths}"
        )
        raise ValueError(msg)

    if query_index is None:
        msg = (
            f"Could not locate the query sequence in MSA {msa_path}. "
            "Ensure the query is present as an aligned sequence."
        )
        raise ValueError(msg)

    alignment_length = alignment_lengths.pop()
    column_scores = np.zeros(alignment_length, dtype=np.float32)

    for col_idx in range(alignment_length):
        counts: Dict[str, int] = {}
        for seq in processed_sequences:
            residue = seq[col_idx]
            if residue == "-":
                continue
            counts[residue] = counts.get(residue, 0) + 1

        column_scores[col_idx] = _residue_conservation(_sequence_entropy(counts))

    conservation: list[float] = []
    query_aligned = processed_sequences[query_index]

    for res_char, score in zip(query_aligned, column_scores):
        if res_char == "-":
            continue
        conservation.append(score)

    return np.asarray(conservation, dtype=np.float32)


__all__ = [
    "clean_msa_sequence",
    "compute_query_conservation",
    "extract_query_from_msa",
]
