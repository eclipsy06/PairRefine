"""Parse official CASP domain boundaries into residue-level labels."""

import re
from typing import List, Tuple


_DOMAIN_PATTERN = re.compile(r"^D(?P<domain>\d+):(?P<ranges>\d+-\d+(?:,\d+-\d+)*)$")


def parse_domain_boundaries(boundaries: str, sequence_length: int) -> Tuple[List[int], int]:
    """Return zero-based domain labels and the count of covered residues.

    CASP boundaries are one-based and inclusive. Residues outside the provided
    domains remain ``-1`` so callers do not silently invent assignments.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not boundaries or not boundaries.strip():
        raise ValueError("domain boundaries must be non-empty")

    labels = [-1] * sequence_length
    seen_domain_numbers = set()
    for item in boundaries.split(";"):
        match = _DOMAIN_PATTERN.fullmatch(item.strip())
        if match is None:
            raise ValueError(f"invalid domain boundary item: {item!r}")
        domain_number = int(match.group("domain"))
        if domain_number in seen_domain_numbers:
            raise ValueError(f"duplicate domain label D{domain_number}")
        seen_domain_numbers.add(domain_number)
        compact_domain_id = len(seen_domain_numbers) - 1

        for residue_range in match.group("ranges").split(","):
            start_text, end_text = residue_range.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start < 1 or end < start or end > sequence_length:
                raise ValueError(
                    f"range {residue_range!r} is outside a sequence of length {sequence_length}"
                )
            for residue_index in range(start - 1, end):
                if labels[residue_index] != -1:
                    raise ValueError(
                        f"overlapping domain boundaries at residue {residue_index + 1}"
                    )
                labels[residue_index] = compact_domain_id

    return labels, sum(label >= 0 for label in labels)
