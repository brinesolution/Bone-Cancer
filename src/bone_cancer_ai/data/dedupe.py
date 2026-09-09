"""Exact, canonical-ID, and perceptual duplicate detection for audit manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import pandas as pd

EDGE_COLUMNS = (
    "record_id",
    "image_id",
    "other_record_id",
    "other_image_id",
    "match_type",
    "phash_distance",
    "official_split",
    "other_split",
    "cross_split",
    "label_conflict",
    "width",
    "height",
    "other_width",
    "other_height",
)
GROUPED_EDGE_COLUMNS = ("duplicate_group", *EDGE_COLUMNS)
EVIDENCE_ORDER = ("exact_sha256", "canonical_source_id", "perceptual_phash")
REQUIRED_COLUMNS = {
    "record_id",
    "image_id",
    "official_split",
    "sha256",
    "phash",
    "canonical_source_id",
    "binary_label",
    "width",
    "height",
}


def _normalize_phash(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, int):
        candidate = f"{value:016x}"
    elif isinstance(value, float) and value.is_integer():
        candidate = f"{int(value):016x}"
    else:
        candidate = str(value).strip().lower()
    if len(candidate) != 16:
        return None
    try:
        int(candidate, 16)
    except ValueError:
        return None
    return candidate


def hamming_distance(left: str, right: str) -> int:
    """Return the bit distance between two complete 64-bit hexadecimal hashes."""
    left_hash = _normalize_phash(left)
    right_hash = _normalize_phash(right)
    if left_hash is None or right_hash is None:
        raise ValueError("Perceptual hashes must be 16 hexadecimal characters")
    return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()


class _BKTree:
    """Small metric tree used to avoid an all-pairs perceptual-hash comparison."""

    def __init__(self) -> None:
        self._root: dict[str, Any] | None = None

    def add(self, value: int, row_index: int) -> None:
        if self._root is None:
            self._root = {"value": value, "rows": [row_index], "children": {}}
            return
        node = self._root
        while True:
            distance = (value ^ node["value"]).bit_count()
            if distance == 0:
                node["rows"].append(row_index)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {
                    "value": value,
                    "rows": [row_index],
                    "children": {},
                }
                return
            node = child

    def query(self, value: int, maximum_distance: int) -> list[tuple[int, int]]:
        if self._root is None:
            return []
        matches: list[tuple[int, int]] = []
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = (value ^ node["value"]).bit_count()
            if distance <= maximum_distance:
                matches.extend((row_index, distance) for row_index in node["rows"])
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            pending.extend(
                child
                for edge_distance, child in node["children"].items()
                if lower <= edge_distance <= upper
            )
        return matches


def _nonempty(value: Any) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _add_group_pairs(
    groups: dict[str, list[int]], evidence: str, pair_evidence: dict[tuple[int, int], set[str]]
) -> None:
    for indexes in groups.values():
        for left, right in combinations(sorted(indexes), 2):
            pair_evidence[(left, right)].add(evidence)


def _different_labels(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    return str(left) != str(right)


def find_duplicate_edges(master: pd.DataFrame, max_phash_distance: int = 4) -> pd.DataFrame:
    """Find pairwise duplicate evidence without changing official split membership."""
    missing = REQUIRED_COLUMNS - set(master.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if master["record_id"].isna().any() or master["record_id"].duplicated().any():
        raise ValueError("Manifest record_id values must be present and unique")
    if not 0 <= max_phash_distance <= 64:
        raise ValueError("max_phash_distance must be between 0 and 64")

    ordered = master.sort_values("record_id").reset_index(drop=True)
    pair_evidence: dict[tuple[int, int], set[str]] = defaultdict(set)
    pair_distance: dict[tuple[int, int], int] = {}

    sha_groups: dict[str, list[int]] = defaultdict(list)
    canonical_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in ordered.iterrows():
        if _nonempty(row["sha256"]):
            sha_groups[str(row["sha256"]).strip().lower()].append(index)
        if _nonempty(row["canonical_source_id"]):
            canonical_groups[str(row["canonical_source_id"]).strip().lower()].append(index)
    _add_group_pairs(sha_groups, "exact_sha256", pair_evidence)
    _add_group_pairs(canonical_groups, "canonical_source_id", pair_evidence)

    tree = _BKTree()
    normalized_hashes: dict[int, str] = {}
    for index, value in enumerate(ordered["phash"]):
        normalized = _normalize_phash(value)
        if normalized is None:
            continue
        normalized_hashes[index] = normalized
        integer_hash = int(normalized, 16)
        for other_index, distance in tree.query(integer_hash, max_phash_distance):
            pair = (other_index, index)
            pair_evidence[pair].add("perceptual_phash")
            pair_distance[pair] = distance
        tree.add(integer_hash, index)

    rows: list[dict[str, Any]] = []
    for (left_index, right_index), evidence in sorted(pair_evidence.items()):
        left = ordered.iloc[left_index]
        right = ordered.iloc[right_index]
        distance = pair_distance.get((left_index, right_index))
        if (
            distance is None
            and left_index in normalized_hashes
            and right_index in normalized_hashes
        ):
            distance = hamming_distance(
                normalized_hashes[left_index], normalized_hashes[right_index]
            )
        rows.append(
            {
                "record_id": left["record_id"],
                "image_id": left["image_id"],
                "other_record_id": right["record_id"],
                "other_image_id": right["image_id"],
                "match_type": "|".join(item for item in EVIDENCE_ORDER if item in evidence),
                "phash_distance": distance,
                "official_split": left["official_split"],
                "other_split": right["official_split"],
                "cross_split": left["official_split"] != right["official_split"],
                "label_conflict": _different_labels(
                    left["binary_label"], right["binary_label"]
                ),
                "width": left["width"],
                "height": left["height"],
                "other_width": right["width"],
                "other_height": right["height"],
            }
        )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


class _UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def build_duplicate_groups(master: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Assign stable IDs to connected components of duplicate-pair evidence."""
    del master  # The edge table already carries all fields required for grouping.
    if edges.empty:
        return pd.DataFrame(columns=GROUPED_EDGE_COLUMNS)
    records = set(edges["record_id"].astype(str)) | set(edges["other_record_id"].astype(str))
    components = _UnionFind(records)
    for row in edges.itertuples(index=False):
        components.union(str(row.record_id), str(row.other_record_id))
    component_members: dict[str, list[str]] = defaultdict(list)
    for record in sorted(records):
        component_members[components.find(record)].append(record)
    ordered_components = sorted(component_members.values(), key=lambda members: members[0])
    group_for_record = {
        record: f"dup_{group_number:06d}"
        for group_number, members in enumerate(ordered_components, 1)
        for record in members
    }
    grouped = edges.copy()
    grouped.insert(0, "duplicate_group", grouped["record_id"].map(group_for_record))
    return grouped.sort_values(
        ["duplicate_group", "record_id", "other_record_id"]
    ).reset_index(drop=True)


def summarize_leakage(grouped_edges: pd.DataFrame) -> dict[str, Any]:
    """Summarize duplicate components and the cross-split review workload."""
    if grouped_edges.empty:
        return {
            "duplicate_groups": 0,
            "duplicate_edges": 0,
            "images_in_duplicate_groups": 0,
            "cross_split_edges": 0,
            "cross_split_groups": 0,
            "cross_split_exact_edges": 0,
            "cross_split_canonical_edges": 0,
            "cross_split_perceptual_edges": 0,
            "label_conflict_edges": 0,
            "exact_label_conflict_edges": 0,
            "evidence_counts": {evidence: 0 for evidence in EVIDENCE_ORDER},
            "cross_split_pair_counts": {},
        }
    records = set(grouped_edges["record_id"].astype(str)) | set(
        grouped_edges["other_record_id"].astype(str)
    )
    evidence_counts = Counter(
        evidence
        for match_type in grouped_edges["match_type"].astype(str)
        for evidence in match_type.split("|")
        if evidence
    )
    cross_split_pairs = Counter()
    for row in grouped_edges.loc[grouped_edges["cross_split"]].itertuples(index=False):
        pair = "|".join(sorted((str(row.official_split), str(row.other_split))))
        cross_split_pairs[pair] += 1
    cross_split = grouped_edges["cross_split"].astype(bool)
    label_conflict = grouped_edges["label_conflict"].astype(bool)
    match_types = grouped_edges["match_type"].astype(str)
    return {
        "duplicate_groups": int(grouped_edges["duplicate_group"].nunique()),
        "duplicate_edges": len(grouped_edges),
        "images_in_duplicate_groups": len(records),
        "cross_split_edges": int(cross_split.sum()),
        "cross_split_groups": int(grouped_edges.loc[cross_split, "duplicate_group"].nunique()),
        "cross_split_exact_edges": int(
            (cross_split & match_types.str.contains("exact_sha256", regex=False)).sum()
        ),
        "cross_split_canonical_edges": int(
            (cross_split & match_types.str.contains("canonical_source_id", regex=False)).sum()
        ),
        "cross_split_perceptual_edges": int(
            (cross_split & match_types.str.contains("perceptual_phash", regex=False)).sum()
        ),
        "label_conflict_edges": int(label_conflict.sum()),
        "exact_label_conflict_edges": int(
            (label_conflict & match_types.str.contains("exact_sha256", regex=False)).sum()
        ),
        "evidence_counts": {
            evidence: int(evidence_counts[evidence]) for evidence in EVIDENCE_ORDER
        },
        "cross_split_pair_counts": {
            pair: int(count) for pair, count in sorted(cross_split_pairs.items())
        },
    }
