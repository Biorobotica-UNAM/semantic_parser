"""
Natural language entity extractor using YAML-driven labels.

Design:
    Facade — SentenceEntityExtractor is the public API.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Value Object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """Immutable result of a single entity-extraction pass."""

    label: str
    value: Optional[str | list[str]] = None

    def __bool__(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        return f"{self.label}='{self.value}'" if self.value else f"{self.label}=None"


# ---------------------------------------------------------------------------
# YAML Config Loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityConfig:
    label: str
    patterns: list[str]


class EntityConfigLoader:
    """Loads EntityConfig objects from YAML files."""

    @staticmethod
    def load_all(config_dir: Path) -> list[EntityConfig]:
        configs = []

        for path in sorted(config_dir.glob("*.yaml")):
            with path.open() as f:
                data = yaml.safe_load(f)

            configs.append(EntityConfigLoader._parse(data, path))

        return configs

    @staticmethod
    def _parse(data: dict, path: Path) -> EntityConfig:
        label = data.get("label")
        patterns = data.get("patterns", [])

        if not label:
            raise ValueError(f"Missing 'label' in {path}")

        return EntityConfig(
            label=label.upper(),
            patterns=[p.lower().replace("_", " ") for p in patterns],
        )


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class SentenceEntityExtractor:
    """
    Public facade for label-driven entity extraction.
    """

    _SINGLE_VALUE_LABELS = {"NAME", "FOLLOW", "CONFIRM"}

    def __init__(self, config_dir: Path) -> None:
        self._catalog: dict[str, set[str]] = {}
        self._build_catalog(config_dir)

    def _build_catalog(self, config_dir: Path) -> None:
        configs = EntityConfigLoader.load_all(config_dir)

        for cfg in configs:
            self._catalog[cfg.label] = set(cfg.patterns)

    def extract(self, sentence: str, label: str) -> ExtractionResult:
        """
        Extract entities matching *label* from sentence.

        Two key guarantees:
          1. Word-boundary matching  — "cola" won't fire inside "chocolate".
          2. Match consumption       — once "chocolate milk" is matched, the
             characters it occupied are blanked out so "milk" (and "chocolate")
             can't fire again on the same span.
        """

        sentence = sentence.lower().strip()
        label = label.upper().strip()

        if label not in self._catalog:
            return ExtractionResult(label=label)

        vocabulary = self._catalog[label]

        # Longest patterns first — greedy, so "chocolate milk" wins over "milk".
        ordered_patterns = sorted(vocabulary, key=len, reverse=True)

        # Working copy we'll blank out as patterns are consumed.
        remaining = sentence
        matches: list[str] = []

        for pattern in ordered_patterns:
            # \b anchors ensure we match whole words only.
            # re.escape handles any special characters in the pattern.
            regex = r"\b" + re.escape(pattern) + r"\b"

            if re.search(regex, remaining):
                normalized = pattern.replace(" ", "_")

                if normalized not in matches:
                    matches.append(normalized)

                # Blank out every occurrence so sub-patterns can't re-match
                # the same text (e.g. erase "chocolate milk" → "              "
                # so lone "milk" and "chocolate" no longer appear there).
                remaining = re.sub(regex, " " * len(pattern), remaining)

        if not matches:
            return ExtractionResult(label=label)

        if label in self._SINGLE_VALUE_LABELS:
            return ExtractionResult(label=label, value=matches[0])

        return ExtractionResult(label=label, value=matches)
