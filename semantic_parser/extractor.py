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
        # Carga única al arrancar: modificar los YAML requiere reiniciar el nodo.
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

        if label == "LOCATION":
            return self._extract_location(sentence)
        if label == "CONFIRM":
            return self._extract_confirmation(sentence)

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

    @staticmethod
    def _extract_location(sentence: str) -> ExtractionResult:
        """Extract a proposed destination, leaving existence to navigation.

        Accept a bare name or common destination-request phrases. No semantic
        location dictionary is used: navigation's live service is authoritative.
        """
        text = ' '.join(sentence.lower().split())
        text = text.strip(' .,!?:;\"\'“”‘’()[]{}¡¿')
        prefixes = (
            r'^(?:justina[ ,]+)?(?:please\s+)?'
            r'(?:(?:i would like you to|i want you to|can you|could you)\s+)?'
            r'(?:go to|navigate to|take me to|the location is|the destination is|'
            r'i meant|i mean|instead go to)\s+'
        )
        text = re.sub(prefixes, '', text)
        text = re.sub(r',?\s+please$', '', text)
        text = re.sub(r'^(?:the|a|an)\s+', '', text)
        # Multiple destinations/negations are not a unique navigation answer.
        if re.search(r'\b(?:and|or|then|not|no)\b', text):
            return ExtractionResult(label='LOCATION')
        if not text or not all(c.isalnum() or c in ' _-' for c in text):
            return ExtractionResult(label='LOCATION')
        return ExtractionResult(label='LOCATION', value=[re.sub(r'[\s-]+', '_', text)])

    def _extract_confirmation(self, sentence: str) -> ExtractionResult:
        """Return canonical yes/no; contradictory answers have no value."""
        negative_words = {'no', 'nope', 'wrong', 'incorrect', "that's wrong",
                          "that's incorrect", 'false', 'robot no'}
        patterns = self._catalog.get('CONFIRM', set())
        # Negated positive phrases must not be interpreted as approval.
        positive_patterns = sorted(patterns - negative_words | {'true'}, key=lambda p: (-len(p), p))
        negated = (r"\b(?:not|isn't|is not)\s+(?:" +
                   '|'.join(re.escape(p) for p in positive_patterns) + r')\b')
        negative = bool(re.search(negated, sentence))
        remaining = re.sub(negated, ' ', sentence)
        positive = False
        for pattern in sorted(patterns, key=lambda p: (-len(p), p)):
            regex = r'\b' + re.escape(pattern) + r'\b'
            if re.search(regex, remaining):
                if pattern in negative_words:
                    negative = True
                else:
                    positive = True
                remaining = re.sub(regex, ' ', remaining)
        if positive == negative:
            return ExtractionResult(label='CONFIRM')
        return ExtractionResult(label='CONFIRM', value='no' if negative else 'yes')
