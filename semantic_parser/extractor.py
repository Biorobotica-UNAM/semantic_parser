"""
Natural language entity extractor using spaCy with YAML-driven labels.

Design:
    Factory Method  — NlpPipelineFactory builds the spaCy pipeline from YAML configs.
    Facade          — SentenceEntityExtractor is the only public API.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
import spacy
from spacy.language import Language
from spacy.tokens import Doc

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
    patterns: list[str | list[dict]]


class EntityConfigLoader:
    """Loads EntityConfig objects from a directory of YAML files."""

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
        raw_patterns = data.get("patterns", [])
        if not label:
            raise ValueError(f"Missing 'label' in {path}")
        patterns = [
            [{"LOWER": token} for token in p.split()] if " " in p else p.lower()
            for p in raw_patterns
        ]
        return EntityConfig(label=label, patterns=patterns)


# ---------------------------------------------------------------------------
# Pipeline Factory
# ---------------------------------------------------------------------------


class NlpPipelineFactory:
    """Builds a spaCy pipeline with an EntityRuler from YAML-derived configs."""

    @classmethod
    def create(cls, model: str, configs: list[EntityConfig]) -> Language:
        nlp = spacy.load(model, exclude=["ner"])
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(cls._build_patterns(configs))
        return nlp

    @staticmethod
    def _build_patterns(configs: list[EntityConfig]) -> list[dict]:
        patterns = []
        for cfg in configs:
            for p in cfg.patterns:
                patterns.append({"label": cfg.label, "pattern": p})
        return patterns


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class SentenceEntityExtractor:
    """
    Public facade for label-driven entity extraction.

    Usage:
        extractor = SentenceEntityExtractor(
            model="en_core_web_trf",
            config_dir=Path("config/"),
        )
        result = extractor.extract("I want a coffee", label="DRINK")
        print(result.value)  # "coffee"
    """

    def __init__(self, model: str, config_dir: Path) -> None:
        configs = EntityConfigLoader.load_all(config_dir)
        self._nlp: Language = NlpPipelineFactory.create(model, configs)

    def extract(self, sentence: str, label: str) -> ExtractionResult:
        """Extract entities matching *label* from sentence."""
        doc: Doc = self._nlp(sentence)
        matches = [
            ent.text.lower().replace(" ", "_")
            for ent in doc.ents
            if ent.label_ == label.upper()
        ]

        if not matches:
            return ExtractionResult(label=label)

        if label.upper() == "NAME":
            return ExtractionResult(label=label, value=matches[0])

        return ExtractionResult(label=label, value=matches)
