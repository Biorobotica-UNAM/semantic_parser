"""
Natural language entity extractor using spaCy.

Responsibility: given a raw sentence, extract named entities
(PERSON, DRINK) and return them as an ExtractionResult value object.

The node itself knows nothing about ROS2 — this module is pure Python
and can be used standalone or from any ROS2 node.

Design:
    Chain of Responsibility (GoF) — each handler extracts one entity type.
    Factory Method               — NlpPipelineFactory builds the spaCy pipeline.
    Facade                       — SentenceEntityExtractor is the only public API.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

# ---------------------------------------------------------------------------
# Value Object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """Immutable result of a single entity-extraction pass."""

    name: Optional[str] = None
    drink: Optional[str] = None

    def __str__(self) -> str:
        parts = []
        if self.name:
            parts.append(f"name='{self.name}'")
        if self.drink:
            parts.append(f"drink='{self.drink}'")
        return " | ".join(parts) if parts else "no entities found"


# ---------------------------------------------------------------------------
# Abstract Handler — Chain of Responsibility (GoF)
# ---------------------------------------------------------------------------


class EntityHandler(abc.ABC):
    """
    Abstract handler in the extraction chain.

    Each concrete handler tries to extract ONE entity type.
    If the entity was already extracted earlier in the chain,
    the handler skips and delegates to the next link (OCP).
    """

    def __init__(self) -> None:
        self._next: Optional[EntityHandler] = None

    def set_next(self, handler: EntityHandler) -> EntityHandler:
        """Chain the next handler; returns it for fluent chaining."""
        self._next = handler
        return handler

    def handle(self, doc: Doc, result: dict) -> dict:
        result = self._extract(doc, result)
        if self._next:
            return self._next.handle(doc, result)
        return result

    @abc.abstractmethod
    def _extract(self, doc: Doc, result: dict) -> dict:
        """Entity-specific extraction logic."""


# ---------------------------------------------------------------------------
# Concrete Handlers (SRP: one class — one entity type)
# ---------------------------------------------------------------------------


class PersonNameHandler(EntityHandler):
    """
    Extracts person names using spaCy's statistical NER (label: PERSON).
    Recommended in Mattingly ch.3: filter doc.ents by label_.
    """

    def _extract(self, doc: Doc, result: dict) -> dict:
        if result.get("name"):
            return result
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                result["name"] = ent.text
                break
        return result


class DrinkEntityRulerHandler(EntityHandler):
    """
    Extracts drinks using the spaCy EntityRuler (rule-based, label: DRINK).
    Mattingly's key insight: use rules before ML for domain-specific entities.
    """

    def _extract(self, doc: Doc, result: dict) -> dict:
        if result.get("drink"):
            return result
        for ent in doc.ents:
            if ent.label_ == "DRINK":
                result["drink"] = ent.text
                break
        return result


class DrinkTokenFallbackHandler(EntityHandler):
    """
    Fallback: if the EntityRuler missed the drink token, match by string.
    Covers edge cases like unknown spellings or tokenization quirks.
    """

    _KNOWN_DRINKS = {
        "coffee",
        "tea",
        "iced_tea",
        "water",
        "juice",
        "lemonade",
        "soda",
        "milk",
        "beer",
        "wine",
        "espresso",
        "cappuccino",
        "latte",
    }

    def _extract(self, doc: Doc, result: dict) -> dict:
        if result.get("drink"):
            return result
        for token in doc:
            normalized = token.text.lower().replace(" ", "_")
            if (
                normalized in self._KNOWN_DRINKS
                or token.text.lower() in self._KNOWN_DRINKS
            ):
                result["drink"] = token.text
                break
        return result


# ---------------------------------------------------------------------------
# Pipeline Factory — Factory Method (GoF)
# ---------------------------------------------------------------------------


class NlpPipelineFactory:
    """
    Builds and configures the spaCy pipeline.

    Inserts the EntityRuler BEFORE the statistical NER so that
    rule-matched DRINK entities are preserved (Mattingly pattern).
    """

    _DRINK_PATTERNS = [
        {"label": "DRINK", "pattern": "coffee"},
        {"label": "DRINK", "pattern": "tea"},
        {"label": "DRINK", "pattern": [{"LOWER": "iced"}, {"LOWER": "tea"}]},
        {"label": "DRINK", "pattern": "iced_tea"},
        {"label": "DRINK", "pattern": "water"},
        {"label": "DRINK", "pattern": "juice"},
        {"label": "DRINK", "pattern": "lemonade"},
        {"label": "DRINK", "pattern": "soda"},
        {"label": "DRINK", "pattern": "milk"},
        {"label": "DRINK", "pattern": "beer"},
        {"label": "DRINK", "pattern": "wine"},
        {"label": "DRINK", "pattern": "espresso"},
        {"label": "DRINK", "pattern": "cappuccino"},
        {"label": "DRINK", "pattern": "latte"},
    ]

    @classmethod
    def create(cls, model: str) -> Language:
        nlp = spacy.load(model)
        cls._add_drink_ruler(nlp)
        return nlp

    @classmethod
    def _add_drink_ruler(cls, nlp: Language) -> None:
        ruler = nlp.add_pipe(
            "entity_ruler",
            before="ner",
            config={"overwrite_ents": False},
        )
        ruler.add_patterns(cls._DRINK_PATTERNS)


# ---------------------------------------------------------------------------
# Chain Builder
# ---------------------------------------------------------------------------


class ExtractionChainBuilder:
    """Constructs the handler chain in the correct order."""

    @staticmethod
    def build() -> EntityHandler:
        head = PersonNameHandler()
        head.set_next(DrinkEntityRulerHandler()).set_next(DrinkTokenFallbackHandler())
        return head


# ---------------------------------------------------------------------------
# Facade — public API (DIP: depends on abstractions, not concretions)
# ---------------------------------------------------------------------------


class SentenceEntityExtractor:
    """
    Public facade for entity extraction.

    Usage:
        extractor = SentenceEntityExtractor(model="en_core_web_trf")
        result = extractor.extract("Hello, my name is Morgan")
        print(result.name)   # "Morgan"
        print(result.drink)  # None
    """

    def __init__(self, model: str = "en_core_web_trf") -> None:
        self._nlp: Language = NlpPipelineFactory.create(model)
        self._chain: EntityHandler = ExtractionChainBuilder.build()

    def extract(self, sentence: str) -> ExtractionResult:
        """Process one sentence and return the extracted entities."""
        doc = self._nlp(sentence)
        raw: dict = self._chain.handle(doc, {})
        return ExtractionResult(
            name=raw.get("name"),
            drink=raw.get("drink"),
        )
