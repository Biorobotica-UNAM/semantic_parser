"""
nlp/spacy_pipeline.py

spaCy pipeline factory for the semantic parser.

Pipe order
──────────
    tok2vec → tagger → parser → attribute_ruler → lemmatizer
    → entity_ruler  (BEFORE ner)
    → ner
    → person_desc_tagger  (AFTER ner)

Factory registration note
─────────────────────────
spaCy's @Language.factory validates the factory function signature via
inspect.signature() and requires exactly (nlp, name).  Defining the
factory inside an if-block or a nested scope can fool the inspector in
some spaCy versions, producing E964 even with a correct signature.

Safe pattern: define the factory function at module level (top of file)
so it is a plain, inspectable function when the module is imported.
The Matcher instance is passed through the module-level variable
_PERSON_DESC_MATCHER, which is written before add_pipe() is called.

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

from typing import List, Optional

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

from semantic_parser.domain.entities import (
    BODY_PARTS,
    BODY_SIDES,
    CLOTHING_COLORS,
    CLOTHING_ITEMS,
    LOCATIONS,
    OBJECT_CATEGORIES,
    PERCEPTUAL_ATTRIBUTES,
    PERSON_NAMES,
    PERSON_POSES,
    SUPERLATIVE_MAP,
)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level Matcher store — written by SpacyPipelineFactory before
# add_pipe() is called; read by the factory function below.
# ──────────────────────────────────────────────────────────────────────────────
_PERSON_DESC_MATCHER: Optional[Matcher] = None

_COMPONENT_NAME = "person_desc_tagger"


# ──────────────────────────────────────────────────────────────────────────────
# Factory function — defined at module level so spaCy's signature
# inspector always sees a plain (nlp, name) function, not a closure.
# ──────────────────────────────────────────────────────────────────────────────
def _person_desc_tagger_factory(nlp: Language, name: str) -> "PersonDescTagger":
    """spaCy factory for PersonDescTagger. Signature must be (nlp, name)."""
    return PersonDescTagger(_PERSON_DESC_MATCHER)


# Register once when the module is imported.  The guard prevents E964
# if the module is imported multiple times in the same process (e.g. tests).
if not Language.has_factory(_COMPONENT_NAME):
    Language.factory(_COMPONENT_NAME, func=_person_desc_tagger_factory)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ──────────────────────────────────────────────────────────────────────────────


class SpacyPipelineFactory:
    """
    Assembles and returns a domain-enriched spaCy pipeline.

    Usage
    ─────
        nlp = SpacyPipelineFactory.build("en_core_web_trf")
    """

    @staticmethod
    def build(model: str = "en_core_web_trf") -> Language:
        nlp = spacy.load(model)
        SpacyPipelineFactory._inject_entity_ruler(nlp)
        SpacyPipelineFactory._inject_person_desc_tagger(nlp)
        return nlp

    # ── EntityRuler ───────────────────────────────────────────────────────────

    @staticmethod
    def _inject_entity_ruler(nlp: Language) -> None:
        ruler = nlp.add_pipe(
            "entity_ruler", before="ner", config={"overwrite_ents": True}
        )
        patterns = (
            SpacyPipelineFactory._location_patterns()
            + SpacyPipelineFactory._object_patterns()
            + SpacyPipelineFactory._person_name_patterns()
            + SpacyPipelineFactory._physical_attr_patterns()
            + SpacyPipelineFactory._perceptual_attr_patterns()
        )
        ruler.add_patterns(patterns)

    @staticmethod
    def _location_patterns() -> List[dict]:
        patterns = []
        for loc in LOCATIONS:
            words = loc.split()
            if len(words) == 1:
                patterns.append({"label": "LOCATION", "pattern": loc})
            else:
                patterns.append(
                    {
                        "label": "LOCATION",
                        "pattern": [{"LOWER": w} for w in words],
                    }
                )
        return patterns

    @staticmethod
    def _object_patterns() -> List[dict]:
        ABSTRACT = {"drink", "snack", "fruit", "food", "dish", "toy", "cleaning supply"}
        patterns = []
        for obj in OBJECT_CATEGORIES:
            words = obj.split()
            label = "OBJ_CATEGORY" if obj in ABSTRACT else "OBJ_INSTANCE"
            if len(words) == 1:
                patterns.append({"label": label, "pattern": obj})
            else:
                patterns.append(
                    {
                        "label": label,
                        "pattern": [{"LOWER": w} for w in words],
                    }
                )
        return patterns

    @staticmethod
    def _person_name_patterns() -> List[dict]:
        return [{"label": "PERSON_NAME", "pattern": name} for name in PERSON_NAMES]

    @staticmethod
    def _physical_attr_patterns() -> List[dict]:
        return [{"label": "PHYS_ATTR", "pattern": adj} for adj in SUPERLATIVE_MAP]

    @staticmethod
    def _perceptual_attr_patterns() -> List[dict]:
        return [
            {"label": "PERCEPTUAL", "pattern": attr} for attr in PERCEPTUAL_ATTRIBUTES
        ]

    # ── PersonDescTagger ──────────────────────────────────────────────────────

    @staticmethod
    def _inject_person_desc_tagger(nlp: Language) -> None:
        """
        Update the module-level Matcher store and add the pipe.

        The factory is already registered at module-import time.
        Here we only refresh _PERSON_DESC_MATCHER (so the factory
        uses the vocab from the freshly loaded nlp) and call add_pipe.
        """
        global _PERSON_DESC_MATCHER
        _PERSON_DESC_MATCHER = SpacyPipelineFactory._build_person_matcher(nlp)
        nlp.add_pipe(_COMPONENT_NAME, after="ner")

    @staticmethod
    def _build_person_matcher(nlp: Language) -> Matcher:
        """
        Build a Matcher with all person-descriptor patterns.

        P1 — "the [pose] person"         e.g. "the sitting person"
        P2 — "the person wearing a [color?] [garment]"
        P3 — "the person raising/pointing their [side] [body_part]"
        P4 — standalone PERSON_NAME entity token
        """
        matcher = Matcher(nlp.vocab)

        poses_list = list(PERSON_POSES)
        colors_list = list(CLOTHING_COLORS)
        garments_list = list(CLOTHING_ITEMS)
        sides_list = list(BODY_SIDES)
        parts_list = list(BODY_PARTS)

        matcher.add(
            "PERSON_DESC_POSE",
            [
                [
                    {"LOWER": "the"},
                    {"LOWER": {"IN": poses_list}},
                    {"LOWER": "person"},
                ]
            ],
        )

        matcher.add(
            "PERSON_DESC_CLOTHING",
            [
                # P2a: "the person wearing a [color] [garment]"  (color present)
                [
                    {"LOWER": "the"},
                    {"LOWER": "person"},
                    {"LOWER": "wearing"},
                    {"LOWER": {"IN": ["a", "an"]}},
                    {"LOWER": {"IN": colors_list}},
                    {"LOWER": {"IN": garments_list}},
                ],
                # P2b: "the person wearing a [garment]"  (no color / color is ambiguous)
                [
                    {"LOWER": "the"},
                    {"LOWER": "person"},
                    {"LOWER": "wearing"},
                    {"LOWER": {"IN": ["a", "an"]}},
                    {"LOWER": {"IN": garments_list}},
                ],
                # P2c: "the person wearing a [color] t shirt"  (two-token garment)
                [
                    {"LOWER": "the"},
                    {"LOWER": "person"},
                    {"LOWER": "wearing"},
                    {"LOWER": {"IN": ["a", "an"]}},
                    {"LOWER": {"IN": colors_list}, "OP": "?"},
                    {"LOWER": "t"},
                    {"LOWER": "shirt"},
                ],
            ],
        )

        matcher.add(
            "PERSON_DESC_GESTURE",
            [
                [
                    {"LOWER": "the"},
                    {"LOWER": "person"},
                    {"LOWER": {"IN": ["raising", "pointing"]}},
                    {"LOWER": "their"},
                    {"LOWER": {"IN": sides_list}},
                    {"LOWER": {"IN": parts_list}},
                ]
            ],
        )

        matcher.add("PERSON_DESC_NAMED", [[{"ENT_TYPE": "PERSON_NAME"}]])

        return matcher


# ──────────────────────────────────────────────────────────────────────────────
# Custom pipeline component
# ──────────────────────────────────────────────────────────────────────────────


class PersonDescTagger:
    """
    Runs the Matcher over the Doc and promotes matched spans to PERSON_DESC.
    Longer spans win when overlaps occur (spacy.util.filter_spans).
    """

    def __init__(self, matcher: Matcher) -> None:
        self._matcher = matcher

    def __call__(self, doc: Doc) -> Doc:
        if self._matcher is None:
            return doc
        matches = self._matcher(doc)
        new_spans = [
            Span(doc, start, end, label="PERSON_DESC") for _, start, end in matches
        ]
        try:
            doc.ents = spacy.util.filter_spans(list(doc.ents) + new_spans)
        except Exception:
            pass
        return doc
