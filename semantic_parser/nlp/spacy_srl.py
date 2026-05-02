"""
nlp/spacy_srl.py

Semantic Role Labeler — produces raw action dicts from a spaCy Doc.

Changes vs v1
─────────────
• Entity-first extraction: instead of traversing the dependency tree
  for every field, we first consult doc.ents (enriched by EntityRuler
  and PersonDescTagger) to resolve LOCATION, OBJ_INSTANCE, OBJ_CATEGORY,
  PERSON_NAME, and PERSON_DESC spans.  The dependency tree is only
  consulted when an entity lookup fails.
• PersonDescriptor is populated from PERSON_DESC entity spans, giving
  the downstream pipeline structured person data.
• object_category is propagated from OBJ_INSTANCE/OBJ_CATEGORY labels
  so the CLIPS planner can reason over types without re-parsing strings.
• Clause segmentation logic is unchanged (it worked well in v1).

Output schema (raw action dict)
────────────────────────────────
{
    "action":           str,   # verb lemma or "query"
    "object":           str,   # direct object text
    "object_category":  str,   # semantic category (fruit, drink…) or None
    "source":           str,   # prepositional origin
    "destination":      str,   # prepositional destination
    "location":         str,   # prepositional location
    "actor":            str,   # always "robot"
    "person":           dict,  # PersonDescriptor.to_dict() or {}
}

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

import spacy
from typing import Any, Dict, List, Optional

from semantic_parser.domain.entities import (
    OBJECT_CATEGORY_MAP,
    PERSON_POSES,
    CLOTHING_COLORS,
    CLOTHING_ITEMS,
    BODY_SIDES,
    BODY_PARTS,
)
from semantic_parser.primitives.schema import PersonDescriptor


class SpacySRL:
    """
    Semantic Role Labeler using spaCy's dependency parser + EntityRuler.

    Accepts a pre-built nlp object (injected via constructor) so the
    heavy model is loaded only once (Dependency Injection / IoC).
    """

    _DEST_PREPS = frozenset({"to", "into", "onto", "toward", "towards"})
    _ORIG_PREPS = frozenset({"from", "off"})
    _LOC_PREPS = frozenset(
        {
            "on",
            "in",
            "at",
            "near",
            "beside",
            "inside",
            "next",
            "above",
            "below",
            "behind",
            "front",
            "between",
            "over",
            "under",
        }
    )
    _OBJECT_PREPS = frozenset({"for", "after", "at", "about"})
    _CLAUSE_SEP = frozenset(
        {
            "then",
            "after",
            "afterwards",
            "subsequently",
            "next",
            "finally",
            "also",
            "first",
            "second",
            "third",
        }
    )
    _ANAPHORS = frozenset({"it", "them", "that", "this", "those", "these", "one"})
    _WH_TAGS = frozenset({"WDT", "WP", "WRB", "WP$"})
    _QUERY_VERBS = frozenset(
        {
            "tell",
            "say",
            "report",
            "describe",
            "inform",
            "show",
            "explain",
            "answer",
        }
    )
    _HUMAN_PRONOUNS = frozenset({"me", "us"})
    _TRANSFER_VERBS = frozenset({"give", "bring", "hand", "pass", "offer", "show"})

    def __init__(self, nlp) -> None:
        """
        Args:
            nlp: A spaCy Language object built by SpacyPipelineFactory.
                 Injected to avoid loading the model multiple times.
        """
        self._nlp = nlp

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def parse(self, text: str) -> List[Dict[str, Any]]:
        context = self._make_context()
        doc = self._nlp(text)
        clauses = self._split_clauses(doc)
        results = []
        for clause in clauses:
            action = self._extract_action(clause, context, doc)
            if action:
                results.append(action)
        return results

    # ──────────────────────────────────────────────────────────
    # Context
    # ──────────────────────────────────────────────────────────

    def _make_context(self) -> Dict[str, Optional[str]]:
        return {"last_object": None, "last_location": None}

    def _update_context(
        self,
        context: Dict,
        obj: Optional[str],
        location: Optional[str],
        is_query: bool,
    ) -> None:
        if obj and obj not in self._ANAPHORS and not is_query:
            context["last_object"] = obj
        if location:
            context["last_location"] = location

    # ──────────────────────────────────────────────────────────
    # Clause segmentation (unchanged from v1 — solid logic)
    # ──────────────────────────────────────────────────────────

    def _split_clauses(self, doc) -> List[List]:
        clauses, current = [], []
        for token in doc:
            if self._is_clause_separator(token, current):
                clauses.append(current)
                current = []
                continue
            if self._is_verbal_coordinator(token, current):
                clauses.append(current)
                current = []
                continue
            if self._is_new_conjunct_verb(token, current):
                clauses.append(current)
                current = []
            current.append(token)
        if current:
            clauses.append(current)
        return [c for c in clauses if any(t.pos_ == "VERB" for t in c)]

    def _is_clause_separator(self, token, current: list) -> bool:
        return (
            token.lower_ in self._CLAUSE_SEP
            and bool(current)
            and any(t.pos_ == "VERB" for t in current)
        )

    def _is_verbal_coordinator(self, token, current: list) -> bool:
        if token.lower_ != "and" or token.dep_ != "cc" or token.head.pos_ != "VERB":
            return False
        if not any(t.pos_ == "VERB" for t in current):
            return False
        return token.head.i not in {t.i for t in current}

    def _is_new_conjunct_verb(self, token, current: list) -> bool:
        if token.pos_ != "VERB" or token.dep_ not in ("conj", "dep"):
            return False
        if not any(t.pos_ == "VERB" for t in current):
            return False
        return token.i not in {t.i for t in current}

    # ──────────────────────────────────────────────────────────
    # Action extraction
    # ──────────────────────────────────────────────────────────

    _ANSWER_VERBS = frozenset({"answer", "respond"})

    def _extract_action(
        self, tokens: List, context: Dict, doc
    ) -> Optional[Dict[str, Any]]:
        token_set = {t.i for t in tokens}
        root = self._find_root_verb(tokens)
        if root is None:
            return None

        lemma = root.lemma_.lower()
        is_query = self._is_query_verb(root, lemma)

        obj: Optional[str] = None
        obj_category: Optional[str] = None
        destination: Optional[str] = None
        location: Optional[str] = None
        person_desc = PersonDescriptor()

        if is_query:
            action_str = "query"
            obj = self._extract_query_content(root, token_set)
            destination = self._find_recipient_text(root, token_set)
            # The location lives inside the subordinate clause (ccomp),
            # not as a direct prep child of "tell/say".  Try direct
            # first, then fall back to searching the full subtree.
            location = self._find_prep_arg(
                root, token_set, self._LOC_PREPS
            ) or self._find_location_in_subtree(root, token_set)
        else:
            action_str = lemma

            # ── Entity-first object resolution ────────────────
            ent_obj, ent_cat = self._find_object_entity(token_set, doc)
            if ent_obj:
                obj, obj_category = ent_obj, ent_cat
            else:
                obj = self._find_direct_object(root, token_set, context)
                obj_category = OBJECT_CATEGORY_MAP.get(obj or "", None)

            # ── Entity-first person resolution ────────────────
            person_desc = self._find_person_entity(token_set, doc)

            # ── Destination ───────────────────────────────────
            dest_token = self._find_destination_token(root, token_set)
            location = self._find_prep_arg(root, token_set, self._LOC_PREPS)

            if dest_token:
                destination = self._noun_phrase_base(dest_token, token_set)
                np_loc = self._extract_location_from_np(dest_token, token_set)
                if np_loc:
                    location = np_loc

            destination = self._resolve_recipient_for_transfer(
                root, lemma, destination, token_set
            )

            # ── Answer-verb: extract "of the person..." as recipient ──
            # "Answer the question OF the person pointing to the right in the kitchen"
            # The person is under prep "of", not "to".  Extract them here so
            # MappingHandler receives a structured person and a clean obj.
            if lemma in self._ANSWER_VERBS and destination in (None, "unknown"):
                of_person, of_location = self._find_of_recipient(root, token_set, doc)
                if of_person:
                    person_desc = of_person
                    destination = (
                        "person_recipient"  # sentinel, resolved in MappingHandler
                    )
                    if of_location and location in (None, "unknown"):
                        location = of_location
                # Normalize obj: "the question" → "question", "a quiz" → "quiz"
                if obj:
                    obj = obj.replace("the ", "").replace("a ", "").strip()

        source = self._find_prep_arg(root, token_set, self._ORIG_PREPS)
        self._update_context(context, obj, location, is_query)

        return {
            "action": action_str,
            "object": obj or "unknown",
            "object_category": obj_category,
            "source": source or "unknown",
            "destination": destination or "unknown",
            "location": location or "unknown",
            "actor": "robot",
            "person": person_desc.to_dict(),
        }

    # ──────────────────────────────────────────────────────────
    # Entity-aware helpers (NEW in v2)
    # ──────────────────────────────────────────────────────────

    def _find_object_entity(self, token_set: set, doc) -> tuple:
        """
        Scan doc.ents for an OBJ_INSTANCE or OBJ_CATEGORY whose tokens
        all fall within this clause's token_set.

        Returns (text, category_or_None) or (None, None).
        """
        for ent in doc.ents:
            if ent.label_ not in ("OBJ_INSTANCE", "OBJ_CATEGORY"):
                continue
            if all(t.i in token_set for t in ent):
                text = ent.text.lower()
                category = OBJECT_CATEGORY_MAP.get(text, None)
                if ent.label_ == "OBJ_CATEGORY":
                    category = text  # the category IS the entity
                return text, category
        return None, None

    def _find_person_entity(self, token_set: set, doc) -> PersonDescriptor:
        """
        Scan doc.ents for a PERSON_NAME or PERSON_DESC span in this clause.

        Converts the matched span into a structured PersonDescriptor.
        Falls back to an empty PersonDescriptor if nothing is found.
        """
        for ent in doc.ents:
            if not all(t.i in token_set for t in ent):
                continue

            if ent.label_ == "PERSON_NAME":
                return PersonDescriptor(name=ent.text)

            if ent.label_ == "PERSON_DESC":
                return self._parse_person_desc_span(ent)

        return PersonDescriptor()

    def _parse_person_desc_span(self, span) -> PersonDescriptor:
        """
        Convert a PERSON_DESC span into a PersonDescriptor.

        Handles the three pattern families produced by PersonDescTagger:
          • PERSON_DESC_POSE      → "the sitting person"
          • PERSON_DESC_CLOTHING  → "the person wearing a blue shirt"
          • PERSON_DESC_GESTURE   → "the person raising their right arm"
        """
        tokens = [
            t.lower_ for t in span if t.lower_ not in ("the", "person", "a", "an")
        ]

        # Named person (single PERSON_NAME token already handled above,
        # but guard in case it comes through PERSON_DESC)
        if not tokens:
            return PersonDescriptor()

        # Pose pattern: first token is a pose word
        if tokens[0] in PERSON_POSES:
            return PersonDescriptor(pose=tokens[0])

        # Clothing pattern: "wearing [color?] [garment]"
        if "wearing" in tokens:
            idx = tokens.index("wearing")
            remaining = tokens[idx + 1 :]
            color = next((t for t in remaining if t in CLOTHING_COLORS), None)
            garment = next((t for t in remaining if t in CLOTHING_ITEMS), None)
            # "t shirt" is two tokens — join consecutive non-color tokens
            if not garment:
                non_color = [t for t in remaining if t not in CLOTHING_COLORS]
                joined = " ".join(non_color)
                garment = joined if joined in CLOTHING_ITEMS else None
            return PersonDescriptor(clothing_color=color, clothing_item=garment)

        # Gesture pattern: "raising/pointing their [side] [body_part]"
        gesture_verbs = {"raising", "pointing"}
        gesture = next((t for t in tokens if t in gesture_verbs), None)
        if gesture:
            side = next((t for t in tokens if t in BODY_SIDES), None)
            part = next((t for t in tokens if t in BODY_PARTS), None)
            return PersonDescriptor(gesture=gesture, side=side, body_part=part)

        return PersonDescriptor()

    # ──────────────────────────────────────────────────────────
    # Dependency-tree helpers (used as fallback)
    # ──────────────────────────────────────────────────────────

    def _find_root_verb(self, tokens) -> Optional[object]:
        for dep in ("ROOT", "advcl", "ccomp", "relcl", "xcomp", "dep", "conj"):
            for t in tokens:
                if t.pos_ == "VERB" and t.dep_ == dep:
                    return t
        return next((t for t in tokens if t.pos_ == "VERB"), None)

    def _find_destination_token(self, verb, token_set: set):
        for child in verb.children:
            if child.dep_ == "prep" and child.lower_ in self._DEST_PREPS:
                for gc in child.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return gc
        return None

    def _extract_location_from_np(self, token, token_set: set) -> Optional[str]:
        for t in token.subtree:
            if t.dep_ == "prep" and t.lower_ in self._LOC_PREPS:
                for gc in t.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return self._noun_phrase(gc, token_set)
        return None

    def _resolve_recipient_for_transfer(
        self, verb, lemma: str, destination: Optional[str], token_set: set
    ) -> Optional[str]:
        if lemma not in self._TRANSFER_VERBS:
            return destination
        recipient = self._find_recipient_text(verb, token_set)
        if recipient and recipient != "user":
            return recipient
        return destination

    def _find_direct_object(self, verb, token_set: set, context: Dict) -> Optional[str]:
        for child in verb.children:
            if child.dep_ in ("dobj", "obj") and child.i in token_set:
                if child.lower_ in self._HUMAN_PRONOUNS:
                    continue
                np = self._noun_phrase(child, token_set)
                if np.lower() in self._ANAPHORS:
                    return context.get("last_object", np)
                return np
        return None

    def _find_recipient_text(self, verb, token_set: set) -> str:
        for child in verb.children:
            if child.dep_ in ("dative", "iobj") and child.i in token_set:
                return self._noun_phrase(child, token_set)
            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ in self._HUMAN_PRONOUNS
                and child.i in token_set
            ):
                return "user"
            if child.dep_ == "prep" and child.lower_ == "to":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        return self._noun_phrase(gc, token_set)
        return "user"

    def _is_query_verb(self, verb, lemma: str) -> bool:
        if lemma not in self._QUERY_VERBS:
            return False
        return any(t.tag_ in self._WH_TAGS for t in verb.subtree)

    def _extract_query_content(self, verb, token_set: set) -> str:
        subtree_tokens = sorted(
            [
                t
                for t in verb.subtree
                if not t.is_punct
                and t.lower_ not in self._HUMAN_PRONOUNS
                and t.i != verb.i
            ],
            key=lambda t: t.i,
        )
        return "_".join(t.text.lower() for t in subtree_tokens)

    def _find_prep_arg(self, verb, token_set: set, preps: frozenset) -> Optional[str]:
        for child in verb.children:
            if child.dep_ == "prep" and child.lower_ in preps:
                for gc in child.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return self._noun_phrase(gc, token_set)
        return None

    def _find_location_in_subtree(self, verb, token_set: set) -> Optional[str]:
        """
        Search the full subtree of a query verb for a location preposition.

        Used when the location is embedded in the subordinate clause, e.g.:
          "Tell me how many drinks there are ON THE BEDSIDE TABLE"
                                                 ↑ ccomp subtree
        We skip tokens that are part of a destination prep ("to") so we
        don't confuse recipients with locations.
        """
        for token in verb.subtree:
            if token.i not in token_set:
                continue
            if token.dep_ != "prep":
                continue
            if token.lower_ not in self._LOC_PREPS:
                continue
            # Skip if this prep is itself under a destination-prep ancestor
            if any(a.lower_ in self._DEST_PREPS for a in token.ancestors):
                continue
            for gc in token.children:
                if gc.dep_ in ("pobj", "obj"):
                    return self._noun_phrase(gc, token_set)
        return None

    def _noun_phrase(self, token, token_set: set) -> str:
        parts = sorted(
            [t for t in token.subtree if t.i in token_set and not t.is_punct],
            key=lambda t: t.i,
        )
        return " ".join(t.text.lower() for t in parts)

    def _noun_phrase_base(self, token, token_set: set) -> str:
        parts = []
        for t in sorted(token.subtree, key=lambda x: x.i):
            if t.i not in token_set or t.is_punct:
                continue
            if t.dep_ == "prep" and t.lower_ in self._LOC_PREPS:
                continue
            if any(
                a.dep_ == "prep" and a.lower_ in self._LOC_PREPS for a in t.ancestors
            ):
                continue
            parts.append(t.text.lower())
        return " ".join(parts)

    def _find_of_recipient(self, verb, token_set: set, doc) -> tuple:
        """
        Extract person and location from "of the person..." constructs.

        "Answer the question OF THE PERSON pointing to the right in the kitchen"
                                  ↑ prep "of" → pobj "person" → subtree has PERSON_DESC

        Returns (PersonDescriptor, location_str) or (None, None).
        """
        for child in verb.children:
            if child.dep_ not in ("dobj", "obj"):
                continue
            for sub in child.children:
                if sub.dep_ != "prep" or sub.lower_ != "of":
                    continue
                for pobj in sub.children:
                    if pobj.dep_ not in ("pobj", "obj"):
                        continue
                    # Entity-first: look for PERSON_DESC or PERSON_NAME
                    pobj_set = {t.i for t in pobj.subtree}
                    person = self._find_person_entity(pobj_set, doc)
                    if person.is_generic:
                        span_text = " ".join(
                            t.text.lower()
                            for t in sorted(pobj.subtree, key=lambda x: x.i)
                            if not t.is_punct
                        )
                        person = self._person_from_string_static(span_text)

                    # Extract embedded location ("in the kitchen")
                    location = None
                    for t in pobj.subtree:
                        if t.dep_ == "prep" and t.lower_ in self._LOC_PREPS:
                            for gc in t.children:
                                if gc.dep_ in ("pobj", "obj"):
                                    location = self._noun_phrase(gc, pobj_set)
                    return person, location
        return None, None

    @staticmethod
    def _person_from_string_static(text: str) -> PersonDescriptor:
        """
        Build a PersonDescriptor from raw text without importing MappingHandler.

        Used inside SpacySRL to avoid a circular dependency with handlers.py.
        Mirrors the logic in MappingHandler._person_from_string.
        """
        from semantic_parser.domain.entities import (
            PERSON_NAMES,
            PERSON_POSES,
            CLOTHING_COLORS,
            CLOTHING_ITEMS,
            BODY_SIDES,
            BODY_PARTS,
        )

        tokens = text.lower().split()

        for name in PERSON_NAMES:
            if name.lower() in tokens:
                return PersonDescriptor(name=name)

        for gesture in ("raising", "pointing"):
            if gesture in tokens:
                side = next((t for t in tokens if t in BODY_SIDES), None)
                part = next((t for t in tokens if t in BODY_PARTS), None)
                return PersonDescriptor(gesture=gesture, side=side, body_part=part)

        pose = next((t for t in tokens if t in PERSON_POSES), None)
        if pose:
            return PersonDescriptor(pose=pose)

        color = next((t for t in tokens if t in CLOTHING_COLORS), None)
        garment = next((t for t in tokens if t in CLOTHING_ITEMS), None)
        if color or garment:
            return PersonDescriptor(clothing_color=color, clothing_item=garment)

        return PersonDescriptor()
