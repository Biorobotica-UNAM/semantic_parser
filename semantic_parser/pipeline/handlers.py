"""
pipeline/handlers.py

Concrete pipeline handlers for the semantic parsing pipeline.

Chain
─────
    SRLHandler              text        → raw action dicts  (spaCy + EntityRuler)
    MappingHandler          verb        → structured CDPrimitive dataclass
    QueryExpansionHandler   QUERY       → [ATTEND, MTRANS]
    AttendNormalizationHandler          → canonical ATTEND objects
    SerializationHandler    dataclasses → plain dicts (JSON-ready for ROS2)

Each handler reads from and writes to the shared data dict that travels
through the chain.  No handler knows about its neighbours.

Design notes
────────────
• Single Responsibility: one class per transformation step.
• Open/Closed: add a new primitive type in primitives/schema.py and
  extend MappingHandler._build_primitive() — no other file changes.
• Dependency Inversion: MappingHandler depends on the verb_map dict
  abstraction, not on a concrete file reader.
• Strategy pattern inside MappingHandler: each primitive type is built
  by a dedicated _build_* method, making the dispatch table readable.

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from semantic_parser.pipeline.pipeline_handler import PipelineHandler
from semantic_parser.nlp.spacy_srl import SpacySRL
from semantic_parser.primitives.schema import (
    CDPrimitive,
    PersonDescriptor,
    PTRANSPrimitive,
    ATRANSPrimitive,
    GRASPPrimitive,
    ACCOMPANYPrimitive,
    ATTENDPrimitive,
    QUERYPrimitive,
    MTRANSPrimitive,
    SPEAKPrimitive,
    SOCIALPrimitive,
    WAITPrimitive,
)

# ──────────────────────────────────────────────────────────────────────────────
# Handler 1 — Semantic Role Labeling
# ──────────────────────────────────────────────────────────────────────────────


class SRLHandler(PipelineHandler):
    """
    Pipeline step 1: natural language → raw action dicts.

    Reads:  data["command"]
    Writes: data["raw_actions"]  (list of raw dicts from SpacySRL)
    """

    def __init__(self, srl: SpacySRL) -> None:
        super().__init__()
        self._srl = srl

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        command = data.get("command", "")
        data["raw_actions"] = self._srl.parse(command)
        return self._pass_to_next(data)


# ──────────────────────────────────────────────────────────────────────────────
# Handler 2 — CD Primitive Mapping
# ──────────────────────────────────────────────────────────────────────────────


class MappingHandler(PipelineHandler):
    """
    Pipeline step 2: raw action dicts → structured CDPrimitive dataclasses.

    Reads:  data["raw_actions"]
    Writes: data["cd_primitives"]  (list of CDPrimitive instances)

    Mapping rules
    ─────────────
    • verb_map[verb] → primitive type string.
    • Subtype is inferred from the verb itself (finer-grained than the type).
    • SPEAK with an explicit non-user destination → MTRANS.
    • "query" action → QUERY primitive (expanded later by QueryExpansionHandler).
    """

    _HUMAN_DESTINATIONS: frozenset = frozenset({"me", "us", "you", "to me", "to us"})

    # Maps verb → (primitive_type, subtype)
    _VERB_SUBTYPE_MAP: Dict[str, tuple] = {
        # PTRANS
        "go": ("PTRANS", "navigate"),
        "move": ("PTRANS", "navigate"),
        "navigate": ("PTRANS", "navigate"),
        "walk": ("PTRANS", "navigate"),
        "head": ("PTRANS", "navigate"),
        "proceed": ("PTRANS", "navigate"),
        "return": ("PTRANS", "navigate"),
        "exit": ("PTRANS", "navigate"),
        "enter": ("PTRANS", "navigate"),
        "open": ("PTRANS", "open"),
        "close": ("PTRANS", "close"),
        # ATRANS
        "bring": ("ATRANS", "bring"),
        "deliver": ("ATRANS", "bring"),
        "carry": ("ATRANS", "bring"),
        "give": ("ATRANS", "give"),
        "hand": ("ATRANS", "give"),
        "pass": ("ATRANS", "give"),
        "offer": ("ATRANS", "give"),
        "place": ("ATRANS", "place"),
        "put": ("ATRANS", "place"),
        "leave": ("ATRANS", "place"),
        "set": ("ATRANS", "place"),
        "drop": ("ATRANS", "place"),
        "deposit": ("ATRANS", "place"),
        "pour": ("ATRANS", "place"),
        "take": ("ATRANS", "bring"),
        # GRASP
        "grasp": ("GRASP", "pick_up"),
        "pick": ("GRASP", "pick_up"),
        "grab": ("GRASP", "pick_up"),
        "get": ("GRASP", "fetch"),
        "fetch": ("GRASP", "fetch"),
        "retrieve": ("GRASP", "fetch"),
        "collect": ("GRASP", "fetch"),
        # ACCOMPANY
        "follow": ("ACCOMPANY", "follow"),
        "escort": ("ACCOMPANY", "escort"),
        "guide": ("ACCOMPANY", "guide"),
        "lead": ("ACCOMPANY", "lead"),
        "accompany": ("ACCOMPANY", "escort"),
        # ATTEND
        "find": ("ATTEND", "find_object"),
        "look": ("ATTEND", "find_object"),
        "search": ("ATTEND", "find_object"),
        "locate": ("ATTEND", "find_object"),
        "identify": ("ATTEND", "identify"),
        "detect": ("ATTEND", "find_object"),
        "check": ("ATTEND", "identify"),
        "scan": ("ATTEND", "find_object"),
        "inspect": ("ATTEND", "identify"),
        # SPEAK
        "say": ("SPEAK", "affiliation"),
        "report": ("SPEAK", "affiliation"),
        "announce": ("SPEAK", "affiliation"),
        "answer": ("SPEAK", "answer_quiz"),
        "respond": ("SPEAK", "answer_quiz"),
        # SOCIAL
        "greet": ("SOCIAL", "greet"),
        "introduce": ("SOCIAL", "introduce"),
        "salute": ("SOCIAL", "salute"),
        "wave": ("SOCIAL", "greet"),
        "meet": ("SOCIAL", "meet"),
        # MBUILD
        "count": ("MBUILD", "count"),
        "remember": ("MBUILD", "remember"),
        "memorize": ("MBUILD", "remember"),
        "compare": ("MBUILD", "compare"),
        # WAIT
        "wait": ("WAIT", "unknown"),
        "stop": ("WAIT", "unknown"),
        # QUERY (internal, expanded later)
        "query": ("QUERY", "count"),
        # TELL is a query verb handled by spacy_srl; mapped to QUERY
        "tell": ("QUERY", "count"),
    }

    def __init__(self, verb_map: Dict[str, str]) -> None:
        super().__init__()
        self._verb_map = verb_map  # kept for unknown-verb fallback

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw_actions = data.get("raw_actions", [])
        data["cd_primitives"] = [self._map_single(a) for a in raw_actions]
        return self._pass_to_next(data)

    # ── Mapping ───────────────────────────────────────────────────────────────

    def _map_single(self, action: Dict[str, Any]) -> CDPrimitive:
        verb = action.get("action", "unknown").lower()
        obj = action.get("object", "unknown")
        obj_cat = action.get("object_category")
        source = action.get("source", "unknown")
        destination = self._normalize_destination(action.get("destination", "unknown"))
        location = action.get("location", "unknown")
        destination_location = action.get("destination_location")  # from SRL
        actor = action.get("actor", "robot")
        person_dict = action.get("person", {})

        person = self._build_person_descriptor(person_dict)

        destination, recipient_person, recipient_location = (
            self._resolve_person_destination(destination, person)
        )

        if recipient_location is None and destination_location:
            recipient_location = destination_location

        prim_type, subtype = self._resolve_type_and_subtype(
            verb, destination, obj, person
        )

        return self._build_primitive(
            prim_type,
            subtype,
            actor,
            obj,
            obj_cat,
            source,
            destination,
            location,
            person,
            recipient_person,
            recipient_location,
        )

    # ─────────────────────────────────────────────────────────────────────────

    _PERSON_DEST_PATTERNS = (
        # (trigger_word, side_words, body_parts, pose_words)
        # Order matters: check gesture before pose
        "raising",
        "pointing",
        "sitting",
        "standing",
        "lying",
        "waving",
    )

    def _resolve_person_destination(
        self,
        destination: str,
        person: "PersonDescriptor",
    ) -> tuple:
        """
        Detect destinations that describe a person rather than a place.

        e.g. "to the person raising their right arm in the office"

        Returns
        ───────
        (destination, recipient_person, recipient_location)

        • destination        — "user" if a person was found, else original string
        • recipient_person   — PersonDescriptor of the recipient, or None
        • recipient_location — where the recipient is ("the office"), or None
        """
        from semantic_parser.domain.entities import (
            PERSON_NAMES,
            PERSON_POSES,
            CLOTHING_COLORS,
            CLOTHING_ITEMS,
            BODY_SIDES,
            BODY_PARTS,
        )

        dest_lower = destination.lower().strip()

        if dest_lower in ("unknown", "user", "person_recipient", ""):
            return destination, None, None

        tokens = dest_lower.replace("to ", "", 1).split()

        # Named person: "to Robin in the kitchen"
        for name in PERSON_NAMES:
            if name.lower() in tokens:
                loc = self._extract_loc_from_tokens(tokens, None)
                return "user", PersonDescriptor(name=name), loc

        # Gesture: "to the person raising/pointing their [side] [body_part] in X"
        gesture_verbs = {"raising", "pointing"}
        gesture = next((t for t in tokens if t in gesture_verbs), None)
        if gesture:
            side = next((t for t in tokens if t in BODY_SIDES), None)
            part = next((t for t in tokens if t in BODY_PARTS), None)
            loc = self._extract_loc_from_tokens(tokens, None)
            return (
                "user",
                PersonDescriptor(gesture=gesture, side=side, body_part=part),
                loc,
            )

        # Pose: "to the sitting/standing/lying/waving person in X"
        pose = next((t for t in tokens if t in PERSON_POSES), None)
        if pose:
            loc = self._extract_loc_from_tokens(tokens, None)
            return "user", PersonDescriptor(pose=pose), loc

        # Clothing: "to the person wearing a blue shirt in X"
        color = next((t for t in tokens if t in CLOTHING_COLORS), None)
        garment = next((t for t in tokens if t in CLOTHING_ITEMS), None)
        if color or garment:
            loc = self._extract_loc_from_tokens(tokens, None)
            return (
                "user",
                PersonDescriptor(clothing_color=color, clothing_item=garment),
                loc,
            )

        return destination, None, None

    @staticmethod
    def _extract_loc_from_tokens(tokens: list, fallback) -> Optional[str]:
        """Extract location after 'in/at/on/near' in a token list."""
        _LOC_PREPS = {"in", "at", "on", "near"}
        for i, tok in enumerate(tokens):
            if tok in _LOC_PREPS and i + 1 < len(tokens):
                rest = tokens[i + 1 :]
                # Don't prepend 'the' if it's already the first word
                phrase = " ".join(rest)
                if not phrase.startswith("the "):
                    phrase = "the " + phrase
                return phrase
        return fallback

    def _normalize_destination(self, destination: str) -> str:
        if destination.lower().strip() in self._HUMAN_DESTINATIONS:
            return "user"
        # Sentinel set by SRL for answer-verb "of the person..." pattern —
        # passed through as-is; resolved later by _resolve_person_destination.
        if destination == "person_recipient":
            return "person_recipient"
        return destination

    def _resolve_type_and_subtype(
        self, verb: str, destination: str, obj: str, person: "PersonDescriptor"
    ) -> tuple:
        """
        Look up (primitive_type, subtype) for a verb.

        Special cases
        ─────────────
        • SPEAK with a non-user destination → MTRANS.
        • ATRANS with verb "take"/"move" AND a person descriptor → ACCOMPANY bring.
          "Take the standing person from X to Y" is an escort, not an object transfer.
        """
        type_sub = self._VERB_SUBTYPE_MAP.get(verb)
        if type_sub is None:
            prim = self._verb_map.get(verb, "DO")
            return (prim, "unknown")

        prim_type, subtype = type_sub

        _TAKE_VERBS = {"take", "move"}
        if prim_type == "ATRANS" and verb in _TAKE_VERBS:
            if not person.is_generic or self._obj_is_person(obj):
                prim_type = "ACCOMPANY"
                subtype = "escort"

        if prim_type == "SPEAK" and destination not in ("unknown", "user"):
            prim_type = "MTRANS"

        return prim_type, subtype

    @staticmethod
    def _obj_is_person(obj: str) -> bool:
        """
        Return True when the object string describes a person rather than a thing.
        Heuristic: contains the word "person" or a known pose/gesture word.
        """
        from semantic_parser.domain.entities import PERSON_POSES, PERSON_NAMES

        tokens = obj.lower().split()
        if "person" in tokens:
            return True
        if any(t in PERSON_POSES for t in tokens):
            return True
        if any(name.lower() in tokens for name in PERSON_NAMES):
            return True
        return False

    # ── Primitive builders (Strategy-like dispatch) ───────────────────────────

    def _build_primitive(
        self,
        prim_type: str,
        subtype: str,
        actor: str,
        obj: str,
        obj_cat: Optional[str],
        source: str,
        destination: str,
        location: str,
        person: PersonDescriptor,
        recipient_person: Optional[PersonDescriptor] = None,
    ) -> CDPrimitive:

        builders = {
            "PTRANS": self._build_ptrans,
            "ATRANS": self._build_atrans,
            "GRASP": self._build_grasp,
            "ACCOMPANY": self._build_accompany,
            "ATTEND": self._build_attend,
            "QUERY": self._build_query,
            "SPEAK": self._build_speak,
            "MTRANS": self._build_mtrans,
            "SOCIAL": self._build_social,
            "WAIT": self._build_wait,
        }
        builder = builders.get(prim_type)
        if builder:
            return builder(
                subtype,
                actor,
                obj,
                obj_cat,
                source,
                destination,
                location,
                person,
                recipient_person,
            )
        # Fallback
        return PTRANSPrimitive(actor=actor, destination=destination)

    def _build_primitive(
        self,
        prim_type: str,
        subtype: str,
        actor: str,
        obj: str,
        obj_cat: Optional[str],
        source: str,
        destination: str,
        location: str,
        person: PersonDescriptor,
        recipient_person: Optional[PersonDescriptor] = None,
        recipient_location: Optional[str] = None,
    ) -> CDPrimitive:

        builders = {
            "PTRANS": self._build_ptrans,
            "ATRANS": self._build_atrans,
            "GRASP": self._build_grasp,
            "ACCOMPANY": self._build_accompany,
            "ATTEND": self._build_attend,
            "QUERY": self._build_query,
            "SPEAK": self._build_speak,
            "MTRANS": self._build_mtrans,
            "SOCIAL": self._build_social,
            "WAIT": self._build_wait,
        }
        builder = builders.get(prim_type)
        if builder:
            return builder(
                subtype,
                actor,
                obj,
                obj_cat,
                source,
                destination,
                location,
                person,
                recipient_person,
                recipient_location,
            )
        return PTRANSPrimitive(actor=actor, destination=destination)

    def _build_ptrans(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        return PTRANSPrimitive(
            actor=actor, subtype=subtype, destination=dest, source=source
        )

    def _build_atrans(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        _PLACEMENT = {"place", "put", "set", "drop", "deposit", "pour", "leave"}
        if subtype in _PLACEMENT and dest == "unknown" and loc != "unknown":
            dest, loc = loc, "unknown"
        return ATRANSPrimitive(
            actor=actor,
            subtype=subtype,
            object=obj,
            object_category=obj_cat,
            source=source,
            destination="user" if rp else dest,
            recipient_person=rp or None,
            recipient_location=rl if rp else None,
        )

    def _build_grasp(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        return GRASPPrimitive(
            actor=actor,
            subtype=subtype,
            object=obj,
            object_category=obj_cat,
            location=loc if loc != "unknown" else source,
        )

    def _build_accompany(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        if not person.name and person.is_generic and obj not in ("unknown", ""):
            person = self._person_from_string(obj)
        return ACCOMPANYPrimitive(
            actor=actor,
            subtype=subtype,
            person=person,
            source=source,
            destination=dest,
        )

    def _build_attend(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        refined = subtype
        if person.name or person.pose or person.gesture or person.clothing_item:
            refined = "find_person"
        return ATTENDPrimitive(
            actor=actor,
            subtype=refined,
            object=obj,
            object_category=obj_cat,
            person=person if person.is_named or person.pose else None,
            location=loc,
        )

    def _build_query(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        return QUERYPrimitive(
            actor=actor,
            query_type=subtype,
            target=obj,
            target_category=obj_cat,
            location=loc,
            person_filter=person if person.pose or person.clothing_item else None,
            recipient=dest if dest != "unknown" else "user",
        )

    def _build_speak(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        if dest == "person_recipient":
            has_desc = any(
                [
                    person.pose,
                    person.gesture,
                    person.clothing_item,
                    person.clothing_color,
                    person.name,
                ]
            )
            return SPEAKPrimitive(
                actor=actor,
                subtype=subtype,
                content=obj,
                location=loc,
                recipient="user",
                recipient_person=person if has_desc else None,
                recipient_location=rl,
            )
        if rp:
            return SPEAKPrimitive(
                actor=actor,
                subtype=subtype,
                content=obj,
                location=loc,
                recipient="user",
                recipient_person=rp,
                recipient_location=rl,
            )
        return SPEAKPrimitive(
            actor=actor,
            subtype=subtype,
            content=obj,
            location=loc,
            recipient=dest if dest not in ("unknown", "") else "user",
        )

    def _build_mtrans(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        if rp:
            return MTRANSPrimitive(
                actor=actor,
                content=obj,
                location="unknown",
                recipient="user",
                recipient_person=rp,
                recipient_location=rl,
            )
        # Resolve sentinel left by answer-verb path in SRL
        if dest == "person_recipient":
            has_desc = any(
                [
                    person.pose,
                    person.gesture,
                    person.clothing_item,
                    person.clothing_color,
                    person.name,
                ]
            )
            return MTRANSPrimitive(
                actor=actor,
                content=obj,
                location="unknown",
                recipient="user",
                recipient_person=person if has_desc else None,
                recipient_location=rl,
            )
        return MTRANSPrimitive(
            actor=actor,
            content=obj,
            location="unknown",
            recipient=(
                dest if dest not in ("unknown", "person_recipient", "") else "user"
            ),
        )

    def _build_social(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        if not person.name and person.is_generic and obj not in ("unknown", ""):
            person = self._person_from_string(obj)
        return SOCIALPrimitive(
            actor=actor, subtype=subtype, person=person, location=loc
        )

    def _build_wait(
        self, subtype, actor, obj, obj_cat, source, dest, loc, person, rp=None, rl=None
    ):
        return WAITPrimitive(actor=actor, condition=obj)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _person_from_string(text: str) -> PersonDescriptor:
        """
        Last-resort parser: build a PersonDescriptor from a raw object string
        like "the standing person" or "the person wearing a blue jacket".

        Called when the EntityRuler/Matcher didn't produce a PERSON_DESC span
        (e.g. the token was absorbed by another entity before the Matcher ran).
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

        # Named person
        for name in PERSON_NAMES:
            if name.lower() in tokens:
                return PersonDescriptor(name=name)

        # Gesture: "raising/pointing their [side] [body_part]"
        for gesture in ("raising", "pointing"):
            if gesture in tokens:
                side = next((t for t in tokens if t in BODY_SIDES), None)
                part = next((t for t in tokens if t in BODY_PARTS), None)
                return PersonDescriptor(gesture=gesture, side=side, body_part=part)

        # Pose: "sitting", "standing", "lying", "waving"
        pose = next((t for t in tokens if t in PERSON_POSES), None)
        if pose:
            return PersonDescriptor(pose=pose)

        # Clothing: "wearing a [color] [garment]"
        color = next((t for t in tokens if t in CLOTHING_COLORS), None)
        garment = next((t for t in tokens if t in CLOTHING_ITEMS), None)
        if color or garment:
            return PersonDescriptor(clothing_color=color, clothing_item=garment)

        # Nothing matched — keep it as a name string
        return PersonDescriptor(name=text)

    @staticmethod
    def _build_person_descriptor(person_dict: Dict) -> PersonDescriptor:
        if not person_dict:
            return PersonDescriptor()
        return PersonDescriptor(
            name=person_dict.get("name"),
            pose=person_dict.get("pose"),
            gesture=person_dict.get("gesture"),
            side=person_dict.get("side"),
            body_part=person_dict.get("body_part"),
            clothing_color=person_dict.get("clothing_color"),
            clothing_item=person_dict.get("clothing_item"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Handler 3 — Query Expansion
# ──────────────────────────────────────────────────────────────────────────────


class QueryExpansionHandler(PipelineHandler):
    """
    Pipeline step 3: expand QUERY primitives into [ATTEND, MTRANS].

    A QUERY carries a snake_case content string produced by SpacySRL.
    This handler:
      1. Classifies the query type (count, property, attribute, identity).
      2. Builds a structured ATTENDPrimitive + MTRANSPrimitive pair.
      3. Non-perceptual QUERYs become a single SPEAKPrimitive.

    Reads:  data["cd_primitives"]
    Writes: data["cd_primitives"]  (QUERY replaced by [ATTEND, MTRANS] or [SPEAK])
    """

    _QUERY_PREFIXES: tuple = (
        "how_many_",
        "how_much_",
        "what_",
        "where_",
        "which_",
        "who_",
        "whether_",
        "if_",
    )

    _COUNT_PREFIXES: tuple = ("how_many_", "how_much_")

    _PROPERTY_KEYWORDS = frozenset(
        {
            "largest",
            "biggest",
            "smallest",
            "tiniest",
            "heaviest",
            "lightest",
            "tallest",
            "shortest",
            "oldest",
            "youngest",
            "thinnest",
            "thickest",
            "widest",
            "narrowest",
            "fullest",
            "emptiest",
        }
    )

    _ATTRIBUTE_KEYWORDS = frozenset(
        {
            "pose",
            "name",
            "gesture",
            "color",
            "colour",
            "age",
            "gender",
            "size",
            "weight",
            "height",
        }
    )

    _SELF_INFO_KEYWORDS = frozenset(
        {
            "affiliation",
            "name",
            "country",
            "team",
            "time",
            "date",
            "today",
            "tomorrow",
            "week",
            "day",
            "month",
        }
    )

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        primitives = data.get("cd_primitives", [])
        data["cd_primitives"] = self._expand_all(primitives)
        return self._pass_to_next(data)

    def _expand_all(self, primitives: List[CDPrimitive]) -> List[CDPrimitive]:
        result = []
        for prim in primitives:
            if isinstance(prim, QUERYPrimitive):
                result.extend(self._expand_query(prim))
            else:
                result.append(prim)
        return result

    def _expand_query(self, query: QUERYPrimitive) -> List[CDPrimitive]:
        content = query.target.lower()
        tokens = content.split("_")

        query_type = self._classify(content, tokens)

        if query_type == "self_info":
            return [
                SPEAKPrimitive(
                    actor=query.actor,
                    subtype="affiliation",
                    content=content,
                    recipient=query.recipient,
                    location=query.location,
                )
            ]

        # Perceptual query → ATTEND + MTRANS
        qualifier = self._extract_qualifier(tokens)
        attribute = self._extract_attribute(tokens)
        target, person_filter = self._extract_target_and_filter(tokens)

        attend = ATTENDPrimitive(
            actor=query.actor,
            subtype=self._attend_subtype(query_type),
            object=content,
            object_category=query.target_category,
            person=person_filter or query.person_filter,
            location=query.location,
            query_attribute=attribute or qualifier,
            query_filter=self._build_filter_string(person_filter),
        )

        mtrans = MTRANSPrimitive(
            actor=query.actor,
            content=content,
            recipient=(
                "user" if query.recipient == "person_recipient" else query.recipient
            ),
            location="unknown",  # robot reports to user wherever they are
        )

        return [attend, mtrans]

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, content: str, tokens: List[str]) -> str:
        if any(content.startswith(p) for p in self._COUNT_PREFIXES):
            return "count"
        if any(kw in tokens for kw in self._PROPERTY_KEYWORDS):
            return "property"
        if any(kw in tokens for kw in self._ATTRIBUTE_KEYWORDS):
            return "attribute"
        if any(kw in tokens for kw in self._SELF_INFO_KEYWORDS):
            return "self_info"
        return "identity"

    @staticmethod
    def _attend_subtype(query_type: str) -> str:
        return {
            "count": "count_objects",
            "property": "find_object",
            "attribute": "identify",
            "identity": "find_person",
        }.get(query_type, "find_object")

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_qualifier(self, tokens: List[str]) -> Optional[str]:
        from semantic_parser.domain.entities import SUPERLATIVE_MAP

        for tok in tokens:
            canonical = SUPERLATIVE_MAP.get(tok)
            if canonical:
                return canonical
        return None

    def _extract_attribute(self, tokens: List[str]) -> Optional[str]:
        for tok in tokens:
            if tok in self._ATTRIBUTE_KEYWORDS:
                return tok
        return None

    def _extract_target_and_filter(self, tokens: List[str]) -> tuple:
        """
        Split the token list into (noun_head, PersonDescriptor_or_None).

        Handles patterns like:
          "how_many_people_in_bedroom_wearing_yellow_shirt"
            → target="people", PersonDescriptor(clothing_color="yellow", clothing_item="shirt")
          "how_many_persons_raising_right_arm_in_office"
            → target="persons", PersonDescriptor(gesture="raising", side="right", body_part="arm")
        """
        from semantic_parser.domain.entities import (
            CLOTHING_COLORS,
            CLOTHING_ITEMS,
            BODY_SIDES,
            BODY_PARTS,
            PERSON_POSES,
        )

        # Skip noise and find first real noun
        noise = {
            "how",
            "many",
            "much",
            "what",
            "which",
            "who",
            "where",
            "is",
            "the",
            "a",
            "an",
            "of",
            "there",
            "to",
            "it",
            "its",
            "are",
        }
        spatial = {
            "in",
            "on",
            "at",
            "near",
            "beside",
            "inside",
            "next",
            "above",
            "below",
            "behind",
            "between",
        }

        noun: Optional[str] = None
        noun_idx = -1
        for i, tok in enumerate(tokens):
            if tok in noise:
                continue
            from semantic_parser.domain.entities import SUPERLATIVE_MAP

            if tok in SUPERLATIVE_MAP:
                continue
            if tok in self._ATTRIBUTE_KEYWORDS:
                continue
            noun = tok
            noun_idx = i
            break

        if noun is None:
            return "unknown", None

        # Scan remaining tokens for person descriptors
        remaining = tokens[noun_idx + 1 :]
        # Skip spatial prepositions and location words
        after_loc = False
        descriptor_tokens = []
        for tok in remaining:
            if tok in spatial:
                after_loc = True
                continue
            if after_loc:
                # Stop collecting location; look for "wearing" or pose cues
                if tok in ("wearing", "raising", "pointing") or tok in PERSON_POSES:
                    after_loc = False
                    descriptor_tokens.append(tok)
                else:
                    descriptor_tokens.append(tok)
            else:
                descriptor_tokens.append(tok)

        if not descriptor_tokens:
            return noun, None

        person = PersonDescriptor()
        colors = [t for t in descriptor_tokens if t in CLOTHING_COLORS]
        garments = [t for t in descriptor_tokens if t in CLOTHING_ITEMS]
        gestures = [t for t in descriptor_tokens if t in ("raising", "pointing")]
        sides = [t for t in descriptor_tokens if t in BODY_SIDES]
        parts = [t for t in descriptor_tokens if t in BODY_PARTS]
        poses = [t for t in descriptor_tokens if t in PERSON_POSES]

        if colors or garments:
            person.clothing_color = colors[0] if colors else None
            person.clothing_item = garments[0] if garments else None
        if gestures:
            person.gesture = gestures[0]
            person.side = sides[0] if sides else None
            person.body_part = parts[0] if parts else None
        if poses:
            person.pose = poses[0]

        has_descriptor = any(
            [
                person.clothing_color,
                person.clothing_item,
                person.gesture,
                person.pose,
            ]
        )
        return noun, person if has_descriptor else None

    @staticmethod
    def _build_filter_string(person: Optional[PersonDescriptor]) -> Optional[str]:
        if person is None:
            return None
        parts = []
        if person.clothing_color:
            parts.append(person.clothing_color)
        if person.clothing_item:
            parts.append(person.clothing_item)
        if person.pose:
            parts.append(person.pose)
        if person.gesture:
            parts.append(person.gesture)
        if person.side:
            parts.append(person.side)
        if person.body_part:
            parts.append(person.body_part)
        return "_".join(parts) if parts else None


# ──────────────────────────────────────────────────────────────────────────────
# Handler 4 — ATTEND Object Normalization
# ──────────────────────────────────────────────────────────────────────────────


class AttendNormalizationHandler(PipelineHandler):
    """
    Pipeline step 4: normalize the `object` field of ATTEND primitives.

    Converts the verbose snake_case string produced by QueryExpansionHandler
    into a compact canonical descriptor for the CLIPS planner.

    Canonical form examples
    ────────────────────────
      "how_many_cups_on_the_table"                       → count_cups
      "how_many_people_in_kitchen_wearing_yellow_jacket" → count_people_yellow_jacket
      "what_is_the_largest_food_on_the_shelf"            → biggest_food
      "what_is_the_pose_of_the_person"                   → pose_person

    Reads:  data["cd_primitives"]
    Writes: data["cd_primitives"]  (normalized in-place copies)
    """

    from semantic_parser.domain.entities import SUPERLATIVE_MAP as _SMAP

    _NOISE: frozenset = frozenset(
        {
            "what",
            "which",
            "who",
            "where",
            "how",
            "many",
            "much",
            "is",
            "was",
            "the",
            "a",
            "an",
            "of",
            "there",
            "to",
        }
    )
    _SPATIAL: frozenset = frozenset(
        {
            "in",
            "on",
            "at",
            "near",
            "beside",
            "inside",
            "next",
            "above",
            "below",
            "behind",
            "between",
            "over",
            "under",
        }
    )
    _ATTRIBUTE: frozenset = frozenset(
        {
            "pose",
            "name",
            "gesture",
            "color",
            "colour",
            "age",
            "gender",
            "size",
            "weight",
            "height",
        }
    )
    _DESCRIPTOR_JOINS: frozenset = frozenset(
        {
            "wearing",
            "holding",
            "carrying",
            "with",
            "that",
            "who",
            "raising",
            "sitting",
            "standing",
            "lying",
            "pointing",
            "waving",
        }
    )
    _COPULAS: frozenset = frozenset(
        {
            "are",
            "were",
            "their",
            "its",
            "his",
            "her",
        }
    )

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        primitives = data.get("cd_primitives", [])
        data["cd_primitives"] = self._normalize_all(primitives)
        return self._pass_to_next(data)

    def _normalize_all(self, primitives: List[CDPrimitive]) -> List[CDPrimitive]:
        result = []
        i = 0
        while i < len(primitives):
            prim = primitives[i]
            if isinstance(prim, ATTENDPrimitive):
                raw = prim.object
                normalized = self._normalize(raw)
                attend = ATTENDPrimitive(**{**prim.__dict__, "object": normalized})
                result.append(attend)
                i += 1
                # Sync the immediately-following MTRANS
                if i < len(primitives) and isinstance(primitives[i], MTRANSPrimitive):
                    nxt = primitives[i]
                    if nxt.content == raw:
                        result.append(
                            MTRANSPrimitive(**{**nxt.__dict__, "content": normalized})
                        )
                        i += 1
                        continue
            else:
                result.append(prim)
                i += 1
        return result

    def _normalize(self, obj: str) -> str:
        tokens = obj.lower().split("_")
        from semantic_parser.domain.entities import SUPERLATIVE_MAP

        # Rule 1 — Count query
        if obj.lower().startswith(("how_many_", "how_much_")):
            noun = self._extract_noun_with_desc(tokens)
            return f"count_{noun}" if noun else obj

        # Rule 2 — Superlative property query
        qualifier = next(
            (SUPERLATIVE_MAP[t] for t in tokens if t in SUPERLATIVE_MAP), None
        )
        if qualifier:
            noun = self._extract_noun_with_desc(tokens, skip_sup=True)
            return f"{qualifier}_{noun}" if noun else obj

        # Rule 3 — Perceptual attribute
        attr = next((t for t in tokens if t in self._ATTRIBUTE), None)
        if attr:
            noun = self._noun_after(tokens, attr)
            return f"{attr}_{noun}" if noun else attr

        # Rule 4 — Identity / who
        if tokens and tokens[0] == "who":
            return "person"

        noun = self._extract_noun_with_desc(tokens, skip_sup=False)
        return noun if noun else obj

    def _extract_noun_with_desc(
        self, tokens: List[str], skip_sup: bool = False
    ) -> Optional[str]:
        from semantic_parser.domain.entities import SUPERLATIVE_MAP

        pre, post = [], []
        state = "pre"
        for tok in tokens:
            if state == "pre":
                if tok in self._SPATIAL:
                    state = "loc"
                else:
                    pre.append(tok)
            elif state == "loc":
                if tok in self._DESCRIPTOR_JOINS or tok in self._COPULAS:
                    state = "post"
            else:
                post.append(tok)

        noun = None
        pre_desc = []
        after = False
        for tok in pre:
            if not after:
                if tok in self._NOISE:
                    continue
                if tok in self._COPULAS:
                    continue
                if skip_sup and tok in SUPERLATIVE_MAP:
                    continue
                noun = tok
                after = True
            else:
                if tok in self._DESCRIPTOR_JOINS | self._COPULAS | self._NOISE:
                    continue
                pre_desc.append(tok)

        if noun is None:
            return None

        post_desc = [
            t
            for t in post
            if t not in self._NOISE | self._COPULAS | self._DESCRIPTOR_JOINS
        ]

        desc = pre_desc + post_desc
        return f"{noun}_{'_'.join(desc)}" if desc else noun

    def _noun_after(self, tokens: List[str], attr: str) -> Optional[str]:
        after = False
        for tok in tokens:
            if tok == attr:
                after = True
                continue
            if not after:
                continue
            if tok in self._NOISE | self._SPATIAL:
                continue
            return tok
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Handler 5 — Serialization (NEW in v2)
# ──────────────────────────────────────────────────────────────────────────────


class SerializationHandler(PipelineHandler):
    """
    Pipeline step 5: CDPrimitive dataclasses → plain dicts.

    Responsibility: convert the list of dataclass instances into
    JSON-serializable plain dicts for the ROS2 response.

    Keeping serialization as a dedicated handler means all upstream
    handlers work with typed objects, while only the final step
    produces raw dicts.  This respects SRP and makes unit-testing
    upstream handlers trivial.

    Reads:  data["cd_primitives"]
    Writes: data["cd_actions"]   (list of plain dicts, JSON-ready)
    """

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        primitives = data.get("cd_primitives", [])
        data["cd_actions"] = [p.to_dict() for p in primitives]
        return self._pass_to_next(data)


# ──────────────────────────────────────────────────────────────────────────────
# Verb map loader — standalone utility, not a handler
# ──────────────────────────────────────────────────────────────────────────────


def load_verb_map(logger=None) -> Dict[str, str]:
    """
    Load the verb-to-CD-primitive mapping from verb_to_cd.json.

    Search paths (in order):
      1. src/semantic_parser/semantic_parser/verb_to_cd.json
      2. src/Planning/semantic_parser/semantic_parser/verb_to_cd.json
      3. Same directory as this file's parent (semantic_parser package root)
    """
    file_name = "verb_to_cd.json"
    search_paths = [
        Path(os.getcwd()) / "src" / "semantic_parser" / "semantic_parser" / file_name,
        Path(os.getcwd())
        / "src"
        / "Planning"
        / "semantic_parser"
        / "semantic_parser"
        / file_name,
        Path(__file__).parent.parent / file_name,
    ]
    for path in search_paths:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if logger:
                logger.info(f"verb_to_cd loaded from {path} ({len(data)} verbs).")
            return data
        except json.JSONDecodeError as e:
            if logger:
                logger.error(f"Invalid JSON in {path}: {e}")
            return {}
    if logger:
        logger.error("verb_to_cd.json not found in any search path.")
    return {}
