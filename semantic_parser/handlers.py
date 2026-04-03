"""
Concrete pipeline handlers for the semantic parsing pipeline.

Each class handles exactly one step of the transformation:

    SRLHandler            text        → raw action dicts  (spaCy)
    MappingHandler        verb        → CD primitive       (verb_to_cd.json)
    QueryExpansionHandler QUERY       → [ATTEND, MTRANS]

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from semantic_parser.pipeline_handler import PipelineHandler
from semantic_parser.spacy_srl import SpacySRL


# ============================================================
# Handler 1 — Semantic Role Labeling
# ============================================================


class SRLHandler(PipelineHandler):
    """
    Pipeline step 1: natural language → raw action dicts.

    Responsibility: use SpacySRL to extract semantic roles
    from the input text and store the result in the pipeline data.

    Reads:  data["command"]
    Writes: data["raw_actions"]
    """

    def __init__(self, srl: SpacySRL) -> None:
        super().__init__()
        self._srl = srl

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        command = data.get("command", "")
        data["raw_actions"] = self._srl.parse(command)
        return self._pass_to_next(data)


# ============================================================
# Handler 2 — CD Primitive Mapping
# ============================================================


class MappingHandler(PipelineHandler):
    """
    Pipeline step 2: raw action dicts → CD primitives.

    Responsibility: map each verb lemma to a CD primitive using
    the verb_to_cd dictionary, then build the primitive dict
    with the correct fields for that primitive type.

    Reads:  data["raw_actions"]
    Writes: data["cd_actions"]
    """

    # Human-like pronouns that indicate a recipient, not a destination
    _HUMAN_DESTINATIONS: frozenset = frozenset({"me", "us", "you", "to me", "to us"})

    # Primitives whose main output field is "content" (not "object")
    _CONTENT_PRIMITIVES: frozenset = frozenset({"SPEAK", "MTRANS", "MBUILD"})

    def __init__(self, verb_map: Dict[str, str]) -> None:
        super().__init__()
        self._verb_map = verb_map

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw_actions = data.get("raw_actions", [])
        data["cd_actions"] = [self._map_single(action) for action in raw_actions]
        return self._pass_to_next(data)

    def _map_single(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Map one raw action dict to a CD primitive dict."""
        verb = action.get("action", "unknown").lower()
        object_name = action.get("object", "unknown")
        source = action.get("source", "unknown")
        destination = action.get("destination", "unknown")
        location = action.get("location", "unknown")
        actor = action.get("actor", "robot")

        destination = self._normalize_destination(destination)
        primitive = self._resolve_primitive(verb, destination)

        return self._build_primitive(
            primitive, actor, object_name, source, destination, location
        )

    def _normalize_destination(self, destination: str) -> str:
        """Replace human pronouns with the canonical 'user' token."""
        if destination.lower().strip() in self._HUMAN_DESTINATIONS:
            return "user"
        return destination

    def _resolve_primitive(self, verb: str, destination: str) -> str:
        """
        Look up the CD primitive for a verb.

        Special case: SPEAK with an explicit destination becomes MTRANS,
        because the robot is transferring information to someone.
        """
        primitive = self._verb_map.get(verb, "DO")
        if primitive == "SPEAK" and destination not in ("unknown", "user"):
            primitive = "MTRANS"
        return primitive

    def _build_primitive(
        self,
        primitive: str,
        actor: str,
        object_name: str,
        source: str,
        destination: str,
        location: str,
    ) -> Dict[str, Any]:
        """
        Build the CD primitive dict with the correct fields
        for each primitive type.

        Each primitive type has a specific schema:
          PTRANS/ATRANS/ACCOMPANY → actor, object, from, to
          GRASP/ATTEND/INGEST     → actor, object (+ location if known)
          SPEAK                   → actor, content
          MTRANS                  → actor, content, to
          MBUILD                  → actor, content
          WAIT                    → actor, condition
          SOCIAL                  → actor, target, location
          QUERY                   → actor, object, to
          DO                      → actor, action (fallback)
        """
        if primitive in {"PTRANS", "ATRANS"}:
            return {
                "primitive": primitive,
                "actor": actor,
                "object": object_name,
                "from": source,
                "to": destination,
            }
        if primitive == "ACCOMPANY":
            return {
                "primitive": primitive,
                "actor": actor,
                "object": object_name,
                "from": source,
                "to": destination,
            }
        if primitive in {"GRASP", "ATTEND", "INGEST"}:
            result = {"primitive": primitive, "actor": actor, "object": object_name}
            if location != "unknown":
                result["location"] = location
            return result
        if primitive == "SPEAK":
            return {"primitive": primitive, "actor": actor, "content": object_name}
        if primitive == "MTRANS":
            return {
                "primitive": primitive,
                "actor": actor,
                "content": object_name,
                "to": destination,
            }
        if primitive == "MBUILD":
            return {"primitive": primitive, "actor": actor, "content": object_name}
        if primitive == "WAIT":
            return {"primitive": primitive, "actor": actor, "condition": object_name}
        if primitive == "SOCIAL":
            return {
                "primitive": primitive,
                "actor": actor,
                "target": object_name,
                "location": location,
            }
        if primitive == "QUERY":
            return {
                "primitive": "QUERY",
                "actor": actor,
                "object": object_name,
                "to": destination,
            }
        # Fallback for unknown verbs
        return {"primitive": "DO", "actor": actor, "action": primitive}


# ============================================================
# Handler 3 — Query Expansion
# ============================================================


class QueryExpansionHandler(PipelineHandler):
    """
    Pipeline step 3: expand QUERY primitives into [ATTEND, MTRANS].

    Responsibility: detect QUERY primitives and replace each one
    with the appropriate sequence of perceptual actions.

    A QUERY is perceptual if its object starts with a WH-prefix
    (what_, how_many_, who_...) or contains a known perceptual noun
    (pose, name, color, age...).

    Non-perceptual QUERY → [SPEAK] (the robot just says something).

    Reads:  data["cd_actions"]
    Writes: data["cd_actions"]  (in-place expansion)
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

    _PERCEPTUAL_NOUNS: frozenset = frozenset(
        {
            "pose",
            "name",
            "gender",
            "age",
            "color",
            "colour",
            "size",
            "shape",
            "position",
            "location",
            "number",
            "amount",
            "quantity",
            "time",
            "date",
            "weight",
            "height",
        }
    )

    _SPATIAL_PREPS: tuple = (
        "_on_",
        "_in_",
        "_at_",
        "_near_",
        "_beside_",
        "_inside_",
        "_next_to_",
    )

    _STOP_WORDS: tuple = (
        "_wearing_",
        "_holding_",
        "_carrying_",
        "_with_",
        "_that_",
        "_who_",
        "_are_",
    )

    _VERB_STOPS: tuple = ("_are", "_is", "_was", "_were", "_have", "_has")

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cd_actions = data.get("cd_actions", [])
        data["cd_actions"] = self._expand(cd_actions)
        return self._pass_to_next(data)

    def _expand(self, cd_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand every QUERY in the list into its resolution."""
        expanded = []
        for cd in cd_actions:
            if cd.get("primitive") == "QUERY":
                expanded.extend(self._resolve_query(cd))
            else:
                expanded.append(cd)
        return expanded

    def _resolve_query(self, cd: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand a QUERY primitive into [ATTEND, MTRANS] or [SPEAK].

        If the object encodes a perceptual question, the robot must
        first attend (perceive) and then communicate the result.
        Otherwise, the robot simply speaks.
        """
        obj = cd.get("object", "unknown")
        actor = cd.get("actor", "robot")
        destination = cd.get("to", "user") or "user"

        if self._is_perceptual(obj):
            location = self._extract_location(obj)
            return [
                {
                    "primitive": "ATTEND",
                    "actor": actor,
                    "object": obj,
                    "location": location,
                },
                {
                    "primitive": "MTRANS",
                    "actor": actor,
                    "content": obj,
                    "to": destination,
                },
            ]

        return [{"primitive": "SPEAK", "actor": actor, "content": obj}]

    def _is_perceptual(self, obj: str) -> bool:
        """
        True if the object encodes a perceptual question.

        Two cases:
          1. Explicit WH-prefix:  "what_is_the_color_of..."
          2. Perceptual noun:     "the_pose_of_the_person..."
        """
        lower = obj.lower()

        if lower.startswith(self._QUERY_PREFIXES):
            return True

        # Strip leading articles before checking for perceptual nouns
        stripped = lower.lstrip("the_").lstrip("a_")
        first_word = stripped.split("_")[0] if stripped else ""
        return first_word in self._PERCEPTUAL_NOUNS

    def _extract_location(self, obj: str) -> str:
        """
        Extract the spatial location embedded in the snake_case object.

        Example:
            "how_many_people_in_the_bedroom_wearing_orange"
            → "the_bedroom"
        """
        lower = obj.lower()

        for prep in self._SPATIAL_PREPS:
            if prep not in lower:
                continue

            after = lower.split(prep, 1)[1]

            for stop in self._STOP_WORDS:
                if stop in after:
                    after = after.split(stop)[0]

            for verb_stop in self._VERB_STOPS:
                if after.endswith(verb_stop):
                    after = after[: -len(verb_stop)]
                elif (verb_stop + "_") in after:
                    after = after.split(verb_stop + "_")[0]

            return after

        return "unknown"


# ============================================================
# Verb map loader — standalone utility, not a handler
# ============================================================


def load_verb_map(logger=None) -> Dict[str, str]:
    """
    Load the verb-to-CD-primitive mapping from verb_to_cd.json.

    Search paths (in order):
      1. src/semantic_parser/semantic_parser/verb_to_cd.json
      2. src/Planning/semantic_parser/semantic_parser/verb_to_cd.json
      3. Same directory as this file

    Args:
        logger: Optional ROS2 logger for info/error messages.

    Returns:
        Dict mapping verb lemmas to CD primitive names.
        Returns empty dict if file is not found or invalid.
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
        Path(__file__).parent / file_name,
    ]

    for path in search_paths:
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
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
