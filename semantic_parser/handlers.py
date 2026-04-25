"""
Concrete pipeline handlers for the semantic parsing pipeline.

Each class handles exactly one step of the transformation:

    SRLHandler                  text        → raw action dicts  (spaCy)
    MappingHandler              verb        → CD primitive       (verb_to_cd.json)
    QueryExpansionHandler       QUERY       → [ATTEND, MTRANS]
    AttendNormalizationHandler  ATTEND obj  → canonical form     (e.g. biggest_food)

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
                "location": location,
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
                "location": location,
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
            "question",
            "answer",
            "status",
            "description",
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
        True si el objeto codifica una pregunta perceptual.

        Tres casos (en orden de prioridad):
          1. Prefijo WH al inicio:    "what_is_the_color_of..."
          2. Prefijo WH en cualquier segmento de la cadena, para manejar
             residuos de cláusulas previas que no se hayan filtrado en SRL.
          3. Sustantivo perceptual como primera palabra significativa:
             "the_pose_of_the_person..."
        """
        lower = obj.lower()

        # Caso 1: prefijo WH al inicio (caso más común tras el fix de SRL)
        if any(lower.startswith(p) for p in self._QUERY_PREFIXES):
            return True

        # Caso 2: prefijo WH en cualquier segmento (defensa en profundidad)
        segments = lower.split("_")
        for i, _ in enumerate(segments):
            fragment = "_".join(segments[i:])
            if any(fragment.startswith(p) for p in self._QUERY_PREFIXES):
                return True

        # Caso 3: sustantivo perceptual como primera palabra significativa
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
# Handler 4 — ATTEND Object Normalization
# ============================================================


class AttendNormalizationHandler(PipelineHandler):
    """
    Pipeline step 4: normalize the 'object' field of ATTEND primitives.

    Responsibility: convert verbose snake_case query strings produced by
    QueryExpansionHandler into compact, canonical object descriptors that
    downstream planners can dispatch on directly.

    The normalization resolves four cases in priority order:

      1. Quantifier query  → "count_{noun}[_{descriptor}]"
            "how_many_cups_on_the_table"                        → "count_cups"
            "how_many_people_in_kitchen_wearing_yellow_jacket"  → "count_people_yellow_jacket"
            "how_many_persons_raising_right_arm_in_bedroom"     → "count_persons_right_arm"

      2. Superlative query → "{qualifier}_{noun}"
            "what_is_the_largest_food_on_the_shelf"             → "biggest_food"
            "which_is_the_youngest_person_in_room"              → "youngest_person"

      3. Attribute query   → "{attribute}_{noun}"
            "what_is_the_color_of_the_cup"                      → "color_cup"
            "what_is_the_name_of_the_person"                    → "name_person"

      4. Plain identity    → "{noun}"
            "who_is_in_the_living_room"                         → "person"

    If no rule matches, the object is returned unchanged (fail-safe).

    The companion MTRANS 'content' field is updated in sync with its
    preceding ATTEND so that both primitives always carry the same value.

    Reads:  data["cd_actions"]
    Writes: data["cd_actions"]  (normalized copies, originals not mutated)
    """

    # ── Quantifier prefixes ───────────────────────────────────────────────────
    _QUANTIFIER_PREFIXES: Dict[str, str] = {
        "how_many_": "count",
        "how_much_": "count",
    }

    # ── Superlative adjective → canonical qualifier ───────────────────────────
    _SUPERLATIVE_MAP: Dict[str, str] = {
        # size
        "largest": "biggest",
        "biggest": "biggest",
        "greatest": "biggest",
        "smallest": "smallest",
        "tiniest": "smallest",
        "littlest": "smallest",
        # height / length
        "tallest": "tallest",
        "highest": "tallest",
        "shortest": "shortest",
        "lowest": "shortest",
        # weight
        "heaviest": "heaviest",
        "lightest": "lightest",
        # age
        "oldest": "oldest",
        "eldest": "oldest",
        "youngest": "youngest",
        # other physical properties
        "widest": "widest",
        "narrowest": "narrowest",
        "thickest": "thickest",
        "thinnest": "thinnest",
        "fullest": "fullest",
        "emptiest": "emptiest",
    }

    # ── Perceptual-attribute words ────────────────────────────────────────────
    _ATTRIBUTE_WORDS: frozenset = frozenset(
        {
            "color",
            "colour",
            "name",
            "age",
            "gender",
            "size",
            "shape",
            "weight",
            "height",
            "pose",
            "position",
            "location",
            "status",
            "description",
        }
    )

    # ── Noise words — always discarded, never part of noun or descriptor ──────
    # Note: spatial preps (in, on, at…) are intentionally NOT in this set so
    # the state machine in _extract_noun_with_descriptors can detect them first.
    _NOISE_WORDS: frozenset = frozenset(
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
            "it",
            "its",
        }
    )

    # ── Spatial prepositions — trigger transition to location state ───────────
    _SPATIAL_PREPS: frozenset = frozenset(
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

    # ── Connector/participial words — consumed silently after noun head ───────
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
            "looking",
            "facing",
        }
    )

    # ── Copulas and possessives — discarded wherever they appear ─────────────
    # Includes common typos: "ar" → are, "thir" → their.
    _COPULAS: frozenset = frozenset(
        {
            "are",
            "were",
            "ar",  # typo: are
            "their",
            "thir",  # possessive + typo
            "its",
            "his",
            "her",
        }
    )

    # ── "who" queries resolve to this canonical noun ──────────────────────────
    _WHO_NOUN = "person"

    # ─────────────────────────────────────────────────────────────────────────

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cd_actions = data.get("cd_actions", [])
        data["cd_actions"] = self._normalize_all(cd_actions)
        return self._pass_to_next(data)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _normalize_all(self, cd_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Walk the action list and normalize every ATTEND object.
        Update the immediately-following MTRANS content to match.

        Uses an index-based loop so we can consume the paired MTRANS
        in the same iteration as its ATTEND, avoiding the stale-iterator
        bug that occurs when reassigning the list mid-loop.
        """
        result = []
        i = 0
        while i < len(cd_actions):
            cd = cd_actions[i]

            if cd.get("primitive") != "ATTEND":
                result.append(cd)
                i += 1
                continue

            raw_obj = cd.get("object", "unknown")
            normalized = self._normalize_object(raw_obj)

            attend = dict(cd)
            attend["object"] = normalized
            result.append(attend)
            i += 1

            # Consume and update the immediately-following MTRANS in the
            # same pass, so it is never appended a second time.
            if i < len(cd_actions):
                nxt = cd_actions[i]
                if nxt.get("primitive") == "MTRANS" and nxt.get("content") == raw_obj:
                    mtrans = dict(nxt)
                    mtrans["content"] = normalized
                    result.append(mtrans)
                    i += 1  # skip: already emitted

        return result

    # ── Normalization rules ───────────────────────────────────────────────────

    def _normalize_object(self, obj: str) -> str:
        """
        Apply normalization rules in priority order.

        Returns a compact canonical string or the original obj as fallback.
        """
        tokens = obj.lower().split("_")

        # Rule 1 — Quantifier  ("how many / how much")
        quantifier = self._detect_quantifier(obj)
        if quantifier:
            noun = self._extract_noun_with_descriptors(tokens, skip_superlative=False)
            if noun:
                return f"{quantifier}_{noun}"

        # Rule 2 — Superlative  ("largest", "oldest", ...)
        qualifier, _ = self._detect_superlative(tokens)
        if qualifier:
            noun = self._extract_noun_with_descriptors(tokens, skip_superlative=True)
            if noun:
                return f"{qualifier}_{noun}"

        # Rule 3 — Perceptual attribute  ("color", "name", "age", ...)
        attribute = self._detect_attribute(tokens)
        if attribute:
            noun = self._extract_noun_after_attribute(tokens, attribute)
            if noun:
                return f"{attribute}_{noun}"
            return attribute

        # Rule 4 — Plain identity query ("who is ...", "what is ...")
        if tokens and tokens[0] == "who":
            return self._WHO_NOUN

        noun = self._extract_noun(tokens, skip_superlative=False)
        if noun:
            return noun

        # Rule 5 — Fail-safe: return unchanged
        return obj

    # ── Detectors ─────────────────────────────────────────────────────────────

    def _detect_quantifier(self, obj: str) -> Optional[str]:
        """Return the canonical quantifier token if the object is a count query."""
        for prefix, token in self._QUANTIFIER_PREFIXES.items():
            if obj.lower().startswith(prefix):
                return token
        return None

    def _detect_superlative(self, tokens: List[str]):
        """Scan tokens for a superlative. Returns (canonical, index) or (None, -1)."""
        for i, tok in enumerate(tokens):
            canonical = self._SUPERLATIVE_MAP.get(tok)
            if canonical:
                return canonical, i
        return None, -1

    def _detect_attribute(self, tokens: List[str]) -> Optional[str]:
        """Return the first perceptual-attribute word found in the token list."""
        for tok in tokens:
            if tok in self._ATTRIBUTE_WORDS:
                return tok
        return None

    # ── Noun extraction ───────────────────────────────────────────────────────

    def _extract_noun_with_descriptors(
        self, tokens: List[str], skip_superlative: bool
    ) -> Optional[str]:
        """
        Extract the noun head plus any descriptor tokens.

        Handles descriptors that appear either BEFORE or AFTER the location:

          Before:  "how_many_persons_raising_right_arm_in_bedroom"
                    pre_loc=[how,many,persons,raising,right,arm]  loc_buf=[bedroom]
                    → noun=persons  pre_desc=[right,arm]  → "persons_right_arm"

          After:   "how_many_people_in_kitchen_wearing_yellow_jacket"
                    pre_loc=[how,many,people]  loc_buf=[kitchen]  post_loc=[yellow,jacket]
                    → noun=people  post_desc=[yellow,jacket]  → "people_yellow_jacket"

          None:    "how_many_cups_on_the_table"
                    pre_loc=[how,many,cups]  loc_buf=[the,table]
                    → noun=cups  → "cups"

        State machine for splitting:
          "pre"  — tokens before the first spatial preposition
          "loc"  — tokens inside the location phrase (discarded)
          "post" — tokens after a connector/copula that follows the location
        """
        # ── Phase 1: split tokens into three segments ─────────────────────────
        pre_loc: List[str] = []
        loc_buf: List[str] = []  # discarded
        post_loc: List[str] = []
        state = "pre"

        for tok in tokens:
            if state == "pre":
                if tok in self._SPATIAL_PREPS:
                    state = "loc"
                else:
                    pre_loc.append(tok)
            elif state == "loc":
                # A connector/copula closes the location phrase.
                if tok in self._DESCRIPTOR_JOINS or tok in self._COPULAS:
                    state = "post"
                    # connector itself is not collected
                else:
                    loc_buf.append(tok)
            else:  # "post"
                post_loc.append(tok)

        # ── Phase 2: extract noun + pre-location descriptors from pre_loc ──────
        # pre_loc structure: [noise...] [superlative?] noun [connector?] [descriptor...]
        noun: Optional[str] = None
        pre_desc: List[str] = []
        after_noun = False

        for tok in pre_loc:
            if not after_noun:
                if tok in self._NOISE_WORDS:
                    continue
                if tok in self._COPULAS:
                    continue
                if skip_superlative and tok in self._SUPERLATIVE_MAP:
                    continue
                noun = tok
                after_noun = True
            else:
                if tok in self._DESCRIPTOR_JOINS:
                    continue
                if tok in self._COPULAS:
                    continue
                if tok in self._NOISE_WORDS:
                    continue
                pre_desc.append(tok)

        if noun is None:
            return None

        # ── Phase 3: collect post-location descriptors from post_loc ──────────
        post_desc: List[str] = []
        for tok in post_loc:
            if tok in self._NOISE_WORDS:
                continue
            if tok in self._COPULAS:
                continue
            if tok in self._DESCRIPTOR_JOINS:
                continue
            post_desc.append(tok)

        all_desc = pre_desc + post_desc
        if all_desc:
            return f"{noun}_{'_'.join(all_desc)}"
        return noun

    def _extract_noun(self, tokens: List[str], skip_superlative: bool) -> Optional[str]:
        """
        Extract only the head noun (single word).

        Used for attribute and identity queries where descriptors would not
        be meaningful. For quantifier/superlative queries use
        _extract_noun_with_descriptors instead.
        """
        for tok in tokens:
            if tok in self._SPATIAL_PREPS:
                break
            if tok in self._NOISE_WORDS:
                continue
            if tok in self._COPULAS:
                continue
            if skip_superlative and tok in self._SUPERLATIVE_MAP:
                continue
            return tok
        return None

    def _extract_noun_after_attribute(
        self, tokens: List[str], attribute: str
    ) -> Optional[str]:
        """
        Extract the noun that appears after the attribute word.

        Example:
            tokens=["what","is","the","color","of","the","cup"], attribute="color"
            → "cup"
        """
        after_attr = False
        for tok in tokens:
            if tok == attribute:
                after_attr = True
                continue
            if not after_attr:
                continue
            if tok in self._NOISE_WORDS:
                continue
            if tok in self._SPATIAL_PREPS:
                break
            return tok
        return None


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
                logger.info(f"verb_to_cd loaded from {path} ({len(data)} verbs.")
            return data
        except json.JSONDecodeError as e:
            if logger:
                logger.error(f"Invalid JSON in {path}: {e}")
            return {}

    if logger:
        logger.error("verb_to_cd.json not found in any search path.")
    return {}
