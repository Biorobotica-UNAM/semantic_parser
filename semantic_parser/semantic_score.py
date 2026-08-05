"""Score semantic-parser CD outputs for shadow comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from semantic_parser.domain.entities import LOCATIONS, OBJECT_CATEGORIES


ALLOWED_PRIMITIVES = {
    "PTRANS",
    "ATRANS",
    "GRASP",
    "ACCOMPANY",
    "ATTEND",
    "QUERY",
    "MTRANS",
    "SPEAK",
    "SOCIAL",
    "WAIT",
}

REQUIRED_FIELDS = {
    "PTRANS": ("destination",),
    "ATRANS": ("object", "destination"),
    "GRASP": ("object",),
    "ACCOMPANY": ("person",),
    "ATTEND": ("subtype",),
    "QUERY": ("query_type", "target"),
    "MTRANS": ("content", "recipient"),
    "SPEAK": ("content",),
    "SOCIAL": ("subtype",),
    "WAIT": ("condition",),
}

ORDER_RANK = {
    "PTRANS": 10,
    "ATTEND": 20,
    "GRASP": 30,
    "ATRANS": 40,
    "ACCOMPANY": 40,
    "MTRANS": 50,
    "SPEAK": 50,
    "SOCIAL": 50,
    "WAIT": 60,
}


@dataclass
class ScoreResult:
    score: int = 0
    max_score: int = 18
    valid_json: bool = False
    valid_list: bool = False
    primitive_count: int = 0
    reasons: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.valid_json and self.valid_list and self.primitive_count > 0


def _walk_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def _has_none(value: Any) -> bool:
    return any(item is None for item in _walk_values(value))


def _mentioned_terms(command: str, candidates: tuple[str, ...]) -> set[str]:
    command = command.lower().replace("_", " ")
    return {term.replace(" ", "_") for term in candidates if term.lower() in command}


def _action_values(actions: list[dict[str, Any]]) -> set[str]:
    values = set()
    for action in actions:
        for value in _walk_values(action):
            if isinstance(value, str):
                values.add(value.lower())
    return values


def _mentions_are_resolved(
    mentioned_objects: set[str],
    mentioned_locations: set[str],
    values: set[str],
) -> bool:
    return mentioned_objects.issubset(values) and mentioned_locations.issubset(values)


def _order_is_reasonable(actions: list[dict[str, Any]]) -> bool:
    ranks = [ORDER_RANK.get(action.get("primitive", ""), 999) for action in actions]
    return ranks == sorted(ranks)


def score_cd_response(command: str, raw_response: str) -> ScoreResult:
    result = ScoreResult()

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        result.reasons.append(f"invalid_json: {exc}")
        return result

    result.valid_json = True
    result.score += 2

    if not isinstance(data, list):
        result.reasons.append("response_is_not_a_list")
        return result

    result.valid_list = True
    result.score += 2

    if not data:
        result.reasons.append("empty_primitive_list")
        return result

    if not all(isinstance(item, dict) for item in data):
        result.reasons.append("not_all_primitives_are_objects")
        return result

    result.actions = data
    result.primitive_count = len(data)

    if all("primitive" in item for item in data):
        result.score += 2
    else:
        result.reasons.append("missing_primitive_field")

    primitives = [str(item.get("primitive", "")) for item in data]
    if all(primitive in ALLOWED_PRIMITIVES for primitive in primitives):
        result.score += 2
    else:
        bad = sorted({primitive for primitive in primitives if primitive not in ALLOWED_PRIMITIVES})
        result.reasons.append(f"unknown_primitives: {bad}")

    missing_required = []
    for index, action in enumerate(data):
        primitive = str(action.get("primitive", ""))
        for field_name in REQUIRED_FIELDS.get(primitive, ()):
            value = action.get(field_name)
            if value in (None, "", "unknown"):
                missing_required.append(f"{index}:{primitive}.{field_name}")
    if not missing_required:
        result.score += 2
    else:
        result.reasons.append(f"missing_required: {missing_required}")

    mentioned_objects = _mentioned_terms(command, OBJECT_CATEGORIES)
    mentioned_locations = _mentioned_terms(command, LOCATIONS)
    values = _action_values(data)
    missing_mentions = sorted((mentioned_objects | mentioned_locations) - values)
    if not missing_mentions:
        result.score += 2
    else:
        result.reasons.append(f"mentioned_terms_missing: {missing_mentions}")

    if _mentions_are_resolved(mentioned_objects, mentioned_locations, values):
        result.score += 2
    else:
        result.reasons.append("known_terms_resolved_as_unknown")

    if _order_is_reasonable(data):
        result.score += 2
    else:
        result.reasons.append("primitive_order_inversion")

    if not _has_none(data):
        result.score += 2
    else:
        result.reasons.append("none_value_breaks_clips_contract")

    return result
