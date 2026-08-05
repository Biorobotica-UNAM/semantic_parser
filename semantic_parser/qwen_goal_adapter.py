"""Convert Qwen goal strings into CD primitive dictionaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from semantic_parser.domain.entities import OBJECT_CATEGORY_MAP, PERSON_NAMES


_GOAL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$")


@dataclass(frozen=True)
class ParsedGoal:
    name: str
    args: list[str]
    kwargs: dict[str, str]


def _normalize_value(value: str) -> str:
    value = value.strip().strip("\"'")
    value = value.replace(" ", "_")
    return value.lower()


def _split_args(raw: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""

    for char in raw:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                args.append(item)
            current = []
            continue
        current.append(char)

    item = "".join(current).strip()
    if item:
        args.append(item)
    return args


def parse_goal(goal: str) -> ParsedGoal:
    match = _GOAL_RE.match(goal)
    if not match:
        raise ValueError(f"Invalid goal syntax: {goal}")

    name = match.group(1).lower()
    args: list[str] = []
    kwargs: dict[str, str] = {}
    for item in _split_args(match.group(2)):
        if "=" in item:
            key, value = item.split("=", 1)
            kwargs[key.strip().lower()] = _normalize_value(value)
        else:
            args.append(_normalize_value(item))
    return ParsedGoal(name=name, args=args, kwargs=kwargs)


def _object_category(obj: str) -> str | None:
    return OBJECT_CATEGORY_MAP.get(obj.replace("_", " "))


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _person_descriptor(value: str) -> dict[str, str]:
    if value.title() in PERSON_NAMES:
        return {"name": value}
    if value in {"sitting", "standing", "lying", "waving"}:
        return {"pose": value}
    if value in {"pointing", "raising"}:
        return {"gesture": value}
    return {"name": value} if value not in {"person", "someone"} else {}


def _adapt_goal(parsed: ParsedGoal) -> list[dict[str, Any]]:
    first = parsed.args[0] if parsed.args else "unknown"

    if parsed.name in {"go", "move", "navigate", "head"}:
        return [
            {
                "primitive": "PTRANS",
                "actor": "robot",
                "subtype": "navigate",
                "destination": first,
                "source": parsed.kwargs.get("from", "unknown"),
            }
        ]

    if parsed.name in {"find", "locate", "search"}:
        if first.title() in PERSON_NAMES or first in {"person", "someone"}:
            return [
                {
                    "primitive": "ATTEND",
                    "actor": "robot",
                    "subtype": "find_person",
                    "object": "unknown",
                    "person": _person_descriptor(first),
                    "location": parsed.kwargs.get("in", "unknown"),
                }
            ]
        return [
            _clean(
                {
                    "primitive": "ATTEND",
                    "actor": "robot",
                    "subtype": "find_object",
                    "object": first,
                    "object_category": _object_category(first),
                    "location": parsed.kwargs.get("in", "unknown"),
                }
            )
        ]

    if parsed.name in {"take", "grasp", "pick", "pickup", "get"}:
        return [
            _clean(
                {
                    "primitive": "GRASP",
                    "actor": "robot",
                    "subtype": "pick_up",
                    "object": first,
                    "object_category": _object_category(first),
                    "location": parsed.kwargs.get("from", "unknown"),
                }
            )
        ]

    if parsed.name in {"deliver", "bring", "give"}:
        destination = parsed.kwargs.get("to") or (parsed.args[1] if len(parsed.args) > 1 else "user")
        if destination in {"me", "user"}:
            destination = "user"
            subtype = "give"
        else:
            subtype = "bring"
        return [
            _clean(
                {
                    "primitive": "ATRANS",
                    "actor": "robot",
                    "subtype": subtype,
                    "object": first,
                    "object_category": _object_category(first),
                    "source": parsed.kwargs.get("from", "unknown"),
                    "destination": destination,
                }
            )
        ]

    if parsed.name in {"guide", "escort", "lead", "follow"}:
        person = first if first not in {"unknown", ""} else "person"
        return [
            {
                "primitive": "ACCOMPANY",
                "actor": "robot",
                "subtype": "follow" if parsed.name == "follow" else "guide",
                "person": _person_descriptor(person),
                "source": parsed.kwargs.get("from", "unknown"),
                "destination": parsed.kwargs.get("to", "unknown"),
            }
        ]

    if parsed.name in {"count"}:
        return [
            _clean(
                {
                    "primitive": "ATTEND",
                    "actor": "robot",
                    "subtype": "count_objects",
                    "object": first,
                    "object_category": _object_category(first),
                    "location": parsed.kwargs.get("in", "unknown"),
                    "query_attribute": "count",
                }
            )
        ]

    if parsed.name in {"talk", "say", "answer"}:
        return [
            {
                "primitive": "SPEAK",
                "actor": "robot",
                "subtype": "answer_quiz",
                "content": first,
                "recipient": parsed.kwargs.get("to", "user"),
            }
        ]

    if parsed.name in {"wait"}:
        return [{"primitive": "WAIT", "actor": "robot", "condition": first}]

    raise ValueError(f"Unsupported Qwen goal: {parsed.name}")


def goals_to_cd(goals: list[str]) -> list[dict[str, Any]]:
    cd_actions: list[dict[str, Any]] = []
    for goal in goals:
        cd_actions.extend(_adapt_goal(parse_goal(goal)))
    return cd_actions
