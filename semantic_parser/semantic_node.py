"""
semantic_parser_node.py
Nodo ROS2 que expone el servicio 'semantic_parser'.
Pipeline: lenguaje natural → spaCy SRL → primitivas CD → JSON
"""

import rclpy
import json
import os
import time
from rclpy.node import Node
from semantic_parser_interfaces.srv import Parser
from typing import Any, Dict, List
from pathlib import Path
from semantic_parser.spacy_srl import SpacySRL


# ============================================================
# =================== FILE UTILITIES =========================
# ============================================================


def load_verb_to_primitive(file_name: str, logger=None) -> dict:
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
        if path.exists():
            if logger:
                logger.info(f"Loading verb_to_cd from: {path}")
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if logger:
                        logger.info(f"verb_to_cd loaded: {len(data)} verbs.")
                    return data
            except json.JSONDecodeError as e:
                if logger:
                    logger.error(f"Invalid JSON: {e}")
                return {}
    if logger:
        logger.error(f"verb_to_cd.json not found in any search path.")
    return {}


# ============================================================
# ================ QUERY RESOLUTION ==========================
# ============================================================

QUERY_PREFIXES = (
    "how_many_",
    "how_much_",
    "what_",
    "where_",
    "which_",
    "who_",
    "whether_",
    "if_",
)
SPATIAL_PREPS = ["_on_", "_in_", "_at_", "_near_", "_beside_", "_inside_", "_next_to_"]
STOP_WORDS = [
    "_wearing_",
    "_holding_",
    "_carrying_",
    "_with_",
    "_that_",
    "_who_",
    "_are_",
]


def is_perceptual_query(obj: str) -> bool:
    # Queries WH explícitas
    if obj.lower().startswith(QUERY_PREFIXES):
        return True
    # Queries nominales: sustantivos perceptuales sin prefijo WH
    # "the_pose_of...", "the_name_of...", "the_color_of...", etc.
    PERCEPTUAL_NOUNS = {
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
    # Extraer el primer sustantivo del snake_case (ignorar "the_", "a_")
    parts = obj.lower().lstrip("the_").lstrip("a_").split("_")
    if parts and parts[0] in PERCEPTUAL_NOUNS:
        return True
    return False


def extract_location_from_object(obj: str) -> str:
    lower = obj.lower()
    for prep in SPATIAL_PREPS:
        if prep in lower:
            after = lower.split(prep, 1)[1]
            for stop in STOP_WORDS:
                if stop in after:
                    after = after.split(stop)[0]
            VERB_STOPS = ["_are", "_is", "_was", "_were", "_have", "_has"]
            for vs in VERB_STOPS:
                if after.endswith(vs):
                    after = after[: -len(vs)]
                elif vs + "_" in after:
                    after = after.split(vs + "_")[0]
            return after
    return "unknown"


def resolve_query(cd: Dict[str, Any]) -> List[Dict[str, Any]]:
    obj = cd.get("object", "unknown")
    actor = cd.get("actor", "robot")
    destination = cd.get("to", "user")

    if destination == "unknown":
        destination = "user"

    if is_perceptual_query(obj):
        location = extract_location_from_object(obj)
        return [
            {
                "primitive": "ATTEND",
                "actor": actor,
                "object": obj,
                "location": location,
            },
            {"primitive": "MTRANS", "actor": actor, "content": obj, "to": destination},
        ]
    return [{"primitive": "SPEAK", "actor": actor, "content": obj}]


# ============================================================
# ================ CD MAPPING ================================
# ============================================================


def map_fields_by_primitive(
    primitive: str,
    actor: str,
    object_name: str,
    source: str,
    destination: str,
    location: str,
) -> Dict[str, Any]:
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
        data = {"primitive": primitive, "actor": actor, "object": object_name}
        if location != "unknown":
            data["location"] = location
        return data
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
    return {"primitive": "DO", "actor": actor, "action": primitive}


def map_to_primitive(
    action_json: Dict[str, Any], verb_map: dict, logger
) -> Dict[str, Any]:
    verb = action_json.get("action", "unknown").lower()
    object_name = action_json.get("object", "unknown")
    source = action_json.get("source", "unknown")
    destination = action_json.get("destination", "unknown")
    location = action_json.get("location", "unknown")
    actor = action_json.get("actor", "robot")

    # Normalizar pronombres humanos como destinatario -> "user"
    if destination.lower() in {"me", "us", "you"}:
        destination = "user"

    primitive = verb_map.get(verb, "DO")

    # SPEAK con destinatario explícito → MTRANS
    if primitive == "SPEAK" and destination != "unknown":
        primitive = "MTRANS"

    if primitive == "DO":
        logger.warning(f"Unmapped verb: '{verb}'")

    return map_fields_by_primitive(
        primitive, actor, object_name, source, destination, location
    )


def expand_cd_actions(cd_actions: List[Dict[str, Any]], logger) -> List[Dict[str, Any]]:
    expanded = []
    for cd in cd_actions:
        if cd.get("primitive") == "QUERY":
            resolved = resolve_query(cd)
            logger.info(f"QUERY → {len(resolved)} actions: {json.dumps(resolved)}")
            expanded.extend(resolved)
        else:
            expanded.append(cd)
    return expanded


# ============================================================
# ===================== ROS2 NODE ============================
# ============================================================


class SemanticParserNode(Node):

    def __init__(self):
        super().__init__("semantic_parser_node")

        self.verb_map = load_verb_to_primitive("verb_to_cd.json", self.get_logger())

        # Carga el modelo spaCy una sola vez al iniciar el nodo
        self.get_logger().info("Loading spaCy model (en_core_web_trf)...")
        self.srl = SpacySRL(model="en_core_web_trf")
        self.get_logger().info("spaCy model ready.")

        self.srv = self.create_service(
            Parser, "semantic_parser", self.handle_parser_service
        )
        self.get_logger().info("Semantic parser service ready.")

    def handle_parser_service(self, request, response):
        command = request.command.strip()

        if not command:
            self.get_logger().warning("Empty command received.")
            response.response = "[]"
            return response

        self.get_logger().info(f"Command: '{command}'")
        t0 = time.perf_counter()

        # 1) spaCy SRL: lenguaje natural → roles semánticos
        raw_actions = self.srl.parse(command)
        self.get_logger().info(f"spaCy extracted {len(raw_actions)} raw actions.")

        # 2) Mapeo: verbo → primitiva CD
        cd_actions = [
            map_to_primitive(a, self.verb_map, self.get_logger()) for a in raw_actions
        ]

        # 3) Expansión: QUERY → [ATTEND, MTRANS]
        cd_actions = expand_cd_actions(cd_actions, self.get_logger())

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(f"{len(cd_actions)} CD actions in {dt_ms:.1f} ms.")

        response.response = json.dumps(cd_actions)
        return response


# ============================================================


def main():
    rclpy.init()
    node = SemanticParserNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
