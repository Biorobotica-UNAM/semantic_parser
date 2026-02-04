import rclpy
import httpx
import json
import os
from rclpy.logging import set_logger_level
from rclpy.node import Node
from semantic_parser_interfaces.srv import Parser
from typing import Any, Dict, List


def load_verb_to_primitive(file_name: str) -> dict:
    """Carga el JSON desde la carpeta del script."""
    base_dir = os.path.dirname(__file__)  # carpeta semantic_parser
    file_path = os.path.join(base_dir, file_name)
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Error loading verb_to_cd JSON:", e)
        return {}


VERB_TO_PRIMITIVE = load_verb_to_primitive("verb_to_cd.json")


def map_fields_by_primitive(
    primitive: str, actor: str, object_name: str, source: str, destination: str
) -> Dict[str, Any]:
    if primitive in {"PTRANS", "ATRANS"}:
        return {
            "primitive": primitive,
            "actor": actor,
            "object": object_name,
            "from": source,
            "to": destination,
        }
    if primitive in {"GRASP", "ATTEND", "INGEST"}:
        return {"primitive": primitive, "actor": actor, "object": object_name}
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
    return {"primitive": "DO", "actor": actor, "action": primitive}


def map_to_primitive(action_json: Dict[str, Any], logger) -> Dict[str, Any]:
    verb = action_json.get("action", "unknown").lower()
    object_name = action_json.get("object", "unknown")
    source = action_json.get("source", "unknown")
    destination = action_json.get("destination", "unknown")
    actor = action_json.get("actor", "robot")

    primitive = VERB_TO_PRIMITIVE.get(verb, "DO")

    if primitive == "SPEAK" and destination != "unknown":
        primitive = "MTRANS"

    if primitive == "DO":
        logger.warning(f"No se pudo mapear la action: {verb}")
    return map_fields_by_primitive(primitive, actor, object_name, source, destination)


def llm_generate_cd(client, command: str) -> List[Dict[str, Any]]:
    payload = {"model": "SRL-qwen", "prompt": command, "stream": False}
    try:
        r = client.post("http://localhost:11434/api/generate", json=payload)
        raw_text = r.json().get("response", "").strip()
        return json.loads(raw_text)
    except Exception as e:
        print("LLM or JSON parse error:", e)
        return []


# --- Nodo ROS2 ---
class OllamaNode(Node):
    def __init__(self):
        super().__init__("semantic_parser_node")
        self.client = httpx.Client(timeout=60.0)
        self.srv = self.create_service(
            Parser, "semantic_parser", self.handle_parser_service
        )
        self.get_logger().info("Semantic_parser service ready.")

    def handle_parser_service(self, request, response):
        command = request.command.strip()
        if not command:
            response.response = ""
            return response

        actions = llm_generate_cd(self.client, command)
        cd_actions = [map_to_primitive(action, self.get_logger()) for action in actions]

        response.response = json.dumps(cd_actions)
        return response


def main():
    rclpy.init()
    node = OllamaNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
