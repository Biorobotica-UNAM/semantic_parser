"""
ROS2 service node for natural language semantic parsing.

Responsibility: expose the 'semantic_parser' ROS2 service and
delegate all processing to the internal pipeline.

The node itself knows nothing about NLP, verb mappings, or query
expansion — those are handled by the pipeline handlers.

Pipeline:
    SRLHandler → MappingHandler → QueryExpansionHandler → AttendNormalizationHandler

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import json
import time

import rclpy
from rclpy.node import Node
from semantic_parser_interfaces.srv import Parser

from semantic_parser.spacy_srl import SpacySRL
from semantic_parser.handlers import (
    SRLHandler,
    MappingHandler,
    QueryExpansionHandler,
    AttendNormalizationHandler,
    load_verb_map,
)


class SemanticParserNode(Node):
    """
    ROS2 node that exposes the 'semantic_parser' service.

    Responsibility: receive a natural language command,
    run it through the parsing pipeline, and return
    a JSON string with the resulting CD primitives.

    This class contains no NLP logic — it only wires together
    the pipeline and handles ROS2 communication.
    """

    _SPACY_MODEL = "en_core_web_trf"

    def __init__(self) -> None:
        super().__init__("semantic_parser_node")

        self._pipeline = self._build_pipeline()

        self._service = self.create_service(
            Parser,
            "semantic_parser",
            self._handle_request,
        )

        self.get_logger().info("Semantic parser service ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> SRLHandler:
        """
        Assemble the parsing pipeline.

        Returns the first handler in the chain. Calling handle()
        on it automatically runs all subsequent steps.

        To add a new step, create a new handler class in handlers.py
        and chain it here — no other code needs to change.
        """
        self.get_logger().info(f"Loading spaCy model ({self._SPACY_MODEL})...")
        srl = SpacySRL(model=self._SPACY_MODEL)
        self.get_logger().info("spaCy model ready.")

        verb_map = load_verb_map(self.get_logger())

        first_handler = SRLHandler(srl)
        (
            first_handler.set_next(MappingHandler(verb_map))
            .set_next(QueryExpansionHandler())
            .set_next(AttendNormalizationHandler())
        )

        return first_handler

    # ──────────────────────────────────────────────────────────────────────────
    # Service handler
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_request(self, request, response):
        """
        ROS2 service callback.

        Runs the pipeline on the incoming command and
        returns the result as a JSON string.
        """
        command = request.command.strip()

        if not command:
            self.get_logger().warning("Empty command received.")
            response.response = "[]"
            return response

        self.get_logger().info(f"Command: '{command}'")
        t0 = time.perf_counter()

        result = self._pipeline.handle({"command": command})
        cd_actions = result.get("cd_actions", [])

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(f"{len(cd_actions)} CD actions in {dt_ms:.1f} ms.")

        response.response = json.dumps(cd_actions)
        return response


# ============================================================


def main() -> None:
    rclpy.init()
    node = SemanticParserNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
