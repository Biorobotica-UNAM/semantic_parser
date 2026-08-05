"""
semantic_node.py

ROS2 service node for natural language semantic parsing.

Responsibility: expose the 'semantic_parser' ROS2 service and
delegate all processing to the internal pipeline.

The node itself knows nothing about NLP, verb mappings, or query
expansion — those are handled by the pipeline handlers.

Pipeline (v2)
─────────────
    SRLHandler
        → MappingHandler
            → QueryExpansionHandler
                → AttendNormalizationHandler
                    → SerializationHandler

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

import json
import time

import rclpy
from rclpy.node import Node
from semantic_parser_interfaces.srv import Parser

from semantic_parser.nlp.spacy_pipeline import SpacyPipelineFactory
from semantic_parser.nlp.spacy_srl import SpacySRL
from semantic_parser.pipeline.handlers import (
    SRLHandler,
    MappingHandler,
    QueryExpansionHandler,
    AttendNormalizationHandler,
    ValidationHandler,
    SerializationHandler,
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
        self.declare_parameter("service_name", "semantic_parser")
        self.declare_parameter("spacy_model", self._SPACY_MODEL)
        self._service_name = (
            self.get_parameter("service_name").get_parameter_value().string_value
        )
        self._spacy_model = (
            self.get_parameter("spacy_model").get_parameter_value().string_value
        )
        self._pipeline = self._build_pipeline()
        self._service = self.create_service(
            Parser,
            self._service_name,
            self._handle_request,
        )
        self.get_logger().info(f"Semantic parser service ready on '{self._service_name}'.")

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> SRLHandler:
        """
        Assemble the parsing pipeline.

        Returns the first handler in the chain. Calling handle() on it
        automatically runs all subsequent steps.

        To add a new step: create a new handler in pipeline/handlers.py
        and chain it here — no other code needs to change.
        """
        self.get_logger().info(f"Loading spaCy model ({self._spacy_model})...")
        nlp = SpacyPipelineFactory.build(self._spacy_model)
        self.get_logger().info("spaCy pipeline ready (EntityRuler + PersonDescTagger).")

        srl = SpacySRL(nlp)
        verb_map = load_verb_map(self.get_logger())

        first = SRLHandler(srl)
        (
            first.set_next(MappingHandler(verb_map))
            .set_next(QueryExpansionHandler())
            .set_next(AttendNormalizationHandler())
            .set_next(ValidationHandler())
            .set_next(SerializationHandler())
        )
        return first

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

        # Validation rejected the command — return an aborted response.
        if result.get("aborted"):
            reason = result.get("abort_reason", "Unknown validation error.")
            self.get_logger().warning(f"Command aborted: {reason}")
            response.response = json.dumps({"status": "aborted", "reason": reason})
            return response

        cd_actions = result.get("cd_actions", [])

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(f"{len(cd_actions)} CD actions in {dt_ms:.1f} ms.")

        response.response = json.dumps(cd_actions)
        return response


# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    rclpy.init()
    node = SemanticParserNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
