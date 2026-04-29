"""
ROS2 service node for named-entity extraction from natural language.

Responsibility: expose the 'extract_entity' ROS2 service and
delegate all NLP processing to SentenceEntityExtractor.

This node knows nothing about spaCy internals, patterns, or handlers —
those are encapsulated in extractor.py.

Service interface: semantic_parser_interfaces/srv/ExtractEntity
    Request  → string sentence
    Response → string name
               string drink

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import time

import rclpy
from rclpy.node import Node
from semantic_parser_interfaces.srv import ExtractEntity

from semantic_parser.extractor import SentenceEntityExtractor


class EntityExtractorNode(Node):
    """
    ROS2 node that exposes the 'extract_entity' service.

    Responsibility: receive a natural language sentence,
    run it through SentenceEntityExtractor, and return
    the extracted name and drink as separate response fields.

    This class contains no NLP logic — it only wires together
    the extractor and handles ROS2 communication.
    """

    _SPACY_MODEL = "en_core_web_trf"
    _SERVICE_NAME = "extract_entity"

    def __init__(self) -> None:
        super().__init__("entity_extractor_node")

        self._extractor = self._build_extractor()

        self._service = self.create_service(
            ExtractEntity,
            self._SERVICE_NAME,
            self._handle_request,
        )

        self.get_logger().info(
            f"Entity extractor service ready → '{self._SERVICE_NAME}'"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Extractor construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_extractor(self) -> SentenceEntityExtractor:
        """
        Instantiate the entity extractor.

        Separated into its own method so subclasses or tests can
        override the model without touching the constructor.
        """
        self.get_logger().info(f"Loading spaCy model ({self._SPACY_MODEL})...")
        extractor = SentenceEntityExtractor(model=self._SPACY_MODEL)
        self.get_logger().info("spaCy model ready.")
        return extractor

    # ──────────────────────────────────────────────────────────────────────────
    # Service callback
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_request(
        self,
        request: ExtractEntity.Request,
        response: ExtractEntity.Response,
    ) -> ExtractEntity.Response:
        """
        ROS2 service callback.

        Runs the extractor on the incoming sentence and
        populates the response fields individually.
        """
        sentence = request.sentence.strip()

        if not sentence:
            self.get_logger().warning("Empty sentence received.")
            response.name = ""
            response.drink = ""
            return response

        self.get_logger().info(f"Sentence: '{sentence}'")
        t0 = time.perf_counter()

        result = self._extractor.extract(sentence)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"Extracted in {dt_ms:.1f} ms → name='{result.name}' drink='{result.drink}'"
        )

        response.name = result.name or ""
        response.drink = result.drink or ""
        return response


# ============================================================


def main() -> None:
    rclpy.init()
    node = EntityExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
