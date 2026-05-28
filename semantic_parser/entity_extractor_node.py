"""
ROS2 service node for named-entity extraction from natural language.

Service interface: semantic_parser_interfaces/srv/ExtractEntity
    Request  → string sentence
               string category
    Response → string value
               string[] values

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from ament_index_python import get_package_share_directory

from semantic_parser_interfaces.srv import ExtractEntity
from semantic_parser.extractor import SentenceEntityExtractor


class EntityExtractorNode(Node):
    """ROS2 node that exposes the 'extract_entity' service."""

    _SERVICE_NAME = "extract_entity"
    _CONFIG_DIR = Path(get_package_share_directory("knowledge_bank")) / "dictionaries"

    def __init__(self) -> None:
        super().__init__("entity_extractor_node")

        self._extractor = self._build_extractor()

        self._service = self.create_service(
            ExtractEntity,
            self._SERVICE_NAME,
            self._handle_request,
        )

        self.get_logger().info(f"Service ready → '{self._SERVICE_NAME}'")
        self.get_logger().info(f"Dictionaries loaded from → '{self._CONFIG_DIR}'")

    def _build_extractor(self) -> SentenceEntityExtractor:
        self.get_logger().info("Loading entity catalogs...")

        extractor = SentenceEntityExtractor(
            config_dir=self._CONFIG_DIR,
        )

        self.get_logger().info("Entity catalogs ready.")

        return extractor

    def _handle_request(
        self,
        request: ExtractEntity.Request,
        response: ExtractEntity.Response,
    ) -> ExtractEntity.Response:

        sentence = request.sentence.strip().lower()
        category = request.category.strip().upper()

        if not sentence or not category:
            self.get_logger().warning("Empty sentence or category received.")
            response.value = ""
            return response

        self.get_logger().info(f"[{category}] '{sentence}'")

        t0 = time.perf_counter()

        result = self._extractor.extract(
            sentence,
            label=category,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.get_logger().info(f"Extracted in {dt_ms:.1f} ms → {result}")

        if isinstance(result.value, list):
            response.values = result.value
            response.value = ""
        else:
            response.value = result.value or ""
            response.values = []

        return response


def main() -> None:
    rclpy.init()

    node = EntityExtractorNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()
