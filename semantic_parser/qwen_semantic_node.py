"""ROS2 Parser service backed by the remote Qwen HTTP server."""

from __future__ import annotations

import json
import os
import time

import rclpy
from rclpy.node import Node
from semantic_parser_interfaces.srv import Parser

from semantic_parser.qwen_goal_adapter import goals_to_cd
from semantic_parser.qwen_http_client import QwenHTTPClient


class QwenSemanticNode(Node):
    def __init__(self) -> None:
        super().__init__("qwen_semantic_node")
        self.declare_parameter("service_name", "semantic_parser_qwen")
        self.declare_parameter("qwen_url", os.environ.get("QWEN_SERVER_URL", "http://127.0.0.1:8008"))
        self.declare_parameter("qwen_api_key", os.environ.get("QWEN_API_KEY", ""))
        self.declare_parameter("timeout_sec", float(os.environ.get("QWEN_TIMEOUT_SEC", "15.0")))
        self.declare_parameter("response_format", os.environ.get("QWEN_RESPONSE_FORMAT", "goals"))

        self._service_name = self.get_parameter("service_name").value
        qwen_url = self.get_parameter("qwen_url").value or os.environ.get(
            "QWEN_SERVER_URL",
            "http://127.0.0.1:8008",
        )
        api_key = self.get_parameter("qwen_api_key").value or os.environ.get("QWEN_API_KEY", "")
        self._response_format = str(self.get_parameter("response_format").value or "goals").lower()
        self._client = QwenHTTPClient(
            base_url=qwen_url,
            api_key=api_key,
            timeout_sec=float(self.get_parameter("timeout_sec").value),
        )
        self._service = self.create_service(Parser, self._service_name, self._handle_request)
        self.get_logger().info(
            f"Qwen service ready on '{self._service_name}' "
            f"url='{qwen_url}' response_format='{self._response_format}'."
        )

    def _handle_request(self, request, response):
        command = request.command.strip()
        if not command:
            response.response = "[]"
            return response

        t0 = time.perf_counter()
        try:
            qwen_response = self._client.translate(command)
        except Exception as exc:  # noqa: BLE001 - expose failure to comparator.
            self.get_logger().error(f"Qwen semantic request failed: {exc}")
            response.response = json.dumps({"status": "aborted", "reason": str(exc)})
            return response

        dt_ms = (time.perf_counter() - t0) * 1000.0
        if self._response_format == "cd":
            cd_actions = goals_to_cd(qwen_response.goals)
            response.response = json.dumps(cd_actions)
            self.get_logger().info(
                f"Qwen goals={qwen_response.goals} -> {len(cd_actions)} CD actions in {dt_ms:.1f} ms."
            )
            return response

        response.response = json.dumps(
            {
                "input": qwen_response.normalized_input or command,
                "goals": qwen_response.goals,
            }
        )
        self.get_logger().info(
            f"Qwen goals={qwen_response.goals} in {dt_ms:.1f} ms."
        )
        return response


def main() -> None:
    rclpy.init()
    node = QwenSemanticNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
