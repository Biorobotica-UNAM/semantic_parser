"""ROS2 Parser service backed by the remote Qwen HTTP server."""

from __future__ import annotations

import json
import os
import time

import httpx

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from listener_interfaces.srv import Listen
from semantic_parser_interfaces.srv import Parser

from semantic_parser.qwen_http_client import QwenHTTPClient


class QwenSemanticNode(Node):
    def __init__(self) -> None:
        super().__init__("qwen_semantic_node")
        self.declare_parameter("service_name", "semantic_parser_qwen")
        self.declare_parameter("qwen_url", os.environ.get("QWEN_SERVER_URL", "http://127.0.0.1:8008"))
        self.declare_parameter("qwen_api_key", os.environ.get("QWEN_API_KEY", ""))
        self.declare_parameter("timeout_sec", float(os.environ.get("QWEN_TIMEOUT_SEC", "15.0")))
        self.declare_parameter("response_format", os.environ.get("QWEN_RESPONSE_FORMAT", "goals"))
        self.declare_parameter("gpsr_listen_service_name", "listen_gpsr")
        self.declare_parameter("listener_service", "/hri/listener/listen")
        self.declare_parameter("gpsr_listener_grammar", "")
        self.declare_parameter("gpsr_listen_timeout_sec", 30.0)

        self._service_name = self.get_parameter("service_name").value
        # Los callbacks pueden esperar al listener sin impedir su respuesta.
        self._callback_group = ReentrantCallbackGroup()
        qwen_url = self.get_parameter("qwen_url").value or os.environ.get(
            "QWEN_SERVER_URL",
            "http://127.0.0.1:8008",
        )
        api_key = self.get_parameter("qwen_api_key").value or os.environ.get("QWEN_API_KEY", "")
        self._response_format = str(self.get_parameter("response_format").value or "goals").lower()
        # El adaptador CLIPS consume metas JSON; el antiguo modo CD fue retirado.
        if self._response_format != "goals":
            raise ValueError("response_format must be 'goals'; CD support was removed")
        self._client = QwenHTTPClient(
            base_url=qwen_url,
            api_key=api_key,
            timeout_sec=float(self.get_parameter("timeout_sec").value),
        )
        self._service = self.create_service(
            Parser,
            self._service_name,
            self._handle_request,
            callback_group=self._callback_group,
        )
        self._listener_service_name = self.get_parameter("listener_service").value
        self._gpsr_listener_grammar = self.get_parameter("gpsr_listener_grammar").value
        self._gpsr_listen_timeout_sec = float(
            self.get_parameter("gpsr_listen_timeout_sec").value
        )
        self._listener_client = self.create_client(
            Listen,
            self._listener_service_name,
            callback_group=self._callback_group,
        )
        self._gpsr_listen_service_name = self.get_parameter(
            "gpsr_listen_service_name"
        ).value
        self._gpsr_listen_service = self.create_service(
            Parser,
            self._gpsr_listen_service_name,
            self._handle_gpsr_listen_request,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            f"Qwen service ready on '{self._service_name}' "
            f"url='{qwen_url}' response_format='{self._response_format}'."
        )
        self.get_logger().info(
            f"GPSR listen service ready on '{self._gpsr_listen_service_name}' "
            f"using listener='{self._listener_service_name}' "
            f"grammar='{self._gpsr_listener_grammar}'."
        )

    def _handle_request(self, request, response):
        command = request.command.strip()
        if not command:
            response.response = "[]"
            return response

        # Devuelve metas; el cliente coordinador realiza después la llamada a CLIPS.
        response.response = self._translate_command(command)
        return response

    def _handle_gpsr_listen_request(self, request, response):
        # Este servicio escucha un comando nuevo; no utiliza el texto de la petición.
        del request
        command = self._listen_for_gpsr_command()
        if not command:
            response.response = json.dumps(
                {
                    "status": "aborted",
                    "reason": "No GPSR command was heard by the listener.",
                }
            )
            return response

        response.response = self._translate_command(command)
        return response

    def _listen_for_gpsr_command(self) -> str:
        if not self._listener_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                f"Listener service unavailable: {self._listener_service_name}"
            )
            return ""

        request = Listen.Request()
        request.enable = True
        request.grammar = self._gpsr_listener_grammar

        future = self._listener_client.call_async(request)
        # El segundo hilo del executor atiende la respuesta mientras este callback espera.
        deadline = time.monotonic() + self._gpsr_listen_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                break
            time.sleep(0.05)

        if not future.done():
            self.get_logger().error(
                f"Timed out waiting for GPSR listener after "
                f"{self._gpsr_listen_timeout_sec:.1f} s"
            )
            return ""

        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - expose listener failure.
            self.get_logger().error(f"GPSR listener request failed: {exc}")
            return ""

        if not result.success or not result.recognized_speech.strip():
            self.get_logger().warning(
                "GPSR listener returned no command: "
                f"success={result.success} text='{result.recognized_speech}'"
            )
            return ""

        command = result.recognized_speech.strip()
        self.get_logger().info(f"GPSR listener heard: '{command}'")
        return command

    def _translate_command(self, command: str) -> str:
        t0 = time.perf_counter()
        try:
            qwen_response = self._client.translate(command)
        except Exception as exc:  # noqa: BLE001 - expose failure to comparator.
            self.get_logger().error(f"Qwen semantic request failed: {exc}")
            return json.dumps(
                {
                    "status": "aborted",
                    "input": command,
                    "reason": str(exc),
                    "error_code": (
                        "gpsr_unavailable"
                        if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))
                        else "semantic_error"
                    ),
                }
            )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Se conserva el orden de metas: CLIPS las procesa por su posición en la lista.
        payload = {
            "input": qwen_response.normalized_input or command,
            "goals": qwen_response.goals,
        }
        self.get_logger().info(
            f"Qwen goals={qwen_response.goals} in {dt_ms:.1f} ms."
        )
        return json.dumps(payload)


def main() -> None:
    rclpy.init()
    node = QwenSemanticNode()
    # Necesario para que el callback bloqueante de escucha reciba su future.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()
