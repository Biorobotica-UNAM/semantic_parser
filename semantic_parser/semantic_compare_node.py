"""Shadow comparator for spaCy and Qwen semantic parser backends."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from semantic_parser_interfaces.srv import Parser

from semantic_parser.semantic_score import ScoreResult, score_cd_response


@dataclass
class ProviderResult:
    name: str
    ok: bool
    response: str
    elapsed_ms: float
    score: ScoreResult
    error: str = ""


class SemanticCompareNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_compare_node")
        self.declare_parameter("service_name", "semantic_parser")
        self.declare_parameter("spacy_service", "semantic_parser_spacy")
        self.declare_parameter("qwen_service", "semantic_parser_qwen")
        self.declare_parameter("return_backend", "best")
        self.declare_parameter("timeout_sec", 20.0)

        self._service_name = self.get_parameter("service_name").value
        self._return_backend = self.get_parameter("return_backend").value
        self._timeout_sec = float(self.get_parameter("timeout_sec").value)

        self._callback_group = ReentrantCallbackGroup()
        self._spacy_client = self.create_client(
            Parser,
            self.get_parameter("spacy_service").value,
            callback_group=self._callback_group,
        )
        self._qwen_client = self.create_client(
            Parser,
            self.get_parameter("qwen_service").value,
            callback_group=self._callback_group,
        )
        self._service = self.create_service(
            Parser,
            self._service_name,
            self._handle_request,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            f"Semantic compare service ready on '{self._service_name}' "
            f"(return_backend={self._return_backend})."
        )

    def _handle_request(self, request, response):
        command = request.command.strip()
        if not command:
            response.response = "[]"
            return response

        spacy = self._call_provider("spacy", self._spacy_client, command)
        qwen = self._call_provider("qwen", self._qwen_client, command)
        selected = self._select_result(spacy, qwen)

        self._log_comparison(command, spacy, qwen, selected)
        if selected.ok and selected.score.valid:
            response.response = selected.response
        else:
            response.response = json.dumps(
                {
                    "status": "aborted",
                    "reason": "No valid semantic parser response",
                    "spacy_error": spacy.error,
                    "qwen_error": qwen.error,
                }
            )
        return response

    def _call_provider(self, name: str, client, command: str) -> ProviderResult:
        if not client.wait_for_service(timeout_sec=self._timeout_sec):
            return self._failed_provider(name, "service unavailable")

        request = Parser.Request()
        request.command = command
        event = threading.Event()
        start = time.perf_counter()
        future = client.call_async(request)
        future.add_done_callback(lambda _: event.set())

        if not event.wait(self._timeout_sec):
            return self._failed_provider(name, f"timeout after {self._timeout_sec:.1f}s")

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            service_response = future.result()
            raw_response = service_response.response
            score = score_cd_response(command, raw_response)
            ok = score.valid
            return ProviderResult(name, ok, raw_response, elapsed_ms, score)
        except Exception as exc:  # noqa: BLE001 - keep comparator alive.
            return self._failed_provider(name, str(exc), elapsed_ms=elapsed_ms)

    def _failed_provider(self, name: str, error: str, elapsed_ms: float = 0.0) -> ProviderResult:
        score = ScoreResult(reasons=[error])
        return ProviderResult(name, False, "", elapsed_ms, score, error=error)

    def _select_result(self, spacy: ProviderResult, qwen: ProviderResult) -> ProviderResult:
        if self._return_backend == "spacy":
            return spacy
        if self._return_backend == "qwen":
            return qwen
        if qwen.score.score > spacy.score.score:
            return qwen
        return spacy

    def _log_comparison(
        self,
        command: str,
        spacy: ProviderResult,
        qwen: ProviderResult,
        selected: ProviderResult,
    ) -> None:
        summary = {
            "command": command,
            "selected": selected.name,
            "spacy": {
                "score": spacy.score.score,
                "max_score": spacy.score.max_score,
                "elapsed_ms": spacy.elapsed_ms,
                "reasons": spacy.score.reasons,
            },
            "qwen": {
                "score": qwen.score.score,
                "max_score": qwen.score.max_score,
                "elapsed_ms": qwen.elapsed_ms,
                "reasons": qwen.score.reasons,
            },
        }
        self.get_logger().info(f"Semantic comparison: {json.dumps(summary, ensure_ascii=False)}")


def main() -> None:
    rclpy.init()
    node = SemanticCompareNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
