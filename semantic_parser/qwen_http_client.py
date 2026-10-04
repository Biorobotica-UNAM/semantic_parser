"""Small HTTP client for the remote Qwen GPSR server."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class QwenResponse:
    goals: list[str]
    normalized_input: str
    elapsed_ms: float | None = None


class QwenHTTPClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_sec: float = 15.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_sec = timeout_sec

    def translate(self, command: str) -> QwenResponse:
        """Envía un comando y extrae metas; los errores se propagan al nodo ROS."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = httpx.post(
            f"{self._base_url}/translate",
            headers=headers,
            json={"command": command},
            timeout=self._timeout_sec,
        )
        # Distingue errores HTTP de respuestas con JSON o estructura inválidos.
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise RuntimeError(f"Qwen server rejected command: {payload}")

        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid Qwen result shape: {payload}")

        prediction = result.get("prediction", {})
        if not isinstance(prediction, dict):
            raise RuntimeError(f"Invalid Qwen prediction shape: {payload}")

        # Aquí se valida la forma; la sintaxis y semántica de metas las trata CLIPS.
        goals = prediction.get("goals", [])
        if not isinstance(goals, list) or not all(isinstance(goal, str) for goal in goals):
            raise RuntimeError(f"Invalid Qwen goals shape: {payload}")

        elapsed = payload.get("elapsed_ms")
        return QwenResponse(
            goals=[goal.strip() for goal in goals if goal.strip()],
            normalized_input=str(result.get("normalized_input", command)),
            elapsed_ms=float(elapsed) if isinstance(elapsed, (int, float)) else None,
        )
