import rclpy
from rclpy.node import Node
import httpx

from semantic_parser_interfaces.srv import Parser


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "cd-qwen-big"


class OllamaNode(Node):
    def __init__(self):
        super().__init__("semantic_parser_node")

        self.client = httpx.Client(timeout=60.0)

        self.srv = self.create_service(
            Parser, "semantic_parser", self.handle_parser_service
        )

        self.get_logger().info("Semantic_parser service ready.")

        # Warmup opcional
        try:
            self.client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": "Hello",
                    "stream": False,
                },
            )
        except Exception as e:
            self.get_logger().warn(f"Ollama warmup failed: {e}")

    def handle_parser_service(self, request, response):
        command = request.command.strip()

        if not command:
            response.response = ""
            return response

        payload = {
            "model": MODEL_NAME,
            "prompt": command,
            "stream": False,
        }

        try:
            r = self.client.post(OLLAMA_URL, json=payload)
            response.response = r.json().get("response", "").strip()
        except Exception as e:
            self.get_logger().error(f"Ollama call failed: {e}")
            response.response = ""

        return response


def main():
    rclpy.init()
    node = OllamaNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
