"""
pipeline/pipeline_handler.py

Abstract base class for the semantic parsing pipeline.

Implements the Chain of Responsibility pattern (GoF):
each handler processes one step of the pipeline and
passes the enriched data dict to the next handler.

Usage
─────
    pipeline = (SRLHandler(srl)
                .set_next(MappingHandler(verb_map))
                .set_next(QueryExpansionHandler())
                .set_next(AttendNormalizationHandler()))

    result = pipeline.handle({"command": "Go to the kitchen"})

Design notes
────────────
• Single Responsibility: this module only defines the chain contract.
• Interface Segregation: handle() is the only public method callers need.
• Open/Closed: new steps are added by subclassing, not by modifying this file.

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PipelineHandler(ABC):
    """
    Abstract base for a single pipeline step.

    Each concrete handler must implement handle(), which:
      1. Processes its specific transformation.
      2. Calls self._pass_to_next(data) to forward control.
    """

    def __init__(self) -> None:
        self._next: Optional["PipelineHandler"] = None

    def set_next(self, handler: "PipelineHandler") -> "PipelineHandler":
        """
        Attach the next handler and return it for fluent chaining.

            SRLHandler().set_next(MappingHandler()).set_next(ExpansionHandler())
        """
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process this step and pass the result forward.

        Args:
            data: Shared dict that travels through the entire pipeline.
                  Each handler reads from it and writes its output into it.

        Returns:
            The same dict, enriched with this handler's output.
        """
        ...

    def _pass_to_next(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward data to the next handler if one exists."""
        if self._next:
            return self._next.handle(data)
        return data
