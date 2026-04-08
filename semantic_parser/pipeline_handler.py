"""
Abstract base class for the semantic parsing pipeline.

Implements the Chain of Responsibility pattern:
each handler processes one step of the pipeline and
passes the result to the next handler in the chain.

Usage:
    pipeline = (SRLHandler(srl)
                .set_next(MappingHandler(verb_map))
                .set_next(QueryExpansionHandler()))

    result = pipeline.handle({"command": "Go to the kitchen"})

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PipelineHandler(ABC):
    """
    Abstract base class for a pipeline step.

    Responsibility: define the contract that every handler
    in the chain must fulfill.

    Each concrete handler must implement handle(), which:
      1. Processes its specific step.
      2. Calls self._next.handle(data) to pass control forward.

    This class is never instantiated directly.
    """

    def __init__(self) -> None:
        self._next: Optional["PipelineHandler"] = None

    def set_next(self, handler: "PipelineHandler") -> "PipelineHandler":
        """
        Attach the next handler in the chain.

        Returns the handler itself so calls can be chained fluently:
            SRLHandler().set_next(MappingHandler()).set_next(ExpansionHandler())

        Args:
            handler: The next PipelineHandler to execute after this one.

        Returns:
            The handler passed in, enabling fluent chaining.
        """
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process this step and pass the result to the next handler.

        Args:
            data: A shared dict that travels through the entire pipeline.
                  Each handler reads from it and writes its output into it.

        Returns:
            The same dict, enriched with this handler's output.
        """
        ...

    def _pass_to_next(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pass data to the next handler if one exists.

        Concrete handlers should call this at the end of handle()
        instead of calling self._next.handle(data) directly.
        This keeps the chain-forwarding logic in one place.
        """
        if self._next:
            return self._next.handle(data)
        return data
