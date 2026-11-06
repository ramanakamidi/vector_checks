"""
Common interface every connector implements.

Every check in this framework is written against these five methods only.
This is what makes the framework database-agnostic: add a new connector
by implementing this interface, and every existing check works against it
with no changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class VectorRecord:
    """One stored item: its id, its vector, and its metadata tags."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]


class VectorDBConnector(ABC):
    """Abstract base class. Implement this once per database."""

    @abstractmethod
    def connect(self) -> None:
        """Open the connection / client. Called once before use."""
        raise NotImplementedError

    @abstractmethod
    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        """Get one record by id, or None if it doesn't exist."""
        raise NotImplementedError

    @abstractmethod
    def knn_search(self, vector: List[float], k: int = 10) -> List[VectorRecord]:
        """Return the k closest records to the given vector."""
        raise NotImplementedError

    @abstractmethod
    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        """Page through every stored record. Used for full audits."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Total number of records actually stored (raw count)."""
        raise NotImplementedError

    @abstractmethod
    def indexed_count(self) -> int:
        """
        Number of records the database reports as searchable
        (i.e. present in its internal search structure).
        On most databases this should equal count(); a mismatch
        means records exist but are not actually retrievable.
        """
        raise NotImplementedError

    def filter_metadata(self, field: str, value: Any, batch_size: int = 500) -> Iterator[VectorRecord]:
        """
        Default implementation: scan everything and filter client-side.
        Override in a connector if the database supports server-side
        metadata filtering (faster on large collections).
        """
        for record in self.scan(batch_size=batch_size):
            if record.metadata.get(field) == value:
                yield record
