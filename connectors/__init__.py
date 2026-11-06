"""
Database connectors package.

Each connector implements the VectorDBConnector interface from base.py,
making every check in the framework database-agnostic.
"""

from .base import VectorDBConnector, VectorRecord

__all__ = ["VectorDBConnector", "VectorRecord"]
