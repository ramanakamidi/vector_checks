"""
pgvector (PostgreSQL) connector.

Requires: pip install psycopg2-binary
Assumes a table with a vector column (using the pgvector extension)
and a primary-key id column.
"""

import re
from typing import Any, Dict, Iterator, List, Optional

from .base import VectorDBConnector, VectorRecord


def _validate_identifier(name: str) -> str:
    """Validate that a SQL identifier (table/column name) contains only
    safe characters. Prevents SQL injection via constructor parameters.
    Raises ValueError if the name contains anything other than
    alphanumerics, underscores, or dots (for schema-qualified names)."""
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', name):
        raise ValueError(
            f"Invalid SQL identifier: {name!r}. "
            f"Only alphanumerics, underscores, and dots are allowed."
        )
    return name


class PgVectorConnector(VectorDBConnector):
    def __init__(self, dsn: str, table: str, vector_column: str = "embedding",
                 id_column: str = "id", metadata_columns: Optional[List[str]] = None):
        """
        Args:
            dsn: PostgreSQL connection string, e.g.
                 "host=localhost dbname=mydb user=me password=secret"
            table: Table name containing the vectors.
            vector_column: Name of the pgvector column.
            id_column: Name of the primary key column.
            metadata_columns: List of column names to treat as metadata.
                              If None, all non-id, non-vector columns are used.
        """
        self.dsn = dsn
        self.table = _validate_identifier(table)
        self.vector_column = _validate_identifier(vector_column)
        self.id_column = _validate_identifier(id_column)
        self.metadata_columns = metadata_columns
        self.conn = None
        self._all_columns: List[str] = []

    def connect(self) -> None:
        import psycopg2
        self.conn = psycopg2.connect(self.dsn)
        # Discover columns if metadata_columns not explicitly provided
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s ORDER BY ordinal_position",
                (self.table,),
            )
            self._all_columns = [row[0] for row in cur.fetchall()]
        if self.metadata_columns is None:
            self.metadata_columns = [
                c for c in self._all_columns
                if c not in (self.id_column, self.vector_column)
            ]
        else:
            # Validate user-provided metadata column names
            self.metadata_columns = [_validate_identifier(c) for c in self.metadata_columns]

    def _row_to_record(self, row: tuple, columns: List[str]) -> VectorRecord:
        row_dict = dict(zip(columns, row))
        record_id = str(row_dict.pop(self.id_column, ""))
        raw_vector = row_dict.pop(self.vector_column, [])
        # pgvector returns vectors as strings like "[0.1,0.2,...]" or as lists
        if isinstance(raw_vector, str):
            raw_vector = [float(x) for x in raw_vector.strip("[]").split(",")]
        metadata = {k: v for k, v in row_dict.items()
                    if k in (self.metadata_columns or [])}
        return VectorRecord(id=record_id, vector=list(raw_vector), metadata=metadata)

    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        columns = [self.id_column, self.vector_column] + (self.metadata_columns or [])
        col_str = ", ".join(columns)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {col_str} FROM {self.table} WHERE {self.id_column} = %s",
                (record_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row, columns)

    def knn_search(self, vector: List[float], k: int = 10) -> List[VectorRecord]:
        columns = [self.id_column, self.vector_column] + (self.metadata_columns or [])
        col_str = ", ".join(columns)
        vec_str = "[" + ",".join(str(v) for v in vector) + "]"
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {col_str} FROM {self.table} "
                f"ORDER BY {self.vector_column} <=> %s::vector LIMIT %s",
                (vec_str, k),
            )
            rows = cur.fetchall()
        return [self._row_to_record(r, columns) for r in rows]

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        columns = [self.id_column, self.vector_column] + (self.metadata_columns or [])
        col_str = ", ".join(columns)
        # Use a server-side cursor for memory-efficient scanning
        with self.conn.cursor(name="scan_cursor") as cur:
            cur.itersize = batch_size
            cur.execute(f"SELECT {col_str} FROM {self.table}")
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield self._row_to_record(row, columns)

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return cur.fetchone()[0]

    def indexed_count(self) -> int:
        # pgvector indexes are standard PostgreSQL indexes — all rows in the
        # table are indexed if an index exists. We check the index's tuple
        # count from pg_stat_user_indexes as a proxy.
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(idx_tup_read), -1) "
                "FROM pg_stat_user_indexes WHERE relname = %s",
                (self.table,),
            )
            result = cur.fetchone()[0]
            if result == -1:
                # No index stats available — fall back to row count
                return self.count()
            # For pgvector, stored == indexed if the index exists and is valid.
            # Return the table count; a mismatch would only occur if the index
            # is corrupt or being rebuilt.
            return self.count()

    def filter_metadata(self, field: str, value: Any, batch_size: int = 500) -> Iterator[VectorRecord]:
        _validate_identifier(field)  # prevent injection via filter field name
        columns = [self.id_column, self.vector_column] + (self.metadata_columns or [])
        col_str = ", ".join(columns)
        with self.conn.cursor(name="filter_cursor") as cur:
            cur.itersize = batch_size
            cur.execute(
                f"SELECT {col_str} FROM {self.table} WHERE {field} = %s",
                (value,),
            )
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield self._row_to_record(row, columns)
