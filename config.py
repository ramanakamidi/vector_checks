"""
FieldMapper and ReportConfig — the extensibility layer.

Configure FieldMapper once for your database schema. Every check and report
reads from it to know which metadata fields to extract and display.

Usage::

    from config import FieldMapper, ReportConfig

    # Use defaults (doc_id, doc_name, text, page_number, …)
    fm = FieldMapper()

    # Or customise for your schema
    fm = FieldMapper(
        doc_id_field="document_id",
        text_field="chunk_text",
        page_field="page_num",
        required_fields=["document_id", "chunk_text"],
    )

    # Wrap in ReportConfig for report generation
    rc = ReportConfig(field_mapper=fm, max_text_preview_length=200)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class FieldMapper:
    """Maps your vector DB's metadata field names to framework concepts.

    Configure this once for your database schema. Every check and report
    reads from this config to know which fields to extract and display.
    """

    # Document identity
    doc_id_field: str = "doc_id"
    doc_name_field: Optional[str] = "doc_name"

    # Content
    text_field: Optional[str] = "text"  # original chunk text

    # Position within document
    page_field: Optional[str] = "page_number"
    chunk_index_field: Optional[str] = "chunk_index"

    # Classification
    object_type_field: Optional[str] = "object_type"
    category_field: Optional[str] = None

    # Provenance
    source_url_field: Optional[str] = "source_url"
    file_path_field: Optional[str] = "file_path"

    # Time (for drift detection)
    ingestion_date_field: Optional[str] = "ingestion_date"
    batch_id_field: Optional[str] = "batch_id"

    # Required fields for metadata completeness check
    required_fields: List[str] = field(default_factory=lambda: ["doc_id", "object_type"])

    # Fields to display in failure report drilldown tables
    display_fields: List[str] = field(default_factory=lambda: [
        "doc_id", "doc_name", "page_number", "object_type", "text"
    ])

    # Grouping for auto-pair generation
    similar_group_field: str = "doc_id"
    different_group_field: str = "object_type"
    drift_time_field: Optional[str] = "ingestion_date"

    def extract_display_metadata(self, metadata: dict) -> dict:
        """Extract only the display fields from a record's metadata."""
        result = {}
        for f in self.display_fields:
            if f in metadata:
                val = metadata[f]
                # Truncate long text for table display
                if isinstance(val, str) and len(val) > 150:
                    result[f] = val[:150] + "..."
                else:
                    result[f] = val
        return result

    def extract_full_metadata(self, metadata: dict) -> dict:
        """Extract all mapped fields from a record's metadata."""
        all_fields = [
            self.doc_id_field, self.doc_name_field, self.text_field,
            self.page_field, self.chunk_index_field, self.object_type_field,
            self.category_field, self.source_url_field, self.file_path_field,
            self.ingestion_date_field, self.batch_id_field,
        ]
        result = {}
        for f in all_fields:
            if f and f in metadata:
                result[f] = metadata[f]
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "FieldMapper":
        """Create from a dict (e.g., loaded from JSON config file)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Serialize to dict (e.g., for JSON config file)."""
        return asdict(self)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    field_mapper: FieldMapper = field(default_factory=FieldMapper)
    include_text_in_drilldowns: bool = True
    max_text_preview_length: int = 150
    max_sample_ids: int = 20
    include_vector_preview: bool = False  # show first N vector values
    vector_preview_dims: int = 10
