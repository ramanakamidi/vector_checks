from __future__ import annotations

from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from services.visual_qa import load_workbench_qa_workbook, match_chunk_qa


def _workbook_without_source_columns() -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append([
        "pdf_name",
        "page_no",
        "object_id",
        "object_type",
        "object_no",
        "description",
        "question",
        "expected_answer",
        "chunk",
    ])
    worksheet.append([
        "sample.pdf",
        3,
        None,
        "text",
        None,
        "Details",
        "What does this passage explain?",
        "It explains the source passage.",
        "The full source passage contains the details.",
    ])
    worksheet.append([
        "sample.pdf",
        2,
        "doc-1:page-2:table-1",
        "table",
        "table 1",
        "Revenue table",
        "What does the table show?",
        "Revenue by year.",
        None,
    ])
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_load_and_match_workbook_using_existing_columns() -> None:
    rows = load_workbench_qa_workbook(_workbook_without_source_columns())

    text_row = match_chunk_qa(
        {
            "metadata": {
                "chunk_type": "text",
                "pdf_name": "sample.pdf",
                "section_path": "Details",
            },
            "text": "The full source passage contains the details.",
        },
        rows,
    )
    table_row = match_chunk_qa(
        {
            "metadata": {
                "chunk_type": "table",
                "pdf_name": "sample.pdf",
                "page_number": 2,
            },
            "text": "Revenue table",
        },
        rows,
    )

    assert len(text_row) == 1
    assert text_row[0]["question"] == "What does this passage explain?"
    assert text_row[0]["source_section"] == "Details"
    assert text_row[0]["source_text"] == "The full source passage contains the details."
    assert len(table_row) == 1
    assert table_row[0]["object_id"] == "doc-1:page-2:table-1"
    assert table_row[0]["source_pages"] == [2]


def test_rejects_workbooks_with_obsolete_source_columns() -> None:
    workbook = load_workbook(BytesIO(_workbook_without_source_columns()))
    worksheet = workbook.active
    worksheet.cell(row=1, column=worksheet.max_column + 1, value="source_type")
    output = BytesIO()
    workbook.save(output)

    with pytest.raises(ValueError, match="Regenerate it"):
        load_workbench_qa_workbook(output.getvalue())
