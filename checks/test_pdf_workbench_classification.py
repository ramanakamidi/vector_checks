from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdf_workbench import custom_logic


class PdfWorkbenchClassificationTests(unittest.TestCase):
    def test_text_selection_is_removed_before_visual_object_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for selected_types in (["text"], ["table", "text"]):
                with (
                    patch.object(custom_logic, "client", object()),
                    patch.object(custom_logic, "MODEL_NAME", "test-deployment"),
                    patch.object(
                        custom_logic,
                        "extract_objects_from_pdf",
                        return_value=[],
                    ) as extract,
                ):
                    custom_logic.process_pdf_files(
                        [Path(temp_dir) / "sample.pdf"],
                        str(Path(temp_dir) / "output"),
                        text_documents=[],
                        object_types=selected_types,
                    )

                self.assertEqual(
                    extract.call_args.args[5],
                    set(selected_types) - {"text"},
                )

    def test_table_audit_can_correct_a_false_table_classification(self) -> None:
        table_response = {
            "classification": "table",
            "classification_confidence": "medium",
            "classification_reason": "The figure has aligned labels.",
            "characteristics": {
                "description": "A figure with labeled elements.",
                "content_summary": "A process diagram.",
                "confidence": "medium",
                "table": {
                    "row_enumeration": ["row 1: Start", "row 2: End"],
                    "column_enumeration": ["column 1: Start", "column 2: End"],
                    "num_rows": 2,
                    "num_columns": 2,
                    "num_header_rows": 1,
                    "num_header_columns": 0,
                    "num_data_rows": 1,
                    "num_data_columns": 2,
                },
            },
        }
        image_response = {
            "classification": "image",
            "classification_confidence": "high",
            "classification_reason": "Labeled boxes are connected by arrows.",
            "characteristics": {
                "description": "A process diagram.",
                "content_summary": "A flow from start to end.",
                "confidence": "high",
                "image": {
                    "image_subtype": "flow_diagram",
                    "diagram_components": ["Start", "End"],
                    "diagram_connections": ["Start points to End"],
                },
            },
        }

        with patch.object(
            custom_logic,
            "_call_analysis_json",
            side_effect=[table_response, image_response],
        ):
            result = custom_logic._run_analysis(
                "Analyze the target.",
                "unused.png",
                "table",
                custom_logic.CROP_MAX_TOKENS,
            )

        self.assertEqual(result["classification"], "image")
        self.assertEqual(
            result["characteristics"]["image_subtype"],
            "flow_diagram",
        )

    def test_table_audit_can_correct_a_horizontal_bar_chart_classification(self) -> None:
        table_response = {
            "classification": "table",
            "classification_confidence": "medium",
            "classification_reason": "Category and value labels are aligned.",
            "characteristics": {
                "confidence": "medium",
                "table": {
                    "row_enumeration": ["row 1: AWS", "row 2: Azure"],
                    "column_enumeration": ["column 1: Provider", "column 2: Share"],
                    "num_rows": 2,
                    "num_columns": 2,
                    "num_header_rows": 1,
                    "num_header_columns": 0,
                    "num_data_rows": 1,
                    "num_data_columns": 2,
                },
            },
        }
        graph_response = {
            "classification": "graph",
            "classification_confidence": "high",
            "classification_reason": "Bar lengths encode the market-share values.",
            "characteristics": {
                "confidence": "high",
                "graph": {
                    "chart_type": "horizontal_bar",
                    "categories": ["Amazon Web Services", "Microsoft Azure"],
                    "key_values": ["Amazon Web Services: 30%", "Microsoft Azure: 22%"],
                },
            },
        }

        with patch.object(
            custom_logic,
            "_call_analysis_json",
            side_effect=[table_response, graph_response],
        ):
            result = custom_logic._run_analysis(
                custom_logic.MODE1_PROMPT,
                "unused.png",
                "table",
                custom_logic.CROP_MAX_TOKENS,
            )

        self.assertEqual(result["classification"], "graph")
        self.assertEqual(result["characteristics"]["chart_type"], "horizontal_bar")

    def test_classification_rules_distinguish_tables_from_diagrams_and_charts(self) -> None:
        rules = custom_logic.CLASSIFICATION_RULES.casefold()

        self.assertIn("actual tabular data arranged in aligned rows", rules)
        self.assertIn("if boxes are connected by arrows", rules)
        self.assertIn("quantitative plotted marks", rules)
        self.assertIn("bars of", rules)
        self.assertIn("bar lengths encode the values", rules)
        self.assertIn("ambiguous", rules)

    def test_table_verifier_distinguishes_bar_charts_from_aligned_table_rows(self) -> None:
        prompt = " ".join(custom_logic.VERIFY_PROMPT.casefold().split())

        self.assertIn("differently sized horizontal bars", prompt)
        self.assertIn("bar lengths encode the values", prompt)
        self.assertIn("labels aligned with each bar do not create data cells", prompt)

    def test_table_counting_checks_are_conditional_on_table_classification(self) -> None:
        checks = custom_logic.CROP_FINAL_CHECKS.casefold()

        self.assertIn('only when classification is "table"', checks)

    def test_table_counting_rules_are_explicitly_conditional(self) -> None:
        prompt = custom_logic.MODE1_PROMPT.casefold()
        self.assertIn("only if the target has", prompt)
        self.assertIn("classified as a genuine data table", prompt)


if __name__ == "__main__":
    unittest.main()
