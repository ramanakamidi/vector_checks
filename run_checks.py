"""
Main entry point. Pick a connector, run the checks that make sense for
your setup, get a plain report back.

Example (Elasticsearch):

    from connectors.elastic_connector import ElasticConnector
    from run_checks import run_full_report

    connector = ElasticConnector(
        hosts=["https://your-cluster:9200"],
        index="orion-chunks",
        vector_field="embedding",
        api_key="...",
    )
    connector.connect()
    report = run_full_report(connector, required_metadata_fields=["doc_id", "artifact_type"])
    print_report(report)

Swap ElasticConnector for PineconeConnector / WeaviateConnector /
MilvusConnector / PgVectorConnector / ChromaConnector -- everything
below this line does not change.
"""

import json
import traceback
from typing import Dict, List, Optional, Tuple

from connectors.base import VectorDBConnector
from checks import vector_integrity, metadata_completeness, duplicate_detection
from checks import neighbor_consistency


def safe_connect(connector: VectorDBConnector) -> Tuple[bool, Optional[str]]:
    """Attempt to connect with clean error reporting instead of raw exceptions.

    Returns:
        (success, error_message) — if success is False, error_message
        explains what went wrong.
    """
    try:
        connector.connect()
        return True, None
    except ImportError as e:
        return False, (
            f"Missing client library: {e}. "
            f"Install the appropriate package for your database "
            f"(see README.md for pip install commands)."
        )
    except ConnectionError as e:
        return False, f"Connection failed: {e}"
    except Exception as e:
        return False, (
            f"Unexpected error during connect(): {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )


def run_full_report(connector: VectorDBConnector,
                     required_metadata_fields: Optional[List[str]] = None,
                     batch_size: int = 500,
                     sample_size: int = 500,
                     field_mapper=None) -> Dict:
    """Run the checks that can be performed without user-curated examples.

    Args:
        connector: A connected VectorDBConnector instance.
        required_metadata_fields: List of metadata field names that every
                                   record should have (for Category 4).
        batch_size: Batch size for scanning operations.
        sample_size: Sample size for duplicate detection.
        field_mapper: Optional FieldMapper instance (from config.py).
            When provided, failure records include display metadata
            alongside their IDs for richer report drilldowns.
    """
    results = []

    # Category 1: Vector Integrity
    try:
        results.extend(vector_integrity.run_all(
            connector, batch_size=batch_size, field_mapper=field_mapper))
    except Exception as e:
        results.append(_error_result("vector_integrity", e))

    # Category 3: Duplicate Detection
    try:
        results.append(duplicate_detection.find_near_duplicates(
            connector, sample_size=sample_size, field_mapper=field_mapper))
    except Exception as e:
        results.append(_error_result("duplicate_detection", e))

    # Category 4: Metadata Completeness
    if required_metadata_fields:
        try:
            results.append(metadata_completeness.check_required_fields(
                connector, required_metadata_fields, batch_size=batch_size,
                field_mapper=field_mapper))
        except Exception as e:
            results.append(_error_result("metadata_completeness", e))

    # Category 2: Generate likely-similar examples from adjacent chunks.
    try:
        pairs = neighbor_consistency.generate_adjacent_chunk_pairs(connector)
        neighbor_result = neighbor_consistency.check_known_similar_pairs(
            connector, pairs, field_mapper=field_mapper)
        neighbor_result["passed"] = None
        detail = neighbor_result.get("detail")
        if isinstance(detail, dict):
            detail.pop("pairs_failed", None)
            detail["pairs_not_mutual_top_k"] = sum(
                item.get("status") in {"one_way", "neither"}
                for item in detail.get("pair_results", [])
            )
            ranking_misses = {
                tuple(item.get("pair", ()))
                for item in detail.get("pair_results", [])
                if item.get("status") in {"one_way", "neither"}
            }
            detail["pairs_outside_top_k"] = [
                failure
                for failure in detail.pop("failures", [])
                if tuple(failure.get("pair", ())) in ranking_misses
            ]
            detail["unavailable_pairs"] = detail.pop("unavailable_pairs", [])
            detail.update({
                "pair_source": "automatic adjacent chunks from the same document",
                "pairs_sampled": len(pairs),
                "methodology_note": (
                    "This is an informational diagnostic, not a pass/fail quality "
                    "score. Pairs are consecutive chunks from the same source, "
                    "filtered to the same section and page when available, excluding "
                    "reference sections and incompatible content types. Every "
                    "sampled pair is listed; metadata and overlap still cannot prove "
                    "semantic similarity."
                ),
            })
        else:
            neighbor_result["detail"] = {
                "message": detail,
                "pair_source": "automatic adjacent chunks from the same document",
                "pairs_sampled": 0,
                "methodology_note": (
                    "No adjacent chunks with usable source and chunk-index "
                    "metadata were available. This check is inconclusive, not a pass."
                ),
            }
        results.append(neighbor_result)
    except Exception as e:
        results.append(_error_result("known_similar_pairs", e))

    return {"results": results, "summary": _summarize(results)}


def _error_result(check_name: str, exception: Exception) -> Dict:
    """Wrap an exception into a standard result dict instead of crashing."""
    return {
        "check": check_name,
        "passed": None,
        "detail": {
            "error": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(),
            "note": "This check failed to run — see error above.",
        },
    }


def _summarize(results: List[Dict]) -> Dict:
    passed = sum(1 for r in results if r.get("passed") is True)
    failed = sum(1 for r in results if r.get("passed") is False)
    inconclusive = sum(1 for r in results if r.get("passed") is None)
    return {"passed": passed, "failed": failed, "inconclusive": inconclusive, "total": len(results)}


def print_report(report: Dict) -> None:
    """Print a human-readable report to stdout."""
    print("=" * 70)
    print("  VECTOR DATABASE HEALTH REPORT")
    print("=" * 70)
    for result in report["results"]:
        status = {"True": "[PASS]", "False": "[FAIL]", "None": "[INFO]"}[str(result["passed"])]
        print(f"\n{status}  {result['check']}")
        print("-" * 60)
        detail = result.get("detail", {})
        if isinstance(detail, str):
            print(f"  {detail}")
        else:
            print(json.dumps(detail, indent=2, default=str)[:3000])
    print("\n" + "=" * 70)
    s = report["summary"]
    print(f"  Summary: {s['passed']} passed, {s['failed']} failed, "
          f"{s['inconclusive']} inconclusive, {s['total']} total")
    print("=" * 70)


def export_report_json(report: Dict, output_path: str = "report.json") -> None:
    """Export the full report as a JSON file for programmatic consumption."""
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report exported to {output_path}")
