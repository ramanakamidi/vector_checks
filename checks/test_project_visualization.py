from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from services.project_visualization import build_project_visualization


class ProjectVisualizationTests(unittest.TestCase):
    def test_hdbscan_clusters_the_displayed_umap_coordinates(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        coordinates = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        )
        collection = Mock()
        collection.name = "test_collection"
        collection.count.return_value = len(vectors)
        collection.get.return_value = {
            "ids": ["first", "second", "third"],
            "embeddings": vectors,
            "documents": ["chunk one", "chunk two", "chunk three"],
            "metadatas": [{}, {}, {}],
        }

        hdbscan = Mock()
        thread_counts = []

        def capture_thread_count():
            from threadpoolctl import threadpool_info

            thread_counts.append(
                [pool["num_threads"] for pool in threadpool_info()]
            )

        def fit_predict(_projected_coordinates):
            capture_thread_count()
            return np.asarray([0, 0, -1])

        hdbscan.fit_predict.side_effect = fit_predict

        def create_projector(**_kwargs):
            projector = Mock()

            def fit_transform(_vectors):
                capture_thread_count()
                return coordinates

            projector.fit_transform.side_effect = fit_transform
            return projector

        with (
            patch(
                "services.project_chroma.open_project_collection",
                return_value=collection,
            ),
            patch("umap.UMAP", side_effect=create_projector),
            patch("sklearn.cluster.HDBSCAN", return_value=hdbscan),
        ):
            result = build_project_visualization("project-1")

        self.assertTrue(thread_counts)
        observed_counts = [
            count for counts in thread_counts for count in counts
        ]
        self.assertTrue(observed_counts)
        self.assertEqual(max(observed_counts), 1)
        np.testing.assert_array_equal(
            hdbscan.fit_predict.call_args.args[0],
            coordinates,
        )
        self.assertEqual(result["clustering_space"], "umap_2d")
        self.assertEqual(
            [point["cluster"] for point in result["points"]],
            [0, 0, -1],
        )
        self.assertEqual(
            [(point["x"], point["y"]) for point in result["points"]],
            [(0.0, 0.0), (0.1, 0.1), (10.0, 10.0)],
        )


if __name__ == "__main__":
    unittest.main()
