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
        hdbscan.fit_predict.return_value = np.asarray([0, 0, -1])
        with (
            patch(
                "services.project_chroma.open_project_collection",
                return_value=collection,
            ),
            patch("umap.UMAP") as umap,
            patch("sklearn.cluster.HDBSCAN", return_value=hdbscan),
        ):
            umap.return_value.fit_transform.return_value = coordinates
            result = build_project_visualization("project-1")

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
