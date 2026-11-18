import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from services import pdf_ingestion


class PdfImageIngestionTests(unittest.TestCase):
    def test_docling_picture_records_save_images_without_text(self):
        class FakeImage:
            def convert(self, _mode):
                return self

            def save(self, path, **_kwargs):
                Path(path).write_bytes(b"png image")

        class FakePictureItem:
            prov = [types.SimpleNamespace(page_no=2)]

            def get_image(self, _document):
                return FakeImage()

        item = FakePictureItem()

        class FakeDocument:
            def iterate_items(self, *, traverse_pictures):
                self.traverse_pictures = traverse_pictures
                yield item, 0

        docling = types.ModuleType("docling")
        docling.__path__ = []
        datamodel = types.ModuleType("docling.datamodel")
        datamodel.__path__ = []
        document_module = types.ModuleType("docling.datamodel.document")
        document_module.PictureItem = FakePictureItem

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules,
            {
                "docling": docling,
                "docling.datamodel": datamodel,
                "docling.datamodel.document": document_module,
            },
        ):
            records = pdf_ingestion._docling_image_records(
                FakeDocument(),
                Path(temp_dir),
                "a" * 64,
            )
            self.assertEqual(len(records), 1)
            page_number, image_path = records[0]
            self.assertEqual(page_number, 2)
            self.assertTrue(Path(image_path).is_file())

    def _run_ingestion(self, image_records, text_records=None, chunking=None):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        projects_dir = Path(temporary_directory.name)
        project_dir = projects_dir / "project"
        project_dir.mkdir()
        (project_dir / "config.json").write_text("{}", encoding="utf-8")
        image_path = project_dir / "objects" / "figure.png"
        image_path.parent.mkdir()
        image_path.write_bytes(b"png image")

        class FakeCollection:
            name = "project_collection"

            def __init__(self):
                self.upsert_call = None
                self.upsert_calls = []

            def get(self, **_kwargs):
                return {"ids": [], "metadatas": []}

            def upsert(self, **kwargs):
                self.upsert_call = kwargs
                self.upsert_calls.append(kwargs)

            def update(self, **_kwargs):
                pass

        collection = FakeCollection()
        conversion = types.SimpleNamespace(
            status=types.SimpleNamespace(value="success"),
            document=types.SimpleNamespace(
                name="sample",
                origin=types.SimpleNamespace(
                    mimetype="application/pdf",
                    binary_hash="source-hash",
                ),
                pages={1: object()},
            ),
        )

        class FakeConverter:
            def convert(self, _path):
                return conversion

        custom_logic = types.ModuleType("pdf_workbench.custom_logic")
        custom_logic.get_converter = lambda: FakeConverter()
        project_chroma = types.ModuleType("services.project_chroma")
        project_chroma.open_project_collection = lambda *_args, **_kwargs: collection
        local_embeddings = types.ModuleType("services.local_embeddings")
        local_embeddings.DOCUMENT_INSTRUCTION = "Represent the user's input."
        captured_embedding_inputs = []

        def embed_records(inputs, **_kwargs):
            captured_embedding_inputs.extend(inputs)
            return [[0.25] * 2048 for _ in inputs]

        local_embeddings.embed_records = embed_records

        with (
            patch.object(pdf_ingestion, "PROJECTS_DIR", projects_dir),
            patch.object(
                pdf_ingestion,
                "_docling_image_records",
                return_value=image_records(image_path),
            ) as get_image_records,
            patch.object(
                pdf_ingestion,
                "_docling_text_records",
                return_value=text_records or [],
            ) as get_text_records,
            patch.dict(
                sys.modules,
                {
                    "pdf_workbench.custom_logic": custom_logic,
                    "services.project_chroma": project_chroma,
                    "services.local_embeddings": local_embeddings,
                },
            ),
        ):
            result = pdf_ingestion.ingest_pdf(
                "project",
                "sample.pdf",
                b"%PDF sample",
                chunking=chunking,
            )

        return (
            result,
            collection,
            get_image_records,
            captured_embedding_inputs,
            get_text_records,
        )

    def test_ingestion_embeds_only_images_and_stores_image_metadata(self):
        result, collection, get_image_records, embedding_inputs, get_text_records = self._run_ingestion(
            lambda image_path: [(2, image_path)]
        )

        self.assertEqual(result["image_records_ingested"], 1)
        self.assertEqual(result["chunks_ingested"], 1)
        self.assertEqual(result["text_records_ingested"], 0)
        self.assertEqual(result["table_records_ingested"], 0)
        get_text_records.assert_called_once()
        self.assertEqual(get_image_records.call_args.args[0].pages.keys(), {1})
        upsert = collection.upsert_call
        self.assertNotIn("documents", upsert)
        metadata = upsert["metadatas"][0]
        self.assertEqual(
            embedding_inputs,
            [{
                "instruction": "Represent the user's input.",
                "image": metadata["image_path"],
            }],
        )
        self.assertEqual(metadata["modality"], "image")
        self.assertEqual(metadata["object_type"], "image")
        self.assertTrue(Path(metadata["image_path"]).is_file())
        self.assertNotIn("section_path", metadata)

    def test_ingestion_embeds_prose_and_tables_as_text_records(self):
        prose = {
            "modality": "text",
            "object_type": "text",
            "page_number": 1,
            "source_pages": "[1]",
            "chunk_id": "doc:page-1:text:1",
            "section_path": "Overview",
            "text": "The project launched in 2025.",
        }
        table = {
            "modality": "text",
            "object_type": "table",
            "page_number": 2,
            "source_pages": "[2]",
            "chunk_id": "doc:table-1:1",
            "section_path": "",
            "text": "| Year | Revenue |\n| --- | --- |\n| 2025 | $12M |",
        }
        settings = {
            "technique": "token",
            "chunk_size": 1024,
            "overlap": 32,
            "min_sentences_per_chunk": 1,
            "semantic_threshold": 0.8,
        }
        result, collection, _, embedding_inputs, get_text_records = self._run_ingestion(
            lambda image_path: [(2, image_path)],
            text_records=[prose, table],
            chunking=settings,
        )

        self.assertEqual(result["chunks_ingested"], 3)
        self.assertEqual(result["image_records_ingested"], 1)
        self.assertEqual(result["text_records_ingested"], 1)
        self.assertEqual(result["table_records_ingested"], 1)
        self.assertEqual(get_text_records.call_args.args[3], settings)
        text_upserts = [
            call for call in collection.upsert_calls if "documents" in call
        ]
        self.assertEqual(
            [document for call in text_upserts for document in call["documents"]],
            [prose["text"], table["text"]],
        )
        self.assertEqual(len(embedding_inputs), 3)
        self.assertTrue(any("text" in record for record in embedding_inputs))
        self.assertTrue(any("image" in record for record in embedding_inputs))

    def test_ingestion_fails_when_docling_finds_no_records(self):
        with self.assertRaisesRegex(ValueError, "no figures, prose, or tables"):
            self._run_ingestion(lambda _image_path: [])

if __name__ == "__main__":
    unittest.main()
