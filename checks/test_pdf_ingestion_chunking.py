import sys
import types
import unittest
from unittest.mock import Mock, patch

from services import pdf_ingestion


class PdfIngestionChunkingTests(unittest.TestCase):
    def test_embedding_service_uses_qwen_process_with_mixed_inputs(self):
        import services.local_embeddings as local_embeddings

        events = []

        class FakeQwenEmbedder:
            def __init__(self):
                self.inputs = None
                self.normalize = None

            def process(self, inputs, *, normalize):
                import numpy as np

                events.append("embedding")
                self.inputs = inputs
                self.normalize = normalize
                return np.ones(
                    (len(inputs), local_embeddings.PROJECT_EMBEDDING_DIMENSION)
                )

        model = FakeQwenEmbedder()

        def load_model():
            events.append("model_loaded")
            return model

        with patch.object(
            local_embeddings,
            "get_project_embedding_function",
            side_effect=load_model,
        ):
            vectors = local_embeddings.embed_records([
                {"text": "Policy wording"},
                {"image": "chart.png", "text": "Revenue chart"},
            ], model_loaded_callback=lambda: events.append("progress_updated"))
            record_inputs = model.inputs
            query_vector = local_embeddings.embed_text_query("find a chart")

        self.assertEqual(
            events[:3],
            ["model_loaded", "progress_updated", "embedding"],
        )
        self.assertEqual(vectors.shape, (2, 2048))
        self.assertEqual(str(vectors.dtype), "float32")
        self.assertTrue(model.normalize)
        self.assertEqual(record_inputs[1]["image"], "chart.png")
        self.assertEqual(query_vector.shape, (2048,))
        self.assertEqual(model.inputs[0]["instruction"], local_embeddings.QUERY_INSTRUCTION)

    def test_heading_aware_is_no_longer_supported(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            pdf_ingestion._chunk_prose(
                "Some narrative.",
                technique="heading_aware",
                size=512,
                overlap=0,
                min_sentences_per_chunk=1,
                semantic_threshold=0.8,
            )

    def test_selected_chonkie_techniques_receive_their_options(self):
        for technique in ("recursive", "sentence", "token", "semantic"):
            with self.subTest(technique=technique):
                captured = {}

                class FakeChunker:
                    def __init__(self, **kwargs):
                        captured.update(kwargs)

                    def chunk(self, text):
                        return [
                            types.SimpleNamespace(
                                start_index=0,
                                end_index=len(text),
                                text=text,
                            )
                        ]

                chonkie = types.ModuleType("chonkie")
                for name in (
                    "RecursiveChunker",
                    "SentenceChunker",
                    "SemanticChunker",
                    "TokenChunker",
                ):
                    setattr(chonkie, name, FakeChunker)

                embeddings = types.ModuleType("services.local_embeddings")
                embeddings.get_project_tokenizer = lambda: "project-tokenizer"
                embeddings.count_embedding_tokens = lambda _text: 10
                custom_logic = types.ModuleType("pdf_workbench.custom_logic")
                custom_logic._get_chonkie_semantic_embedding_model = lambda: "embedding-model"
                modules = {
                    "chonkie": chonkie,
                    "services.local_embeddings": embeddings,
                    "pdf_workbench.custom_logic": custom_logic,
                }

                with patch.dict(sys.modules, modules):
                    result = pdf_ingestion._chunk_prose(
                        "A narrative sentence.",
                        technique=technique,
                        size=128,
                        overlap=16,
                        min_sentences_per_chunk=3,
                        semantic_threshold=0.65,
                    )

                self.assertEqual(result, ["A narrative sentence."])
                self.assertEqual(captured["chunk_size"], 128)
                if technique == "recursive":
                    self.assertEqual(captured["tokenizer"], "project-tokenizer")
                elif technique == "sentence":
                    self.assertEqual(captured["chunk_overlap"], 16)
                    self.assertEqual(captured["min_sentences_per_chunk"], 3)
                elif technique == "token":
                    self.assertEqual(captured["chunk_overlap"], 16)
                else:
                    self.assertEqual(captured["threshold"], 0.65)
                    self.assertEqual(captured["embedding_model"], "embedding-model")
                    self.assertEqual(captured["min_sentences_per_chunk"], 3)

    def test_oversized_chunks_use_token_based_embedding_budget_split(self):
        calls = []
        chunk_calls = 0

        class FakeChunker:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def chunk(self, text):
                nonlocal chunk_calls
                chunk_calls += 1
                if chunk_calls == 1:
                    return [
                        types.SimpleNamespace(
                            start_index=0,
                            end_index=len(text),
                            text=text,
                        )
                    ]
                return [
                    types.SimpleNamespace(
                        start_index=0,
                        end_index=len("safe chunk"),
                        text="safe chunk",
                    )
                ]

        chonkie = types.ModuleType("chonkie")
        chonkie.TokenChunker = FakeChunker
        chonkie.RecursiveChunker = FakeChunker
        chonkie.SentenceChunker = FakeChunker
        chonkie.SemanticChunker = FakeChunker
        embeddings = types.ModuleType("services.local_embeddings")
        embeddings.get_project_tokenizer = lambda: "project-tokenizer"
        embeddings.count_embedding_tokens = lambda text: (
            241 if text == "Long narrative." else 10
        )

        with patch.dict(
            sys.modules,
            {
                "chonkie": chonkie,
                "services.local_embeddings": embeddings,
            },
        ):
            result = pdf_ingestion._chunk_prose(
                "Long narrative.",
                technique="token",
                size=128,
                overlap=16,
                min_sentences_per_chunk=1,
                semantic_threshold=0.8,
            )

        self.assertEqual(result, ["safe chunk"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["chunk_size"], 128)
        self.assertEqual(calls[1]["chunk_overlap"], 16)

    def test_all_chonkie_techniques_respect_the_embedding_content_limit(self):
        for technique in ("recursive", "sentence", "token", "semantic"):
            with self.subTest(technique=technique):
                captured = {}

                class FakeChunker:
                    def __init__(self, **kwargs):
                        captured.update(kwargs)

                    def chunk(self, text):
                        return [
                            types.SimpleNamespace(
                                start_index=0,
                                end_index=len(text),
                                text=text,
                            )
                        ]

                chonkie = types.ModuleType("chonkie")
                for name in (
                    "RecursiveChunker",
                    "SentenceChunker",
                    "SemanticChunker",
                    "TokenChunker",
                ):
                    setattr(chonkie, name, FakeChunker)
                embeddings = types.ModuleType("services.local_embeddings")
                embeddings.get_project_tokenizer = lambda: "project-tokenizer"
                embeddings.count_embedding_tokens = lambda _text: 10
                custom_logic = types.ModuleType("pdf_workbench.custom_logic")
                custom_logic._get_chonkie_semantic_embedding_model = lambda: "embedding-model"

                with patch.dict(
                    sys.modules,
                    {
                        "chonkie": chonkie,
                        "services.local_embeddings": embeddings,
                        "pdf_workbench.custom_logic": custom_logic,
                    },
                ):
                    pdf_ingestion._chunk_prose(
                        "A narrative sentence.",
                        technique=technique,
                        size=512,
                        overlap=16,
                        min_sentences_per_chunk=3,
                        semantic_threshold=0.65,
                    )

                self.assertEqual(
                    captured["chunk_size"],
                    pdf_ingestion.MAX_CHUNK_CONTENT_TOKENS,
                )

    def test_overlap_must_fit_the_effective_embedding_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            pdf_ingestion._chunk_prose(
                "A narrative sentence.",
                technique="token",
                size=512,
                overlap=pdf_ingestion.MAX_CHUNK_CONTENT_TOKENS,
                min_sentences_per_chunk=1,
                semantic_threshold=0.8,
            )

    def test_structure_chunks_passes_settings_to_narrative_chunker(self):
        settings = {
            "technique": "sentence",
            "chunk_size": 512,
            "overlap": 50,
            "min_sentences_per_chunk": 2,
            "semantic_threshold": 0.8,
        }
        with patch(
            "services.pdf_ingestion._chunk_prose",
            return_value=["selected chunk"],
        ) as chunk_prose:
            chunks = pdf_ingestion._structure_chunks(
                [(2, "text", "Narrative text.")],
                settings,
            )

        self.assertEqual(chunks, [(2, "text", "selected chunk", "")])
        self.assertEqual(chunk_prose.call_args.kwargs["technique"], "sentence")
        self.assertEqual(chunk_prose.call_args.kwargs["size"], 512)
        self.assertEqual(chunk_prose.call_args.kwargs["min_sentences_per_chunk"], 2)

    def test_pdf_ingestion_uses_workbench_settings_for_prose_and_table_chunks(self):
        settings = {
            "technique": "token",
            "chunk_size": 1024,
            "overlap": 32,
            "min_sentences_per_chunk": 1,
            "semantic_threshold": 0.8,
        }
        table_markdown = "| Year | Revenue |\n| --- | --- |\n| 2025 | $12M |"
        table = types.SimpleNamespace(
            prov=[
                types.SimpleNamespace(page_no=1),
                types.SimpleNamespace(page_no=2),
            ],
            export_to_markdown=lambda *, doc: table_markdown,
        )
        document = types.SimpleNamespace(
            pages={1: object()},
            tables=[table],
            export_to_markdown=lambda *, page_no: "# Overview\n\nProse.",
        )
        custom_logic = types.ModuleType("pdf_workbench.custom_logic")
        custom_logic.chunk_document_text = Mock(return_value=[{
            "chunk_id": "doc:page-1:text:1",
            "source_text": "Prose.",
            "section": "Overview",
        }])
        custom_logic.chunk_text_by_technique = Mock(return_value=[
            (0, len(table_markdown), table_markdown),
        ])

        with patch.dict(
            sys.modules,
            {"pdf_workbench.custom_logic": custom_logic},
        ):
            records = pdf_ingestion._docling_text_records(
                document,
                "sample.pdf",
                "doc",
                settings,
            )

        custom_logic.chunk_document_text.assert_called_once_with(
            "# Overview\n\nProse.",
            "sample.pdf",
            "doc:page-1",
            **settings,
        )
        custom_logic.chunk_text_by_technique.assert_called_once_with(
            table_markdown,
            **settings,
        )
        self.assertEqual([record["object_type"] for record in records], ["text", "table"])
        self.assertEqual(records[0]["text"], "Prose.")
        self.assertEqual(records[1]["text"], table_markdown)
        self.assertEqual(records[1]["source_pages"], "[1, 2]")


if __name__ == "__main__":
    unittest.main()
