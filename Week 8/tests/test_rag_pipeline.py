import unittest
from pathlib import Path

from rag_pipeline import (
    Document,
    HashEmbeddingModel,
    RAGPipeline,
    SQLiteVectorStore,
    chunk_documents,
)


class ChunkingTests(unittest.TestCase):
    def test_chunks_overlap_and_keep_metadata(self):
        document = Document("one two three four five six", "test.txt", page=2)
        chunks = chunk_documents([document], chunk_size=4, overlap=2)
        self.assertEqual([chunk.text for chunk in chunks], ["one two three four", "three four five six"])
        self.assertEqual(chunks[0].source, "test.txt")
        self.assertEqual(chunks[0].page, 2)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_size=10, overlap=10)


class PipelineTests(unittest.TestCase):
    fixtures = Path(__file__).resolve().parent / "fixtures"
    database = Path(__file__).resolve().parent / "test_vectors.db"

    def tearDown(self):
        self.database.unlink(missing_ok=True)

    def test_retrieval_returns_relevant_document(self):
        with SQLiteVectorStore(self.database) as store:
            pipeline = RAGPipeline(store, HashEmbeddingModel(256))
            documents, chunks = pipeline.ingest(self.fixtures, chunk_size=30, overlap=5)
            answer, results = pipeline.ask(
                "How many annual leave days do employees receive?",
                top_k=1,
                min_score=0.0,
            )

            self.assertEqual(documents, 2)
            self.assertEqual(chunks, 2)
            self.assertEqual(results[0].chunk.source, "leave.txt")
            self.assertIn("twenty days", answer)

    def test_empty_query_is_rejected(self):
        with SQLiteVectorStore(self.database) as store:
            with self.assertRaises(ValueError):
                RAGPipeline(store).ask("   ")


if __name__ == "__main__":
    unittest.main()
