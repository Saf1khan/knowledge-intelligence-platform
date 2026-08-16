import pytest
import tempfile
import uuid
from pathlib import Path

from backend.app.services.ingestion.pdf_parser import (
    PDFParser,
    PDFDocument,
    PDFValidationError,
    PDFExtractionError,
    PDFEncryptionError,
)
from backend.app.services.ingestion.chunking import (
    TextChunker,
    ChunkingConfig,
    ChunkingStrategy,
    DocumentChunk,
    ChunkingResult,
    ChunkValidationError,
)


class TestPDFParser:
    def test_parser_initialization(self):
        parser = PDFParser()
        assert parser is not None

    def test_validate_nonexistent_file(self):
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.validate_pdf(Path("non_existent_file.pdf"))

    def test_validate_invalid_extension(self, tmp_path):
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Not a pdf file content signature")
        parser = PDFParser()
        with pytest.raises(PDFValidationError, match="is not a PDF file based on extension"):
            parser.validate_pdf(txt_file)

    def test_validate_empty_file(self, tmp_path):
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.touch()
        parser = PDFParser()
        with pytest.raises(PDFValidationError, match="is empty"):
            parser.validate_pdf(empty_pdf)


class TestTextChunker:
    def setup_method(self):
        self.config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            strategy=ChunkingStrategy.RECURSIVE
        )
        self.chunker = TextChunker(config=self.config)

    def test_chunker_initialization(self):
        assert self.chunker.config.chunk_size == 100
        assert self.chunker.config.chunk_overlap == 20
        assert self.chunker.config.strategy == ChunkingStrategy.RECURSIVE

    def test_invalid_overlap_raises_error(self):
        with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_empty_text_chunking(self):
        doc_id = uuid.uuid4()
        result = self.chunker.chunk_document(doc_id, "")
        assert result.total_chunks == 0
        assert result.chunks == []

    def test_short_text_chunking(self):
        short_text = "This is a short sample sentence for testing text chunking functionality."
        doc_id = uuid.uuid4()
        result = self.chunker.chunk_document(doc_id, short_text)
        
        assert isinstance(result, ChunkingResult)
        assert result.total_chunks == 1
        assert len(result.chunks) == 1
        assert result.chunks[0].text == short_text
        assert result.chunks[0].document_id == doc_id
        assert result.chunks[0].chunk_index == 0

    def test_recursive_chunking(self):
        text = "Paragraph one is here and it has some content. " * 5 + "\n\n" + "Paragraph two is here with more words. " * 5
        doc_id = uuid.uuid4()
        result = self.chunker.chunk_document(doc_id, text)
        
        assert result.total_chunks > 1
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == doc_id
            assert chunk.token_estimate > 0
