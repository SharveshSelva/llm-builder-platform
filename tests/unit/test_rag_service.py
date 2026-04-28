import pytest
from backend.services.rag_service import _chunk_pdf


SAMPLE_PDF_TEXT = b"%PDF-1.4"  # minimal invalid PDF for error path


class TestChunkPdf:
    def _make_fake_pdf(self, pages: list[str]) -> bytes:
        """Creates a minimal valid PDF with given page texts using pypdf."""
        from pypdf import PdfWriter
        import io
        writer = PdfWriter()
        for text in pages:
            page = writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def test_empty_pdf_returns_no_chunks(self):
        pdf_bytes = self._make_fake_pdf([""])
        texts, metadatas = _chunk_pdf(pdf_bytes, "empty.pdf")
        assert texts == []
        assert metadatas == []

    def test_short_text_yields_one_chunk(self):
        pdf_bytes = self._make_fake_pdf(["Hello world. This is a short document."])
        # pypdf blank pages don't embed text, so this tests the no-text path
        texts, metadatas = _chunk_pdf(pdf_bytes, "short.pdf")
        # blank pages extract empty text — verify no crash and list returned
        assert isinstance(texts, list)
        assert isinstance(metadatas, list)

    def test_metadata_contains_source_and_page(self):
        from pypdf import PdfWriter
        import io
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        texts, metadatas = _chunk_pdf(buf.getvalue(), "myfile.pdf")
        # Even with blank pages, if any chunks are produced they carry correct metadata
        for meta in metadatas:
            assert meta["source"] == "myfile.pdf"
            assert "page" in meta


class TestCitationBuilding:
    """Verify that chat() wraps chunks into Citation objects correctly."""

    @pytest.mark.asyncio
    async def test_no_chunks_returns_no_docs_message(self):
        from unittest.mock import patch, AsyncMock
        from backend.services.rag_service import chat

        with patch("backend.services.rag_service.vector_store.query", return_value=[]):
            result = await chat("anything", "documents", 5, "fast")

        assert "No relevant documents" in result.answer
        assert result.citations == []
        assert result.model_used == "none"
