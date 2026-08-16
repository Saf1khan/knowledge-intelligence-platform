from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError, FileNotDecryptedError

# Configurable constants
PDF_SIGNATURE: bytes = b"%PDF-"
MIN_EXTRACTED_TEXT_LENGTH: int = 100

# Initialize logger
logger = logging.getLogger("app.services.ingestion.pdf_parser")


class PDFParserError(Exception):
    """Base exception for all PDF parsing and validation errors."""
    pass


class PDFValidationError(PDFParserError):
    """Raised when PDF file validation checks fail."""
    pass


class PDFEncryptionError(PDFParserError):
    """Raised when the PDF file is encrypted and cannot be decrypted."""
    pass


class PDFExtractionError(PDFParserError):
    """Raised when text extraction fails due to format corruption or system error."""
    pass


class PDFDocument(BaseModel):
    """Pydantic model representing the parsed PDF document contents and metrics.
    
    Attributes:
        page_count (int): Total number of pages in the PDF document.
        extracted_character_count (int): Count of characters successfully extracted.
        extraction_duration_ms (float): Execution time of the parsing process in milliseconds.
        text (str): Consolidated and cleaned text extracted from all pages.
        extracted_at (datetime): UTC timestamp recording when the extraction took place.
    """
    page_count: int = Field(
        ..., 
        description="Total number of pages parsed from the PDF document"
    )
    extracted_character_count: int = Field(
        ..., 
        description="The number of characters extracted from the PDF document"
    )
    extraction_duration_ms: float = Field(
        ..., 
        description="Time taken to extract text in milliseconds"
    )
    text: str = Field(
        ..., 
        description="Consolidated and cleaned text extracted from all readable pages"
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp indicating when the text was extracted"
    )


class PDFParser:
    """Enterprise-grade parser to extract textual content and metadata from PDF files.
    
    Adheres to SOLID design principles by separating validation, extraction,
    and document model building into dedicated, testable helper methods.
    """

    def validate_pdf(self, pdf_path: Path) -> None:
        """Validates that a PDF file exists, is non-empty, and has valid header signature.
        
        Args:
            pdf_path (Path): Path to the target PDF file.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            PDFValidationError: If the file has a size of 0, invalid extension,
                or fails the magic bytes signature check.
        """
        # 1. Verify existence
        if not pdf_path.exists() or not pdf_path.is_file():
            logger.error(f"File not found or is invalid: '{pdf_path}'")
            raise FileNotFoundError(f"The PDF file at '{pdf_path}' does not exist.")

        # 2. Check for zero-byte file size
        try:
            if pdf_path.stat().st_size == 0:
                logger.error(f"Validation failed: File '{pdf_path}' is empty (0 bytes).")
                raise PDFValidationError(f"The PDF file '{pdf_path.name}' is empty (0 bytes).")
        except OSError as e:
            logger.error(f"Failed to check stats for file '{pdf_path}': {e}")
            raise PDFValidationError(f"Could not read metadata for file '{pdf_path.name}': {str(e)}")

        # 3. Verify extension
        if pdf_path.suffix.lower() != ".pdf":
            logger.error(f"Invalid file extension: '{pdf_path.suffix}' for file '{pdf_path}'")
            raise PDFValidationError(f"The file '{pdf_path.name}' is not a PDF file based on extension.")

        # 4. Check Magic Bytes Signature
        try:
            with pdf_path.open("rb") as f:
                header = f.read(len(PDF_SIGNATURE))
                if header != PDF_SIGNATURE:
                    logger.error(f"File signature mismatch for '{pdf_path}'. Found: {header}")
                    raise PDFValidationError(f"The file '{pdf_path.name}' is not a valid PDF (invalid signature).")
        except OSError as e:
            logger.error(f"Failed to read file signature for '{pdf_path}': {e}", exc_info=True)
            raise PDFValidationError(f"Unable to read file signature for '{pdf_path.name}': {str(e)}")

    def extract_page_text(self, reader: PdfReader, page_num: int) -> str:
        """Extracts and sanitizes text from a specific PDF page.
        
        Args:
            reader (PdfReader): Instantiated PdfReader instance.
            page_num (int): 0-indexed page number to extract.
            
        Returns:
            str: The extracted and trimmed text. Returns an empty string if page has no text.
            
        Raises:
            PDFExtractionError: If page parsing fails.
        """
        try:
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                return text.strip()
        except Exception as page_err:
            logger.warning(f"Failed to extract text from page {page_num + 1}: {page_err}")
            raise PDFExtractionError(f"Error parsing page {page_num + 1}: {str(page_err)}")
        return ""

    def build_document(self, page_count: int, text: str, duration_ms: float) -> PDFDocument:
        """Constructs a PDFDocument Pydantic model with calculated metrics.
        
        Args:
            page_count (int): Total number of pages parsed.
            text (str): Consolidated extracted text.
            duration_ms (float): Total processing time in milliseconds.
            
        Returns:
            PDFDocument: The populated document model container.
        """
        return PDFDocument(
            page_count=page_count,
            extracted_character_count=len(text),
            extraction_duration_ms=round(duration_ms, 2),
            text=text,
            extracted_at=datetime.now(timezone.utc)
        )

    def extract_text(self, pdf_path: Path) -> PDFDocument:
        """Extracts text content and metrics from a PDF file.
        
        Performs validation, decryption handling, sequential page extraction,
        failure threshold checks, and text length warning checks.
        
        Args:
            pdf_path (Path): Path to the target PDF file on disk.
            
        Returns:
            PDFDocument: Structured representation containing extracted text and metrics.
            
        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            PDFValidationError: If PDF validation fails.
            PDFEncryptionError: If the file is encrypted and cannot be decrypted.
            PDFExtractionError: If no text can be extracted or format is corrupted.
        """
        start_time = time.perf_counter()
        logger.info(f"Starting text extraction for PDF: '{pdf_path}'")

        # 1. Validate PDF file
        self.validate_pdf(pdf_path)

        try:
            # 2. Load PDF document
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)

            # 3. Handle Encryption
            if reader.is_encrypted:
                logger.warning(f"PDF file '{pdf_path}' is encrypted. Attempting empty password decryption.")
                try:
                    reader.decrypt("")
                except (PdfReadError, FileNotDecryptedError) as decrypt_err:
                    logger.error(f"Failed to decrypt PDF '{pdf_path}': {decrypt_err}")
                    raise PDFEncryptionError(f"The PDF file '{pdf_path.name}' is encrypted and cannot be parsed.")

            page_texts = []
            failed_pages = 0

            # 4. Extract text sequentially to optimize memory footprint
            for page_num in range(total_pages):
                try:
                    page_text = self.extract_page_text(reader, page_num)
                    if page_text:
                        page_texts.append(page_text)
                    else:
                        logger.debug(f"Skipping empty or non-textual page {page_num + 1} of '{pdf_path.name}'.")
                except PDFExtractionError:
                    failed_pages += 1

            # 5. Check page failure rate warning threshold (more than 20% failed)
            if total_pages > 0 and (failed_pages / total_pages) > 0.20:
                logger.warning(
                    f"High page extraction failure rate on '{pdf_path.name}': {failed_pages}/{total_pages} "
                    f"pages ({failed_pages / total_pages * 100:.1f}%) failed to parse."
                )

            # 6. Consolidate extracted text
            full_text = "\n\n".join(page_texts)
            char_count = len(full_text)

            # 7. Validate that we extracted at least some text (reject scan-only/image-only PDFs without text)
            if char_count == 0:
                logger.error(f"Extraction failed: No text could be extracted from '{pdf_path.name}'.")
                raise PDFExtractionError(
                    f"The PDF document '{pdf_path.name}' contains no extractable text. It may be image-only."
                )

            # 8. Check for suspiciously small text content warning
            if char_count < MIN_EXTRACTED_TEXT_LENGTH:
                logger.warning(
                    f"Suspiciously short extracted text for '{pdf_path.name}': {char_count} characters "
                    f"(minimum expected: {MIN_EXTRACTED_TEXT_LENGTH})."
                )

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            logger.info(
                f"Successfully completed extraction for '{pdf_path.name}' in {duration_ms:.2f}ms. "
                f"Pages: {total_pages}, Characters: {char_count}."
            )

            # 9. Return populated document model
            return self.build_document(total_pages, full_text, duration_ms)

        except (PdfReadError, ValueError) as pypdf_err:
            logger.error(f"Corrupted or invalid PDF format structure for '{pdf_path}': {pypdf_err}", exc_info=True)
            raise PDFExtractionError(f"The PDF file '{pdf_path.name}' is corrupted or invalid: {pypdf_err}")
        except (PDFValidationError, PDFEncryptionError, PDFExtractionError):
            # Pass custom exceptions directly
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing PDF '{pdf_path}': {e}", exc_info=True)
            raise PDFExtractionError(f"An unexpected error occurred while parsing the PDF: {e}")
