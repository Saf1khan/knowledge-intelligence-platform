from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import shutil
import uuid
from typing import Set

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Constants for validation and configuration
MAX_FILE_SIZE_MB: int = 50
ALLOWED_CONTENT_TYPES: Set[str] = {"application/pdf"}
PDF_SIGNATURE: bytes = b"%PDF-"

# Initialize router and logger
router = APIRouter()
logger = logging.getLogger("app.api.upload")

# Resolve the absolute path to the backend/uploads directory using pathlib
BASE_DIR: Path = Path(__file__).resolve().parents[2]
UPLOAD_DIR: Path = BASE_DIR / "uploads"


class UploadResponse(BaseModel):
    """Pydantic response model for the file upload endpoint.
    
    Attributes:
        document_id (uuid.UUID): Unique UUID generated for the uploaded document.
        filename (str): The original name of the uploaded file.
        stored_filename (str): The filename under which the file was stored.
        status (str): The current status of the upload action (defaults to "uploaded").
        upload_timestamp (str): ISO 8601 formatted timestamp of the upload time.
    """
    document_id: uuid.UUID = Field(
        ..., 
        description="Unique UUID generated for the uploaded document"
    )
    filename: str = Field(
        ..., 
        description="The original name of the uploaded file"
    )
    stored_filename: str = Field(
        ..., 
        description="The filename under which the file was stored on disk to prevent collisions"
    )
    status: str = Field(
        "uploaded", 
        description="The current status of the upload action"
    )
    upload_timestamp: str = Field(
        ..., 
        description="ISO 8601 formatted timestamp of the upload time"
    )


def sanitize_filename(filename: str) -> str:
    """Sanitizes the filename to prevent directory traversal and remove unsafe characters.
    
    Args:
        filename (str): The original filename.
        
    Returns:
        str: A sanitized, path-safe filename.
    """
    # Extract filename to prevent directory traversal attacks
    name = Path(filename).name
    
    # Replace spaces with underscores
    name = re.sub(r"\s+", "_", name)
    
    # Remove any characters that are not alphanumeric, dot, underscore, or hyphen
    name = re.sub(r"[^\w\-\.]", "", name)
    
    # Fallback if sanitization results in an empty string
    if not name or name.startswith('.'):
        name = f"document_{uuid.uuid4().hex[:8]}.pdf"
        
    return name


def validate_file_size(file: UploadFile) -> None:
    """Validates that the file is not empty and does not exceed the maximum size.
    
    Args:
        file (UploadFile): The uploaded file stream.
        
    Raises:
        HTTPException: If the file is empty or exceeds the configured limit.
    """
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    file_size = file.size
    filename = file.filename or "unknown"

    # Fallback: if size attribute is not populated, calculate size by seeking
    if file_size is None:
        try:
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)
        except OSError as e:
            logger.error(f"Failed to determine file size for '{filename}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not verify file size due to an internal error."
            )

    if file_size == 0:
        logger.warning(f"Validation failed: File '{filename}' is empty.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty."
        )

    if file_size > max_bytes:
        logger.warning(
            f"Validation failed: File '{filename}' size ({file_size} bytes) "
            f"exceeds maximum allowed limit ({max_bytes} bytes)."
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB."
        )


def validate_pdf(file: UploadFile) -> None:
    """Validates that the file has a PDF mimetype, magic bytes, and readable structure.
    
    Args:
        file (UploadFile): The uploaded file stream.
        
    Raises:
        HTTPException: If the file content type, magic bytes, or file structure is invalid.
    """
    filename = file.filename or "unknown"
    content_type = file.content_type

    # 1. Content-Type Header Validation
    if not content_type or content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            f"Validation failed: File '{filename}' has invalid content-type '{content_type}'."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type. Only PDF files ({', '.join(ALLOWED_CONTENT_TYPES)}) are allowed."
        )

    # 2. PDF Magic Bytes Signature Validation (%PDF-)
    try:
        header = file.file.read(len(PDF_SIGNATURE))
        file.file.seek(0)
        if header != PDF_SIGNATURE:
            logger.warning(f"Validation failed: File '{filename}' does not contain valid PDF magic bytes.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file content. The file is not a valid PDF document."
            )
    except OSError as e:
        logger.error(f"Failed to read file signature for '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading file signature for validation."
        )

    # 3. Structural Integrity & Corruption Check
    try:
        reader = PdfReader(file.file)
        # Attempt to read the page count to trigger parsing of catalog structure
        _ = len(reader.pages)
        file.file.seek(0)
    except PdfReadError as e:
        logger.warning(f"Validation failed: File '{filename}' is corrupted or unreadable. Detail: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is corrupted and cannot be parsed."
        )
    except Exception as e:
        logger.error(f"Unexpected error validating structure of '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is invalid or corrupted."
        )


@router.post(
    "/upload", 
    response_model=UploadResponse, 
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF Document",
    description="Uploads a PDF file, validates it, and stores it securely with a unique identifier."
)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Receives and processes a PDF file upload.
    
    Args:
        file (UploadFile): The uploaded file payload.
        
    Returns:
        UploadResponse: Metadata of the uploaded and saved document.
        
    Raises:
        HTTPException: For invalid file signatures, wrong extensions, sizes, or save failures.
    """
    # 1. Validate file presence and non-None filename
    if not file or not file.filename:
        logger.warning("Validation failed: Upload started with empty payload.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was uploaded."
        )

    # Capture original filename as a non-optional string for type narrowing
    original_filename: str = file.filename
    logger.info(f"File upload started: '{original_filename}'")

    # 2. Validate file extension extension match
    original_path = Path(original_filename)
    if original_path.suffix.lower() != ".pdf":
        logger.warning(
            f"Validation failed: Rejected file '{original_filename}' with invalid extension '{original_path.suffix}'."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Only .pdf files are allowed."
        )

    # 3. Perform file validations
    validate_file_size(file)
    validate_pdf(file)

    # 4. Generate unique document ID and timestamp
    document_id = uuid.uuid4()
    upload_timestamp = datetime.now(timezone.utc).isoformat()

    # 5. Sanitize filename and create unique stored name
    sanitized_name = sanitize_filename(original_filename)
    stored_filename = f"{document_id}_{sanitized_name}"
    
    # Build complete destination path
    destination_path = UPLOAD_DIR / stored_filename

    try:
        # 6. Create target directory if it does not exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 7. Save the file stream to disk
        with destination_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(
            f"Upload completed successfully. Saved '{original_filename}' as '{stored_filename}' with ID {document_id}."
        )

    except OSError as e:
        logger.error(
            f"Upload failed: File write error for '{original_filename}' to '{destination_path}'. Error: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save the uploaded file on the server."
        )
    finally:
        # Guarantee the file upload stream is closed
        await file.close()

    # =========================================================================
    # Future Integration Blueprint (RAG Pipeline):
    #
    # 1. pdf_parser.py:
    #    text_content = pdf_parser.extract_text(destination_path)
    #
    # 2. chunking.py:
    #    chunks = chunking.split_text(text_content, chunk_size=500, overlap=50)
    #
    # 3. embedding_service.py:
    #    embeddings = embedding_service.get_embeddings(chunks)
    #
    # 4. vector_store.py:
    #    vector_store.upsert(document_id, chunks, embeddings)
    # =========================================================================

    return UploadResponse(
        document_id=document_id,
        filename=original_filename,
        stored_filename=stored_filename,
        status="uploaded",
        upload_timestamp=upload_timestamp
    )
