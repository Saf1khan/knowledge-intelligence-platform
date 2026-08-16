from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import uuid
import logging
import time
import re
import hashlib
import tiktoken
from pydantic import BaseModel, Field, model_validator

# Initialize logger
logger = logging.getLogger("app.services.ingestion.chunking")


# Custom Exceptions
class ChunkingError(Exception):
    """Base exception for all chunking-related errors."""
    pass


class ChunkValidationError(ChunkingError):
    """Raised when chunk text validation fails (e.g. empty, duplicate)."""
    pass


class ChunkConfigurationError(ChunkingError):
    """Raised when configuration parameters are invalid."""
    pass


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies for RAG pipeline ingestion."""
    RECURSIVE = "RECURSIVE"
    TOKEN = "TOKEN"
    SEMANTIC = "SEMANTIC"
    HYBRID = "HYBRID"


class ChunkingConfig(BaseModel):
    """Pydantic model validating text chunker parameters.
    
    Attributes:
        chunk_size (int): Maximum chunk length (characters or tokens).
        chunk_overlap (int): Number of overlapping units between adjacent chunks.
        strategy (ChunkingStrategy): Segmentation strategy to apply.
        tokenizer_name (str): Tokenizer model name (e.g., "cl100k_base").
    """
    chunk_size: int = Field(
        default=1000, 
        description="Maximum length per text chunk (characters or tokens)"
    )
    chunk_overlap: int = Field(
        default=200, 
        description="Number of overlapping units between adjacent chunks"
    )
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.RECURSIVE,
        description="Text segmentation strategy"
    )
    tokenizer_name: str = Field(
        default="cl100k_base",
        description="Tokenizer model name to use for token counts"
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        """Ensures that configured overlap is non-negative and smaller than chunk size."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        return self


class DocumentChunk(BaseModel):
    """Pydantic model representing a single segmented chunk of a document.
    
    Attributes:
        chunk_id (uuid.UUID): Globally unique identifier for this specific chunk.
        document_id (uuid.UUID): ID reference of the parent document.
        chunk_index (int): 0-indexed position representing order of the chunk.
        text (str): The text slice content of the chunk.
        character_count (int): Length of the text slice in characters.
        token_estimate (int): Estimated or actual token count.
        chunk_hash (str): Cryptographic SHA-256 hash of the chunk text.
        quality_score (float): Calculated readability and chunk quality score (0.0 to 1.0).
        preview (str): First 150 characters of the chunk for rapid inspection.
        start_char (int): The starting index of the slice in the parent document.
        end_char (int): The ending index of the slice in the parent document.
        metadata (dict): Arbitrary dictionary store for RAG metadata.
    """
    chunk_id: uuid.UUID = Field(
        ..., 
        description="Unique identifier for this chunk"
    )
    document_id: uuid.UUID = Field(
        ..., 
        description="Identifier of the source document"
    )
    chunk_index: int = Field(
        ..., 
        description="0-indexed position indicating ordering"
    )
    text: str = Field(
        ..., 
        description="Text content of the chunk"
    )
    character_count: int = Field(
        ..., 
        description="Number of characters in the chunk text"
    )
    token_estimate: int = Field(
        ...,
        description="Actual token count or estimate for the chunk"
    )
    chunk_hash: str = Field(
        ...,
        description="SHA-256 hash generated from chunk text"
    )
    quality_score: float = Field(
        ...,
        description="Calculated quality indicator score from 0.0 to 1.0"
    )
    preview: str = Field(
        ...,
        description="A 150-character preview of the chunk content"
    )
    start_char: int = Field(
        ..., 
        description="Start character index of the chunk in the original text"
    )
    end_char: int = Field(
        ..., 
        description="End character index of the chunk in the original text"
    )
    metadata: dict = Field(
        default_factory=dict, 
        description="Metadata container for RAG integration"
    )


class ChunkingResult(BaseModel):
    """Consolidated representation of document chunking output and operational metrics.
    
    Attributes:
        total_chunks (int): Total number of chunks generated.
        total_characters (int): Total number of characters processed.
        average_chunk_size (float): The average size of the generated chunks (chars or tokens).
        average_overlap_size (float): The average computed overlap between adjacent chunks.
        chunking_duration_ms (float): Total processing time in milliseconds.
        min_chunk_size (int): Size of the smallest chunk.
        max_chunk_size (int): Size of the largest chunk.
        chunk_size_distribution (dict): Size frequency buckets.
        chunks (List[DocumentChunk]): List of DocumentChunk models.
    """
    total_chunks: int = Field(..., description="Total count of successfully generated chunks")
    total_characters: int = Field(..., description="Total characters processed from the original document")
    average_chunk_size: float = Field(..., description="Average chunk size (characters or tokens)")
    average_overlap_size: float = Field(..., description="Average computed overlap between adjacent chunks")
    chunking_duration_ms: float = Field(..., description="Total chunking process duration in milliseconds")
    min_chunk_size: int = Field(..., description="Minimum chunk size")
    max_chunk_size: int = Field(..., description="Maximum chunk size")
    chunk_size_distribution: Dict[str, int] = Field(..., description="Distribution of chunk sizes")
    chunks: List[DocumentChunk] = Field(..., description="List of generated document chunks")


class TextChunker:
    """Enterprise text chunker that segments documents using recursive character or token splitting.
    
    Adheres to SOLID design principles by separating splitting, merging,
    validation, token estimation, hash generation, and quality metric scoring.
    """

    def __init__(self, config: ChunkingConfig = ChunkingConfig()) -> None:
        """Initializes the TextChunker with validation config and tiktoken encoder.
        
        Args:
            config (ChunkingConfig): Validated chunk size, overlap, strategy, and tokenizer.
        """
        self.config = config
        self.separators = ["\n\n", "\n", " ", ""]
        self.encoder: Optional[tiktoken.Encoding] = None
        try:
            self.encoder = tiktoken.get_encoding(self.config.tokenizer_name)
        except Exception as e:
            logger.error(
                f"Failed to initialize tiktoken encoder for model '{self.config.tokenizer_name}': {e}. "
                f"Fallback logic will be used.", 
                exc_info=True
            )

    def generate_chunk_hash(self, text: str) -> str:
        """Generates a SHA-256 hash for a given text segment.
        
        Args:
            text (str): Chunk text.
            
        Returns:
            str: SHA-256 hexadecimal hash string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def calculate_quality_score(self, text: str, token_count: int) -> float:
        """Calculates a chunk quality score from 0.0 to 1.0.
        
        Evaluates token size relative to config targets and checks for
        excessive whitespace/empty ratios.
        
        Args:
            text (str): Substring content of the chunk.
            token_count (int): Actual token count of the chunk.
            
        Returns:
            float: Quality score between 0.0 and 1.0.
        """
        if not text.strip():
            return 0.0
            
        score = 1.0
        
        # 1. Size-based deduction (penalize extremely small chunks)
        target_size = self.config.chunk_size
        if token_count < (target_size * 0.15):
            score -= 0.25
        elif token_count > target_size:
            score -= 0.1
            
        # 2. Whitespace density penalty
        non_whitespace = len(re.sub(r"\s", "", text))
        total_len = len(text)
        if total_len > 0:
            whitespace_ratio = (total_len - non_whitespace) / total_len
            if whitespace_ratio > 0.35:
                # Deduct based on excess whitespace ratio
                score -= (whitespace_ratio - 0.35) * 0.5
                
        return max(0.0, min(1.0, round(score, 2)))

    def estimate_tokens(self, text: str) -> int:
        """Calculates token count for a text block using tiktoken, falling back to estimation.
        
        Args:
            text (str): Input text string.
            
        Returns:
            int: Token count (guaranteed to be at least 1).
        """
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logger.warning(f"Error encoding text with tiktoken: {e}. Falling back to estimate.")
        return max(1, len(text) // 4)

    def _get_size(self, text: str) -> int:
        """Returns size of text in characters or tokens based on chunking strategy."""
        if self.config.strategy == ChunkingStrategy.TOKEN:
            return self.estimate_tokens(text)
        return len(text)

    def validate_chunk(self, chunk_text: str) -> None:
        """Validates that a chunk contains non-empty, non-whitespace textual data.
        
        Args:
            chunk_text (str): Chunk text to validate.
            
        Raises:
            ChunkValidationError: If the text is empty or only whitespace.
        """
        if not chunk_text or not chunk_text.strip():
            raise ChunkValidationError("Generated chunk text cannot be empty or whitespace-only.")

    def compute_overlap_text(self, text_a: str, text_b: str) -> str:
        """Finds the maximum overlapping substring between the end of text_a and start of text_b.
        
        Args:
            text_a (str): The preceding text chunk.
            text_b (str): The succeeding text chunk.
            
        Returns:
            str: The overlapping substring.
        """
        # Limit window to scan to prevent quadratic runtime on large texts
        max_overlap = min(len(text_a), len(text_b), self.config.chunk_overlap * 3)
        best_overlap = ""
        
        # Test suffixes of text_a against prefixes of text_b
        for i in range(1, max_overlap + 1):
            suffix = text_a[-i:]
            if text_b.startswith(suffix):
                best_overlap = suffix
        return best_overlap

    def calculate_chunk_metrics(
        self, chunks: List[DocumentChunk], duration_ms: float, total_chars: int
    ) -> ChunkingResult:
        """Computes statistical metrics and distributions for the generated chunks.
        
        Args:
            chunks (List[DocumentChunk]): List of built document chunks.
            duration_ms (float): Processing time in milliseconds.
            total_chars (int): Character count of the source text.
            
        Returns:
            ChunkingResult: The calculated metrics wrapper.
        """
        total_chunks = len(chunks)
        if total_chunks == 0:
            return ChunkingResult(
                total_chunks=0,
                total_characters=total_chars,
                average_chunk_size=0.0,
                average_overlap_size=0.0,
                chunking_duration_ms=duration_ms,
                min_chunk_size=0,
                max_chunk_size=0,
                chunk_size_distribution={},
                chunks=[]
            )

        sizes = [self._get_size(c.text) for c in chunks]
        min_size = min(sizes)
        max_size = max(sizes)
        avg_size = sum(sizes) / total_chunks

        # Calculate actual overlap sizes between adjacent chunks
        overlap_sizes = []
        for i in range(total_chunks - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            overlap_str = self.compute_overlap_text(curr_chunk.text, next_chunk.text)
            
            if self.config.strategy == ChunkingStrategy.TOKEN:
                overlap_sizes.append(self.estimate_tokens(overlap_str))
            else:
                overlap_sizes.append(len(overlap_str))
        
        avg_overlap = sum(overlap_sizes) / len(overlap_sizes) if overlap_sizes else 0.0

        # Build basic distribution buckets based on size units (chars or tokens)
        distribution = {"small_less_300": 0, "medium_300_800": 0, "large_above_800": 0}
        for size in sizes:
            if size < 300:
                distribution["small_less_300"] += 1
            elif size <= 800:
                distribution["medium_300_800"] += 1
            else:
                distribution["large_above_800"] += 1

        return ChunkingResult(
            total_chunks=total_chunks,
            total_characters=total_chars,
            average_chunk_size=round(avg_size, 2),
            average_overlap_size=round(avg_overlap, 2),
            chunking_duration_ms=round(duration_ms, 2),
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            chunk_size_distribution=distribution,
            chunks=chunks
        )

    def _split_text(self, text: str, separator: str) -> List[Tuple[str, int]]:
        """Splits text by a separator, returning the substrings and their relative start indices."""
        parts = []
        start = 0
        if separator == "":
            return [(char, idx) for idx, char in enumerate(text)]

        while True:
            idx = text.find(separator, start)
            if idx == -1:
                parts.append((text[start:], start))
                break
            parts.append((text[start:idx], start))
            start = idx + len(separator)
        return parts

    def _recursive_split(
        self, text: str, separators: List[str], start_offset: int, chunk_size: int
    ) -> List[Tuple[str, int, int]]:
        """Recursively segments text using hierarchical separators to fit inside chunk_size limit."""
        if self._get_size(text) <= chunk_size or not separators:
            return [(text, start_offset, start_offset + len(text))]

        separator = separators[0]
        splits = self._split_text(text, separator)

        if len(splits) == 1 and splits[0][0] == text:
            return self._recursive_split(text, separators[1:], start_offset, chunk_size)

        results = []
        for part, relative_start in splits:
            part_start = start_offset + relative_start
            if self._get_size(part) > chunk_size:
                results.extend(
                    self._recursive_split(part, separators[1:], part_start, chunk_size)
                )
            else:
                if part.strip():
                    results.append((part, part_start, part_start + len(part)))
        return results

    def _merge_segments(
        self, segments: List[Tuple[str, int, int]], chunk_size: int, chunk_overlap: int
    ) -> List[Tuple[str, int, int]]:
        """Merges atomic segments into larger overlapping chunks based on size (char or token)."""
        chunks = []
        if not segments:
            return chunks

        current_segment_indices = []
        current_len = 0

        i = 0
        while i < len(segments):
            seg_text, seg_start, seg_end = segments[i]
            seg_size = self._get_size(seg_text)

            # Join separator addition cost (1 space in characters or estimate ~1 token)
            join_cost = 1

            if current_segment_indices and current_len + join_cost + seg_size > chunk_size:
                chunk_text_parts = [segments[idx][0] for idx in current_segment_indices]
                chunk_text = " ".join(chunk_text_parts)
                chunk_start = segments[current_segment_indices[0]][1]
                chunk_end = segments[current_segment_indices[-1]][2]
                chunks.append((chunk_text, chunk_start, chunk_end))

                # Implement sliding window rollback logic using tokens or characters
                overlap_len = 0
                rollback_idx = i - 1
                while rollback_idx >= current_segment_indices[0]:
                    rollback_text = segments[rollback_idx][0]
                    rollback_size = self._get_size(rollback_text)
                    if overlap_len + rollback_size > chunk_overlap:
                        break
                    overlap_len += rollback_size + join_cost
                    rollback_idx -= 1

                # Guard to guarantee forward progress
                next_start_idx = max(rollback_idx + 1, current_segment_indices[0] + 1)
                i = next_start_idx
                current_segment_indices = []
                current_len = 0
            else:
                current_segment_indices.append(i)
                current_len += (join_cost if current_len > 0 else 0) + seg_size
                i += 1

        if current_segment_indices:
            chunk_text_parts = [segments[idx][0] for idx in current_segment_indices]
            chunk_text = " ".join(chunk_text_parts)
            chunk_start = segments[current_segment_indices[0]][1]
            chunk_end = segments[current_segment_indices[-1]][2]
            chunks.append((chunk_text, chunk_start, chunk_end))

        return chunks

    def chunk_document(self, document_id: uuid.UUID, text: str) -> ChunkingResult:
        """Processes document text and returns metrics alongside the generated chunks.
        
        Args:
            document_id (uuid.UUID): ID of the parent document.
            text (str): Full text string to chunk.
            
        Returns:
            ChunkingResult: Detailed object representing the segmented chunks and stats.
            
        Raises:
            ChunkConfigurationError: If chunking config is invalid.
            ChunkingError: If an unexpected error occurs during splitting or merging.
        """
        start_time = time.perf_counter()
        logger.info(
            f"Chunking started for document {document_id} using strategy {self.config.strategy.value}. "
            f"Size: {len(text)} characters."
        )

        if not text or not text.strip():
            logger.warning(f"Aborting chunking: Text is empty for document {document_id}.")
            return self.calculate_chunk_metrics([], 0.0, 0)

        # 1. Recursive splitting phase
        split_start = time.perf_counter()
        try:
            segments = self._recursive_split(
                text, self.separators, start_offset=0, chunk_size=self.config.chunk_size
            )
        except Exception as e:
            logger.error(f"Error during recursive splitting: {e}", exc_info=True)
            raise ChunkingError(f"Failed to segment document {document_id}: {str(e)}")
        split_duration_ms = (time.perf_counter() - split_start) * 1000

        # 2. Merging phase
        merge_start = time.perf_counter()
        try:
            merged_chunks = self._merge_segments(
                segments, self.config.chunk_size, self.config.chunk_overlap
            )
        except Exception as e:
            logger.error(f"Error during segment merging: {e}", exc_info=True)
            raise ChunkingError(f"Failed to merge text segments: {str(e)}")
        merge_duration_ms = (time.perf_counter() - merge_start) * 1000

        # 3. Validation, deduplication, and model construction
        document_chunks: List[DocumentChunk] = []
        seen_texts = set()
        duplicate_count = 0
        chunk_idx = 0

        for raw_text, start_char, end_char in merged_chunks:
            cleaned_text = raw_text.strip()
            
            try:
                self.validate_chunk(cleaned_text)
            except ChunkValidationError as val_err:
                logger.warning(f"Chunk validation failed: {val_err}. Skipping chunk.")
                continue

            if cleaned_text in seen_texts:
                logger.info(f"Duplicate chunk text detected under document {document_id}. Skipping.")
                duplicate_count += 1
                continue

            seen_texts.add(cleaned_text)

            # Metadata creation with page-level defaults, versioning, and score placeholders
            chunk_metadata = {
                "source_document_id": str(document_id),
                "page_number": None,
                "section_title": None,
                "chunk_strategy": self.config.strategy.value,
                "embedding_status": "pending",
                "pipeline_version": "1.0.0",
                "retrieval_score": None
            }

            token_estimate = self.estimate_tokens(cleaned_text)
            chunk_hash = self.generate_chunk_hash(cleaned_text)
            quality_score = self.calculate_quality_score(cleaned_text, token_estimate)
            preview = cleaned_text[:150]

            document_chunks.append(
                DocumentChunk(
                    chunk_id=uuid.uuid4(),
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    text=cleaned_text,
                    character_count=len(cleaned_text),
                    token_estimate=token_estimate,
                    chunk_hash=chunk_hash,
                    quality_score=quality_score,
                    preview=preview,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata
                )
            )
            chunk_idx += 1

        # 4. Improved overlap validation via prefix/suffix text comparison
        if len(document_chunks) > 1:
            for idx in range(len(document_chunks) - 1):
                chunk_a = document_chunks[idx]
                chunk_b = document_chunks[idx + 1]
                
                # Check actual text overlap
                overlap_text = self.compute_overlap_text(chunk_a.text, chunk_b.text)
                
                if self.config.strategy == ChunkingStrategy.TOKEN:
                    overlap_val = self.estimate_tokens(overlap_text)
                    limit = self.config.chunk_overlap
                else:
                    overlap_val = len(overlap_text)
                    limit = self.config.chunk_overlap

                if overlap_val < (limit * 0.5):
                    logger.warning(
                        f"Low overlap detected between chunk {idx} and {idx + 1} "
                        f"for document {document_id}. Expected ~{limit} units, got actual overlap size of {overlap_val}."
                    )

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Chunking completed for document {document_id}. "
            f"Split: {split_duration_ms:.2f}ms, Merge: {merge_duration_ms:.2f}ms, Total: {total_duration_ms:.2f}ms. "
            f"Chunks: {len(document_chunks)} (Duplicates skipped: {duplicate_count})."
        )

        return self.calculate_chunk_metrics(document_chunks, total_duration_ms, len(text))

    def chunk_text(self, document_id: uuid.UUID, text: str) -> List[DocumentChunk]:
        """Segments raw text into ordered, overlapping chunks (Legacy / Public API wrapper).
        
        Args:
            document_id (uuid.UUID): ID of parent document context.
            text (str): Original full text string.
            
        Returns:
            List[DocumentChunk]: List of generated, indexed document chunks.
        """
        result = self.chunk_document(document_id, text)
        return result.chunks
