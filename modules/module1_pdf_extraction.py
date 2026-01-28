"""
Module 1: PDF Text Extraction

PURPOSE:
    This module extracts raw text from PDF research papers as the first step in the semantic
    modeling pipeline. It serves as the foundation for all downstream processing.

WHY THIS MODULE:
    - PDFs contain valuable unstructured data (research papers, technical documents)
    - Text extraction is the critical first step before entity extraction and KG building
    - Different PDFs have different formats/encodings, requiring multiple extraction methods
    - Proper text extraction impacts the quality of all downstream modules

TECHNIQUES USED:
    1. PyPDF2: Fast, lightweight extraction for simple PDFs
    2. pdfplumber: More accurate for complex layouts and tables
    3. Fallback mechanism: If one method fails, tries the other
    4. Text cleaning: Removes PDF artifacts, normalizes whitespace

ALGORITHMS:
    - PyPDF2.PdfReader: Standard PDF parsing algorithm
    - pdfplumber: Advanced layout analysis with table detection
    - Regex-based cleaning: Removes control characters and PDF artifacts

OUTPUT FORMAT:
    JSON file per PDF containing:
    - filename: Source PDF name
    - text: Extracted clean text
    - metadata: Title, author, pages, etc.
    - text_length: Character count
    - extraction_method: Which method was used
    - timestamp: When extraction occurred
"""
import os
from typing import Dict, List, Optional
import PyPDF2
import pdfplumber
from utils import get_logger, save_json, get_timestamp
import config

logger = get_logger(__name__)


class PDFExtractor:
    """
    Extract text from PDF files using multiple extraction methods.

    WHY THIS CLASS:
        Encapsulates PDF extraction logic with fallback mechanisms to handle
        various PDF formats and encodings. Provides a unified interface for
        extracting text regardless of the PDF's internal structure.

    METHODS:
        - extract_with_pypdf2(): Fast extraction for simple PDFs
        - extract_with_pdfplumber(): Accurate extraction for complex layouts
        - extract_metadata(): Get PDF metadata (title, author, pages)
        - extract(): Main extraction with automatic fallback
        - extract_batch(): Process multiple PDFs in a directory

    DESIGN PATTERN:
        Strategy pattern - tries multiple extraction strategies and uses the best result
    """

    def __init__(self):
        """
        Initialize the PDF extractor.

        WHAT THIS DOES:
            Sets up the logger for tracking extraction progress and errors.

        WHY:
            Logging is critical for debugging extraction issues and tracking
            which PDFs succeed or fail during batch processing.
        """
        self.logger = logger

    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text using PyPDF2 library.

        WHAT THIS DOES:
            Uses PyPDF2.PdfReader to parse the PDF structure and extract text from
            each page sequentially. Concatenates all page text with separators.

        WHY USE PyPDF2:
            - Fast and lightweight (~10-20ms per page)
            - Works well for simple, text-based PDFs
            - Handles standard PDF encodings reliably
            - Good for PDFs without complex layouts

        ALGORITHM:
            1. Open PDF in binary read mode
            2. Create PdfReader object (parses PDF structure)
            3. Iterate through each page object
            4. Call extract_text() which decodes text content objects
            5. Concatenate with double newlines for page separation

        LIMITATIONS:
            - May struggle with scanned PDFs (images, not text)
            - Poor handling of complex multi-column layouts
            - Less accurate with tables and embedded graphics

        Args:
            pdf_path: Absolute path to PDF file

        Returns:
            Extracted text as string, or empty string if extraction fails
        """
        try:
            text = ""
            # Open in binary mode - PDFs are binary files, not text
            with open(pdf_path, 'rb') as file:
                # PdfReader parses PDF structure (objects, streams, metadata)
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                # Process each page sequentially
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    # extract_text() decodes text content streams from page object
                    text += page.extract_text() + "\n\n"

            self.logger.info(f"PyPDF2: Extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            # Log errors but don't crash - fallback will try pdfplumber
            self.logger.error(f"PyPDF2 extraction failed: {str(e)}")
            return ""

    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """
        Extract text using pdfplumber library (more accurate for complex layouts).

        WHAT THIS DOES:
            Uses pdfplumber's advanced layout analysis to extract text while preserving
            spatial relationships, tables, and multi-column layouts.

        WHY USE pdfplumber:
            - Better handling of complex layouts (multi-column, tables)
            - More accurate text positioning and reading order
            - Can extract tables as structured data
            - Works better with research papers (2-column format)

        ALGORITHM:
            1. Open PDF using pdfplumber (builds page object tree)
            2. For each page:
               a. Performs layout analysis (identifies text blocks, tables, images)
               b. Orders text blocks by reading order (top-to-bottom, left-to-right)
               c. Extracts text while preserving logical flow
            3. Concatenates all pages with separators

        TRADE-OFFS:
            - Slower than PyPDF2 (~50-100ms per page) due to layout analysis
            - Higher accuracy worth the extra time for complex documents

        Args:
            pdf_path: Absolute path to PDF file

        Returns:
            Extracted text as string, or empty string if extraction fails
        """
        try:
            text = ""
            # pdfplumber uses context manager for proper resource cleanup
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # extract_text() does layout analysis and reading order detection
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

            self.logger.info(f"pdfplumber: Extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            # Log errors - caller will try fallback method
            self.logger.error(f"pdfplumber extraction failed: {str(e)}")
            return ""

    def extract_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing metadata
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata

                return {
                    "title": metadata.get('/Title', 'Unknown'),
                    "author": metadata.get('/Author', 'Unknown'),
                    "subject": metadata.get('/Subject', 'Unknown'),
                    "creator": metadata.get('/Creator', 'Unknown'),
                    "producer": metadata.get('/Producer', 'Unknown'),
                    "num_pages": len(pdf_reader.pages)
                }
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}")
            return {
                "title": "Unknown",
                "author": "Unknown",
                "subject": "Unknown",
                "creator": "Unknown",
                "producer": "Unknown",
                "num_pages": 0
            }

    def extract(self, pdf_path: str, method: str = "pdfplumber") -> Dict:
        """
        Extract text and metadata from PDF

        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pypdf2' or 'pdfplumber')

        Returns:
            Dictionary containing text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.logger.info(f"Extracting text from: {pdf_path}")

        # Extract text
        if method == "pypdf2":
            text = self.extract_with_pypdf2(pdf_path)
        else:
            text = self.extract_with_pdfplumber(pdf_path)

        # Fallback to other method if extraction fails
        if not text or len(text) < 100:
            self.logger.warning(f"{method} extraction poor, trying fallback...")
            if method == "pypdf2":
                text = self.extract_with_pdfplumber(pdf_path)
            else:
                text = self.extract_with_pypdf2(pdf_path)

        # Extract metadata
        metadata = self.extract_metadata(pdf_path)

        # Clean text
        text = self._clean_text(text)

        result = {
            "filename": os.path.basename(pdf_path),
            "filepath": pdf_path,
            "text": text,
            "metadata": metadata,
            "text_length": len(text),
            "extraction_method": method,
            "timestamp": get_timestamp()
        }

        self.logger.info(f"Successfully extracted {len(text)} characters")
        return result

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing PDF artifacts and normalizing whitespace.

        WHAT THIS DOES:
            Post-processes raw extracted text to remove common PDF encoding artifacts,
            normalize whitespace, and prepare text for downstream NLP processing.

        WHY CLEANING IS NEEDED:
            - PDFs often contain Unicode artifacts from font encoding
            - Multiple spaces, tabs, newlines need normalization
            - Special characters (bullets, boxes) can confuse NLP models
            - Clean text improves entity extraction quality in Module 2

        CLEANING STEPS:
            1. Whitespace normalization: Collapses multiple spaces/tabs/newlines into single spaces
            2. Null character removal: \x00 (often from corrupted PDFs)
            3. Bullet point removal: \uf0b7, \u2022 (rendered differently by extractors)
            4. Preserves actual content while removing rendering artifacts

        WHY THESE SPECIFIC CHARACTERS:
            - \x00: Null bytes from encoding issues
            - \uf0b7: Private Use Area character used for bullets
            - \u2022: Unicode bullet point (can interfere with tokenization)

        Args:
            text: Raw extracted text from PDF

        Returns:
            Cleaned text ready for entity extraction
        """
        # Normalize whitespace: split on any whitespace, rejoin with single spaces
        # This handles: multiple spaces, tabs, newlines, carriage returns
        text = ' '.join(text.split())

        # Remove PDF encoding artifacts
        text = text.replace('\x00', '')        # Null bytes
        text = text.replace('\uf0b7', '')      # Private use bullet
        text = text.replace('\u2022', '')      # Unicode bullet

        return text

    def extract_batch(self, pdf_dir: str, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Extract text from multiple PDFs in a directory

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Optional directory to save JSON outputs

        Returns:
            List of extraction results
        """
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

        if not pdf_files:
            self.logger.warning(f"No PDF files found in {pdf_dir}")
            return []

        self.logger.info(f"Found {len(pdf_files)} PDF files")

        results = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                result = self.extract(pdf_path)
                results.append(result)

                # Save individual JSON if output_dir specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    json_filename = pdf_file.replace('.pdf', '_extracted.json')
                    json_path = os.path.join(output_dir, json_filename)
                    save_json(result, json_path)

            except Exception as e:
                self.logger.error(f"Failed to extract {pdf_file}: {str(e)}")

        self.logger.info(f"Successfully extracted {len(results)}/{len(pdf_files)} PDFs")
        return results


def main():
    """Test the PDF extraction module"""
    extractor = PDFExtractor()

    # Extract from all PDFs in the PDF directory
    results = extractor.extract_batch(
        pdf_dir=config.PDF_DIR,
        output_dir=config.JSON_DIR
    )

    if results:
        logger.info(f"Extraction complete. Processed {len(results)} files.")
        for result in results:
            logger.info(f"  - {result['filename']}: {result['text_length']} characters")
    else:
        logger.warning("No PDFs were processed")


if __name__ == "__main__":
    main()
