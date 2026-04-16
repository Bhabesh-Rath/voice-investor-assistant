import os
import re
import logging
import pdfplumber

logger = logging.getLogger(__name__)

def extract_and_chunk_faq(pdf_path: str) -> list[dict]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
    logger.info(f"Parsing PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            
        # Intelligent FAQ boundary chunking
        # Matches Q1, Q.1, Question 1, etc. and captures until next question or end
        pattern = r"((?:Q|Question)\s*\d+[\.\s:-]?[^\n]*)\n([\s\S]*?)(?=(?:Q|Question)\s*\d+[\.\s:-]?|$)"
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        
        if not matches:
            # Fallback: split by double newlines if regex fails
            matches = [(f"Section_{i}", chunk.strip()) for i, chunk in enumerate(full_text.split("\n\n")) if chunk.strip()]
            
        chunks = []
        for i, (header, body) in enumerate(matches):
            if not body.strip():
                continue
            chunks.append({
                "id": f"faq_{i}",
                "text": f"{header}\n{body.strip()}",
                "metadata": {
                    "faq_number": re.search(r"\d+", header).group() if re.search(r"\d+", header) else str(i),
                    "source": os.path.basename(pdf_path)
                }
            })
            
        logger.info(f"Extracted {len(chunks)} FAQ chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        raise