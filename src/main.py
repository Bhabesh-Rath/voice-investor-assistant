import os
import sys
import time
import logging
import argparse
import json
from pathlib import Path
from transcriber import transcribe_audio
from ingestor import extract_and_chunk_faq
from retriever import embed_and_store, query_kb
from generator import generate_answer

# Setup logging - Windows-safe UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def validate_paths(audio_path: str, pdf_path: str) -> None:
    """Ensure input files exist before running the pipeline."""
    audio_p, pdf_p = Path(audio_path), Path(pdf_path)
    if not audio_p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_p.resolve()}")
    if not pdf_p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_p.resolve()}")
    if audio_p.suffix.lower() not in [".mp3", ".wav", ".m4a", ".ogg"]:
        logging.warning("Audio format may not be optimized. .mp3/.wav recommended.")
    if pdf_p.suffix.lower() != ".pdf":
        raise ValueError("PDF file must have a .pdf extension.")

def run_pipeline(audio_path: str, pdf_path: str, model_tag: str):
    validate_paths(audio_path, pdf_path)
    start_total = time.perf_counter()
    os.makedirs("output", exist_ok=True)
    
    # 1. Transcribe
    logging.info("Step 1: Voice Transcription")
    t1 = time.perf_counter()
    transcript_data = transcribe_audio(audio_path, "output/transcript.json")
    transcript = transcript_data["transcript"]
    logging.info(f"Transcription done in {time.perf_counter()-t1:.2f}s | Text: '{transcript[:80]}...'")
    
    # 2. Ingest PDF
    logging.info("Step 2: PDF Ingestion & Chunking")
    t2 = time.perf_counter()
    chunks = extract_and_chunk_faq(pdf_path)
    embed_and_store(chunks)
    logging.info(f"Vector store ready in {time.perf_counter()-t2:.2f}s")
    
    # 3. Retrieve
    logging.info("Step 3: Context Retrieval")
    t3 = time.perf_counter()
    context = query_kb(transcript)
    logging.info(f"Retrieved in {time.perf_counter()-t3:.2f}s")
    
    # 4. Generate
    logging.info("Step 4: LLM Generation")
    t4 = time.perf_counter()
    result = generate_answer(transcript, context, model_tag)
    logging.info(f"Generated in {time.perf_counter()-t4:.2f}s")
    
    total_time = round(time.perf_counter() - start_total, 3)
    
    # FINAL OUTPUT: includes latency_sec for eval.py
    final_output = {
        "transcript": transcript,
        "context_used": context,
        "llm_response": result["answer"],
        "model_used": result["model_used"],
        "latency_sec": total_time
    }
    
    with open("output/final_response.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    logging.info(f"Pipeline complete in {total_time}s. Output saved to output/final_response.json")
    print("\n" + "="*50)
    print("TRANSCRIPT:\n", transcript)
    print("="*50)
    print("RESPONSE:\n", result["answer"])
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice-Powered Investor Support Assistant (Local RAG Pipeline)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
EXAMPLES:
  python src/main.py
  python src/main.py --audio my_query.wav --pdf my_faq.pdf
  python src/main.py --model llama3.2:3b
        """
    )
    parser.add_argument("--audio", default="input/investor_sample.mp3", help="Path to investor audio file (.mp3/.wav)")
    parser.add_argument("--pdf", default="input/SunriseAMC_FAQ.pdf", help="Path to FAQ PDF file")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model tag (e.g., llama3.2:3b, gemma4:e2b)")
    args = parser.parse_args()
    
    try:
        run_pipeline(args.audio, args.pdf, args.model)
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        sys.exit(1)