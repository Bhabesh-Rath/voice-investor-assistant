import os
import json
import time
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Model registry: maps size name -> HF repo + revision + expected files
WHISPER_MODELS = {
    "base": {
        "repo": "Systran/faster-whisper-base",
        "revision": "main",
        "files": ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
    },
    "small": {
        "repo": "Systran/faster-whisper-small",
        "revision": "536b0662742c02347bc0e980a01041f333bce120",
        "files": ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
    }
}

def get_local_model_path(model_size: str, base_dir: str = "data/models/whisper") -> str:
    """Returns the local path where the model should reside."""
    return os.path.join(base_dir, model_size)

def is_model_downloaded(model_size: str, base_dir: str = "data/models/whisper") -> bool:
    """Check if all required model files exist locally."""
    model_path = get_local_model_path(model_size, base_dir)
    required_files = WHISPER_MODELS.get(model_size, {}).get("files", [])
    return all(os.path.exists(os.path.join(model_path, f)) for f in required_files)

def transcribe_audio(
    audio_path: str, 
    output_path: str = "output/transcript.json",
    model_size: str = "small"
) -> dict:
    start = time.perf_counter()
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    if model_size not in WHISPER_MODELS:
        raise ValueError(f"Unsupported Whisper model size: {model_size}. Choose from {list(WHISPER_MODELS.keys())}")
    
    model_cache = os.path.join("data", "models", "whisper")
    local_model_path = get_local_model_path(model_size, model_cache)
    
    # Try local first, fallback to auto-download
    if is_model_downloaded(model_size, model_cache):
        logger.info(f"Loading Whisper '{model_size}' from local path: {local_model_path}")
        model = WhisperModel(local_model_path, device="cpu", compute_type="int8")
    else:
        logger.info(f"Local '{model_size}' not found. Fetching from HuggingFace...")
        os.makedirs(model_cache, exist_ok=True)
        model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8", 
            download_dir=model_cache
        )
        logger.info(f"Model cached to {local_model_path}")
    
    try:
        logger.info("Transcribing audio...")
        segments, info = model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            word_timestamps=True,
            initial_prompt="Sunrise Asset Management, SIP, KYC, redemption, NAV, mutual fund, taxation"
        )
        
        words_data = []
        full_text = []
        for seg in segments:
            for w in seg.words:
                words_data.append({
                    "word": w.word.strip(),
                    "confidence": round(w.probability, 4),
                    "start_sec": round(w.start, 3),
                    "end_sec": round(w.end, 3)
                })
                full_text.append(w.word.strip())
                
        transcript = " ".join(full_text)
        latency = round(time.perf_counter() - start, 3)
        
        result = {
            "transcript": transcript,
            "word_level": words_data,
            "language": info.language,
            "latency_sec": latency
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Transcription complete in {latency}s. Output: {output_path}")
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise