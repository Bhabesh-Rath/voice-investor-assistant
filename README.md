# Voice-Powered Investor Support Assistant
> A local, open-source RAG pipeline for Sunrise Asset Management investor queries.  
> **• Hardware-aware • Offline-capable**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This prototype accepts a voice query from an investor, transcribes it using Faster-Whisper, retrieves grounded answers from a FAQ knowledge base via ChromaDB + sentence-transformers, and generates a cited response using a locally-running LLM via Ollama.

**All components run locally with no paid APIs.**

## Testing Environment (Current Setup)
| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA GTX 1060 (6GB VRAM) |
| **RAM** | 32GB DDR4 |
| **CPU** | Intel Core i5 8th Gen |
| **OS** | Windows 11 |
| **Python** | 3.12.2 |
| **Ollama** | v0.20.7 |

✅ **Verified working** on this configuration. See `DECISIONS.md` for hardware-aware optimizations.

## Quick Start (Single Command)

### 1. Clone & Setup
```bash
git clone <your-repo>
cd voice-investor-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 2. Start Ollama & Pull Model
Ensure Ollama is running (auto-starts on install)
ollama serve
Pull recommended model (~2GB, Q4 quantized)
ollama pull llama3.2:3b
OR use your manually imported model (gemma4:e2b tested)

### 3. Place Input Files
Ensure your input files are in the input/ directory:
input/
├── investor_sample.mp3   # Provided audio query
└── SunriseAMC_FAQ.pdf    # Provided knowledge base

### 5. Run the Full Pipeline
* Default: uses input/ files, gemma4:e2b model
python src/main.py

* Custom paths & model
python src/main.py ^
  --audio path/to/query.mp3 ^
  --pdf path/to/faq.pdf ^
  --model llama3.2:3b
  
## Outputs saved to
* output/transcript.json — Word-level transcription with confidence & timestamps
* output/final_response.json — Grounded answer with FAQ citation

## Project structure
.
├── input/                     # Place audio & PDF here
│   ├── investor_sample.mp3
│   └── SunriseAMC_FAQ.pdf
├── data/                      # Auto-generated (gitignored)
│   ├── models/
│   │   ├── whisper/{base,small}/
│   │   └── sentence-transformers/
│   └── chroma/                # Vector store
├── output/                    # Auto-generated (gitignored)
│   ├── transcript.json
│   └── final_response.json
├── src/
│   ├── main.py               # CLI orchestrator
│   ├── transcriber.py        # Faster-Whisper wrapper
│   ├── ingestor.py           # PDF parsing + smart chunking
│   ├── retriever.py          # ChromaDB + embeddings
│   └── generator.py          # Ollama LLM interface
├── requirements.txt
├── README.md
├── DECISIONS.md
└── .gitignore

## Evaluation & Testing
* Run basic quality check (citation format + non-empty response)
python src/eval.py

* Benchmark latency (unprompted green flag)
python src/main.py --audio input/investor_sample.mp3 2>&1 | findstr "done in"

## Edge case handling
* Empty/malformed audio → Graceful error with logging
* PDF not found → Clear FileNotFoundError with resolved path
* Empty retrieval results → LLM responds: "I don't have that information in my knowledge base."
* Ollama connection failed → Retry logic + fallback suggestion in logs
* GPU OOM → Whisper forced to CPU; LLM quantization enforced

## License & Disclaimer
MIT License — free for assessment and educational use.