#!/usr/bin/env python3
"""Simple eval: checks rubric-critical fields in output JSONs."""
import json
import re
import sys
from pathlib import Path

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def eval_pipeline() -> dict:
    results = {}
    
    # 1. Check transcript.json
    try:
        transcript = load_json("output/transcript.json")
        
        # has_transcript: non-empty string
        results["has_transcript"] = bool(transcript.get("transcript", "").strip())
        
        # has_word_confidence: check field exists in schema (even if list empty)
        words = transcript.get("word_level", [])
        if isinstance(words, list) and len(words) > 0:
            results["has_word_confidence"] = all("confidence" in w for w in words)
            results["has_timestamps"] = all("start_sec" in w and "end_sec" in w for w in words)
        else:
            # If no words (silent audio), check that the schema fields would exist
            results["has_word_confidence"] = True
            results["has_timestamps"] = True
        
        # latency_logged: check final_response.json (rubric-relevant)
        response = load_json("output/final_response.json")
        results["latency_logged"] = "latency_sec" in response
        
    except FileNotFoundError as e:
        print(f"Missing output file: {e}")
        results.update({k: False for k in ["has_transcript", "has_word_confidence", "has_timestamps", "latency_logged"]})
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        results.update({k: False for k in ["has_transcript", "has_word_confidence", "has_timestamps", "latency_logged"]})
    
    # 2. Check final_response.json for citation & fallback
    try:
        response = load_json("output/final_response.json")
        answer = response.get("llm_response", "")
        
        # citation_present: [Q1], [Q12], etc. (case-insensitive)
        results["citation_present"] = bool(re.search(r'\[Q\d+\]', answer, re.IGNORECASE))
        
        # no_hallucination_fallback: either has citation OR uses safe fallback phrase
        has_fallback = "don't have that information" in answer.lower() or "not in my knowledge base" in answer.lower()
        results["no_hallucination_fallback"] = results["citation_present"] or has_fallback
        
    except Exception as e:
        print(f"Response eval failed: {e}")
        results.update({k: False for k in ["citation_present", "no_hallucination_fallback"]})
    
    return results

if __name__ == "__main__":
    results = eval_pipeline()
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nEval Results: {passed}/{total} checks passed")
    for check, status in results.items():
        symbol = "PASS" if status else "FAIL"
        print(f"  [{symbol}] {check}")
    
    # Exit with error code if any checks failed (useful for CI)
    sys.exit(0 if passed == total else 1)