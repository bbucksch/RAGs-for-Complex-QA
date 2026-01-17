"""
Experiment 1: Standard RAG Baseline with BM25 Retrieval (FAST VERSION)
=======================================================================

This is a FAST version using BM25 (lexical retrieval) instead of Contriever.
BM25 doesn't require neural encoding, making it ~100x faster.

Pipeline:
1. Load first 1,200 questions from dev.json (or 50 in quick mode)
2. Use BM25 to retrieve top-100 documents (takes ~2-5 minutes vs 70 hours)
3. For each k in {1, 3, 5}:
   - Slice top-k contexts from retrieved docs
   - Generate answer using LLM with those contexts
   - Evaluate using Cover Exact Match (EM)
4. Log aggregated EM score for each k value

Quick Mode: Use --quick flag to process only first 50 questions for testing.

Note: You can switch back to Contriever later by using experiment1_contriever_rag.py
"""

import os
import sys
import json
import time
import re
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np

# Load environment variables
load_dotenv()

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of questions to evaluate (first 1,200 as per assignment)
NUM_QUESTIONS = 1200

# Quick mode for testing (first 50 questions only)
QUICK_MODE_QUESTIONS = 50

# Retrieval depths to evaluate
K_VALUES = [1, 3, 5]

# Number of docs to retrieve (for efficiency, then slice for different k)
TOP_K_RETRIEVAL = 100

# Rate limiting - Now handled by smart key rotation in GroqEngine!
# With 7 keys at 25 RPM each = 175 RPM theoretical max
# We use a minimal delay just to prevent CPU spinning
MIN_DELAY_BETWEEN_REQUESTS = 0.1  # Minimal delay, key manager handles actual rate limiting

# Checkpoint frequency (save progress every N questions)
CHECKPOINT_FREQUENCY = 50

# Output directory for results
OUTPUT_DIR = os.path.join(project_root, "experiments", "results")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Note: normalize_answer, cover_exact_match, extract_final_answer are imported from rag_prompt


# Import checkpoint utilities and common prompt
try:
    from experiments.checkpoint_utils import (
        load_checkpoint, save_checkpoint, 
        get_indices_to_process, merge_result_into_checkpoint
    )
    from experiments.rag_prompt import (
        create_rag_prompt, extract_final_answer, 
        normalize_answer, cover_exact_match
    )
except ImportError:
    # Fallback for direct execution
    from checkpoint_utils import (
        load_checkpoint, save_checkpoint,
        get_indices_to_process, merge_result_into_checkpoint
    )
    from rag_prompt import (
        create_rag_prompt, extract_final_answer,
        normalize_answer, cover_exact_match
    )


def tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    return text.lower().split()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(num_questions: int = NUM_QUESTIONS):
    """Load the dataset and corpus directly from JSON files."""
    print("\n📂 Loading dataset and corpus...")
    
    # Load dev.json
    dev_path = os.path.join(project_root, "data", "dev.json")
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    print(f"   ✅ Dev questions loaded: {len(dev_data)}")
    
    # Load corpus
    corpus_path = os.path.join(project_root, "wiki_musique_corpus.json")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_raw = json.load(f)
    print(f"   ✅ Corpus loaded: {len(corpus_raw)} documents")
    
    # Prepare corpus for BM25
    print("   ⏳ Preparing corpus for BM25...")
    doc_ids = list(corpus_raw.keys())
    doc_texts = []
    doc_titles = []
    
    if TQDM_AVAILABLE:
        iterator = tqdm(doc_ids, desc="   Preparing", unit="doc", ncols=80)
    else:
        iterator = doc_ids
    
    for doc_id in iterator:
        doc_data = corpus_raw[doc_id]
        title = doc_data.get("title", "")
        text = doc_data.get("text", "")
        doc_texts.append(f"{title} {text}")
        doc_titles.append(title)
    
    print(f"   ✅ Corpus prepared: {len(doc_texts)} documents")
    
    # Prepare queries and answers
    queries = []
    answers = []
    
    for item in dev_data[:num_questions]:
        queries.append({
            "id": item["_id"],
            "text": item["question"]
        })
        answers.append(item["answer"])
    
    print(f"   📋 Using first {num_questions} questions for evaluation")
    
    return queries, answers, doc_ids, doc_texts, doc_titles


# ============================================================================
# BM25 RETRIEVAL (FAST!)
# ============================================================================

def run_bm25_retrieval(queries: List[Dict], doc_ids: List[str], doc_texts: List[str]) -> Dict:
    """
    Run BM25 retrieval - MUCH faster than neural retrieval!
    
    BM25 uses term frequency matching, no neural encoding needed.
    Typically takes 2-5 minutes for 500K+ documents.
    """
    print("\n🔍 Running BM25 Retrieval (FAST!)...")
    print(f"   Building BM25 index for {len(doc_texts)} documents...")
    
    start_time = time.time()
    
    # Tokenize corpus with progress bar
    print("   ⏳ Tokenizing corpus...")
    if TQDM_AVAILABLE:
        tokenized_corpus = []
        for doc in tqdm(doc_texts, desc="   Tokenizing", unit="doc", ncols=80):
            tokenized_corpus.append(tokenize(doc))
    else:
        tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    
    # Build BM25 index
    print("   ⏳ Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    index_time = time.time() - start_time
    print(f"   ✅ BM25 index built in {index_time:.1f}s")
    
    # Retrieve for each query with progress bar
    print(f"   Retrieving top-{TOP_K_RETRIEVAL} documents for {len(queries)} queries...")
    
    results = {}
    
    if TQDM_AVAILABLE:
        iterator = tqdm(queries, desc="   Retrieving", unit="query", ncols=80)
    else:
        iterator = queries
    
    for i, query in enumerate(iterator):
        tokenized_query = tokenize(query["text"])
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]
        
        # Store results
        query_results = {}
        for idx in top_indices:
            query_results[doc_ids[idx]] = float(scores[idx])
        
        results[query["id"]] = query_results
        
        if not TQDM_AVAILABLE and (i + 1) % 200 == 0:
            print(f"   [{i+1}/{len(queries)}] queries processed...")
    
    elapsed = time.time() - start_time
    print(f"   ✅ Retrieval complete in {elapsed:.1f}s")
    
    return results


# ============================================================================
# GENERATION WITH LLM
# ============================================================================

def get_context_for_query(query_id: str, retrieval_results: Dict, doc_ids: List[str], 
                          doc_texts: List[str], k: int) -> str:
    """Get top-k retrieved contexts for a query."""
    if query_id not in retrieval_results:
        return ""
    
    # Get top-k doc IDs (already sorted by score)
    doc_scores = retrieval_results[query_id]
    top_k_doc_ids = list(doc_scores.keys())[:k]
    
    # Create id to index mapping
    id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    
    contexts = []
    for doc_id in top_k_doc_ids:
        if doc_id in id_to_idx:
            contexts.append(doc_texts[id_to_idx[doc_id]])
    
    return "\n\n".join(contexts)


# Note: create_rag_prompt is imported from rag_prompt module


def run_generation_for_k(
    k: int,
    queries: List[Dict],
    answers: List[str],
    retrieval_results: Dict,
    doc_ids: List[str],
    doc_texts: List[str],
    llm_engine,
    checkpoint_dir: str
) -> float:
    """Run generation and evaluation for a specific k value.
    
    Supports resuming from checkpoints and re-processing missing questions.
    """
    print(f"\n{'='*60}")
    print(f"🤖 GENERATION WITH k={k}")
    print(f"{'='*60}")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"bm25_checkpoint_k{k}.json")
    
    # Use new checkpoint validation to find missing questions
    indices_to_process, matches, total, results_list, checkpoint = get_indices_to_process(
        checkpoint_path, len(queries), verbose=True
    )
    
    # If all questions are already processed, just return the existing results
    if not indices_to_process:
        em_score = matches / total if total > 0 else 0
        print(f"   ✅ All questions already processed. EM: {em_score:.4f}")
        return em_score
    
    errors = 0
    processed_count = 0
    
    # Create iterator for remaining questions
    if TQDM_AVAILABLE and len(indices_to_process) > 0:
        iterator = tqdm(
            indices_to_process,
            desc=f"   k={k}",
            unit="q",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        iterator = indices_to_process
    
    for idx in iterator:
        query = queries[idx]
        gold_answer = answers[idx]
        
        question_id = query["id"]
        question_text = query["text"]
        
        # Get context
        context = get_context_for_query(question_id, retrieval_results, doc_ids, doc_texts, k)
        
        if not context:
            if TQDM_AVAILABLE:
                tqdm.write(f"   ⚠️ No context for question {idx+1}, skipping")
            else:
                print(f"   ⚠️ No context for question {idx+1}, skipping")
            errors += 1
            continue
        
        # Create prompt
        system_prompt, user_prompt = create_rag_prompt(question_text, context)
        
        try:
            # Generate answer
            response = llm_engine.get_chat_completion(user_prompt, system_prompt)
            
            # Extract and evaluate
            predicted = extract_final_answer(response)
            is_match = cover_exact_match(predicted, gold_answer)
            
            # Create result entry
            new_result = {
                "idx": idx,
                "question_id": question_id,
                "question": question_text,
                "gold_answer": gold_answer,
                "predicted": predicted,
                "is_match": is_match
            }
            
            # Merge result into checkpoint (handles deduplication)
            if not checkpoint:
                checkpoint = {"k": k, "results": []}
            checkpoint = merge_result_into_checkpoint(checkpoint, new_result)
            results_list = checkpoint.get("results", [])
            matches = checkpoint.get("matches", 0)
            total = checkpoint.get("total", 0)
            
            processed_count += 1
            
            # Update progress bar description with current EM
            if TQDM_AVAILABLE and hasattr(iterator, 'set_postfix'):
                current_em = matches / total if total > 0 else 0
                iterator.set_postfix(EM=f"{current_em:.3f}", correct=matches)
            elif not TQDM_AVAILABLE and processed_count % 10 == 0:
                current_em = matches / total if total > 0 else 0
                print(f"   [Processed {processed_count}] EM: {current_em:.4f} ({matches}/{total})")
            
            # Checkpoint periodically
            if processed_count % CHECKPOINT_FREQUENCY == 0:
                checkpoint["k"] = k
                save_checkpoint(checkpoint_path, checkpoint, silent=TQDM_AVAILABLE)
                if TQDM_AVAILABLE:
                    tqdm.write(f"  💾 Checkpoint saved: {checkpoint_path}")
            
            # Minimal delay - actual rate limiting is handled by GroqEngine's token limiter
            time.sleep(MIN_DELAY_BETWEEN_REQUESTS)
            
        except Exception as e:
            errors += 1
            error_msg = str(e)
            if TQDM_AVAILABLE:
                tqdm.write(f"   ❌ Error at question {idx+1}: {error_msg[:80]}")
            else:
                print(f"   ❌ Error at question {idx+1}: {error_msg[:100]}")
            
            # Rate limiting and daily limits are now handled by GroqEngine
            # Just a small delay before retry
            time.sleep(1)
    
    # Final save - ensure checkpoint is up to date
    if checkpoint:
        checkpoint["k"] = k
        save_checkpoint(checkpoint_path, checkpoint, silent=True)
    
    # Final EM calculation from checkpoint
    matches = checkpoint.get("matches", 0) if checkpoint else matches
    total = checkpoint.get("total", 0) if checkpoint else total
    results_list = checkpoint.get("results", []) if checkpoint else results_list
    em_score = matches / total if total > 0 else 0
    
    # Save final results
    final_results = {
        "k": k,
        "retriever": "BM25",
        "exact_match": em_score,
        "matches": matches,
        "total": total,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
        "results": results_list
    }
    
    final_path = os.path.join(checkpoint_dir, f"bm25_results_k{k}.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n   ✅ k={k} Complete!")
    print(f"   📊 Exact Match: {em_score:.4f} ({matches}/{total})")
    print(f"   ❌ Errors: {errors}")
    print(f"   💾 Results saved: {final_path}")
    
    return em_score


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Run Experiment 1 with BM25 (fast version)."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Experiment 1: Standard RAG with BM25')
    parser.add_argument('--quick', action='store_true', 
                        help=f'Quick mode: only process first {QUICK_MODE_QUESTIONS} questions')
    args = parser.parse_args()
    
    print("="*60)
    print("EXPERIMENT 1: Standard RAG Baseline with BM25 (FAST)")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        print(f"\n🚀 QUICK MODE: Processing only {QUICK_MODE_QUESTIONS} questions")
        num_questions = QUICK_MODE_QUESTIONS
    else:
        num_questions = NUM_QUESTIONS
    
    # Check LLM API
    # Check for Groq API keys (including numbered ones)
    groq_keys_exist = any([
        os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_API_KEY") != "your_groq_api_key_here",
        os.environ.get("GROQ_API_KEY_1") and os.environ.get("GROQ_API_KEY_1") != "your_groq_api_key_here"
    ])
    use_openai = os.environ.get("OPENAI_KEY") and os.environ.get("OPENAI_KEY") != "your_openai_api_key_here"
    
    if not groq_keys_exist and not use_openai:
        print("\n❌ ERROR: No LLM API key configured!")
        print("Set GROQ_API_KEY (free) or OPENAI_KEY (paid) in .env file")
        return
    
    # Initialize LLM
    if groq_keys_exist:
        print("\n✅ Using Groq (FREE) with Llama 3.1 8B Instant")
        from dexter.llms.groq_engine import GroqEngine
        llm_engine = GroqEngine(
            data="",
            model_name="llama-3.1-8b-instant",
            temperature=0.3
        )
    else:
        print("\n💰 Using OpenAI (PAID) with GPT-3.5-turbo")
        from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
        orchestrator = LLMEngineOrchestrator()
        llm_engine = orchestrator.get_llm_engine(
            data="",
            llm_class="openai",
            model_name="gpt-3.5-turbo"
        )
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 Results will be saved to: {OUTPUT_DIR}")
    
    # Load data with specified number of questions
    queries, answers, doc_ids, doc_texts, doc_titles = load_data(num_questions)
    
    # Run BM25 retrieval (FAST!)
    retrieval_results = run_bm25_retrieval(queries, doc_ids, doc_texts)
    
    # Run generation for each k value
    results_summary = {}
    
    for k in K_VALUES:
        em_score = run_generation_for_k(
            k=k,
            queries=queries,
            answers=answers,
            retrieval_results=retrieval_results,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            llm_engine=llm_engine,
            checkpoint_dir=OUTPUT_DIR
        )
        results_summary[k] = em_score
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 EXPERIMENT 1 RESULTS SUMMARY (BM25)")
    print("="*60)
    print(f"\nDataset: MusiqueQA (first {num_questions} questions)")
    print(f"Retriever: BM25 (lexical)")
    print(f"LLM: {'Llama 3.1 8B (Groq)' if groq_keys_exist else 'GPT-3.5-turbo (OpenAI)'}")
    print(f"\nExact Match Scores:")
    print("-" * 30)
    for k, em in results_summary.items():
        print(f"  k={k}: {em:.4f} ({em*100:.2f}%)")
    print("-" * 30)
    
    # Save summary
    suffix = "_quick" if args.quick else ""
    summary_path = os.path.join(OUTPUT_DIR, f"experiment1_bm25_summary{suffix}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "Experiment 1: Standard RAG Baseline (BM25)",
            "num_questions": num_questions,
            "retriever": "BM25 (rank-bm25)",
            "llm": "llama-3.1-8b-instant" if groq_keys_exist else "gpt-3.5-turbo",
            "k_values": K_VALUES,
            "results": results_summary,
            "quick_mode": args.quick,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Summary saved: {summary_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
