import os
import sys
import json
import time
import re
import random
import argparse
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dotenv import load_dotenv

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

from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.retriever.dense.Contriever import Contriever
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of questions to evaluate 
NUM_QUESTIONS = 1200

QUICK_MODE_QUESTIONS = 3

K_VALUES = [1, 3, 5]

# Total documents per query 
TOTAL_DOCS_VALUES = [6, 9, 12]

TOP_K_RETRIEVAL = 100

CACHE_DIR = os.path.join(project_root, "experiments", "cache")

MIN_DELAY_BETWEEN_REQUESTS = 0.1
CHECKPOINT_FREQUENCY = 50
OUTPUT_DIR = os.path.join(project_root, "experiments", "results")

# Random seed for reproducibility
RANDOM_SEED = 42

# Maximum total documents per query
MAX_TOTAL_DOCS = 35

DEBUG_MODE = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(num_questions: int = NUM_QUESTIONS):
    print("\nLoading dataset and corpus...")
    print("   This may take a few minutes for the large corpus...")
    
    # Load dev.json directly
    dev_path = os.path.join(project_root, "data", "dev.json")
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    print(f"   Dev questions loaded: {len(dev_data)}")
    
    # Load corpus
    corpus_path = os.path.join(project_root, "wiki_musique_corpus.json")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_raw = json.load(f)
    
    print(f"   Corpus loaded: {len(corpus_raw)} documents")
    
    # Convert corpus to list of Evidence objects (required by Contriever)
    from dexter.data.datastructures.evidence import Evidence
    
    corpus_list = []
    corpus_dict = {}
    all_doc_ids = []  # For sampling
    title_to_doc_ids = {}  # Map title to doc IDs for golden document lookup
    
    print("   Converting corpus to Evidence objects...")
    
    if TQDM_AVAILABLE:
        corpus_items = tqdm(corpus_raw.items(), desc="   Converting", unit="doc", ncols=80)
    else:
        corpus_items = corpus_raw.items()
    
    for doc_id, doc_data in corpus_items:
        title = doc_data.get("title", "")
        text = doc_data.get("text", "")
        evidence = Evidence(text=text, idx=doc_id, title=title)
        corpus_list.append(evidence)
        corpus_dict[doc_id] = evidence
        all_doc_ids.append(doc_id)
        
        # Build title -> doc_id mapping for golden doc lookup
        if title:
            if title not in title_to_doc_ids:
                title_to_doc_ids[title] = []
            title_to_doc_ids[title].append(doc_id)
    
    print(f"   Corpus converted: {len(corpus_list)} Evidence objects")
    print(f"   Title-to-ID mapping: {len(title_to_doc_ids)} unique titles")
    
    # Prepare queries and answers, extract golden doc IDs
    from dexter.data.datastructures.question import Question
    from dexter.data.datastructures.answer import Answer
    
    queries = []
    answers = []
    qrels = {}
    golden_docs_per_query = {}  # Store golden doc IDs per question
    
    total_golden_found = 0
    
    for item in dev_data[:num_questions]:
        qid = item["_id"]
        question = Question(item["question"], idx=qid)
        answer = Answer(item["answer"])
        
        queries.append(question)
        answers.append(answer)
        qrels[str(qid)] = {}
        
        # Extract golden document titles from supporting_facts
        golden_doc_ids = set()
        supporting_facts = item.get("supporting_facts", [])
        
        for sf in supporting_facts:
            if isinstance(sf, list) and len(sf) >= 1:
                title = sf[0]  # First element is the title
                # Look up doc IDs by title
                if title in title_to_doc_ids:
                    for doc_id in title_to_doc_ids[title]:
                        golden_doc_ids.add(doc_id)
        
        golden_docs_per_query[str(qid)] = golden_doc_ids
        if golden_doc_ids:
            total_golden_found += 1
    
    print(f"   Using first {num_questions} questions for evaluation")
    print(f"   Golden documents found for {total_golden_found}/{num_questions} questions")
    
    return queries, qrels, corpus_list, corpus_dict, answers, dev_data[:num_questions], all_doc_ids, golden_docs_per_query


# ============================================================================
# RETRIEVAL
# ============================================================================

def get_cache_path(corpus_size: int) -> str:
    return os.path.join(CACHE_DIR, f"contriever_corpus_{corpus_size}.pt")


def load_cached_embeddings(corpus_size: int):
    cache_path = get_cache_path(corpus_size)
    if os.path.exists(cache_path):
        print(f"   Loading cached embeddings from: {cache_path}")
        try:
            cached_data = torch.load(cache_path, weights_only=False)
            print(f"   Loaded cached embeddings: {cached_data['embeddings'].shape}")
            return cached_data
        except Exception as e:
            print(f"   Failed to load cache: {e}")
    return None


def run_contriever_retrieval(queries: List, corpus_list: List, qrels: Dict, use_cache: bool = True) -> Dict:
    print("\nRunning Contriever Retrieval...")
    print(f"   Retrieving top-{TOP_K_RETRIEVAL} documents for {len(queries)} queries")
    print(f"   Corpus size: {len(corpus_list)} documents")
    
    config_instance = DenseHyperParams(
        query_encoder_path="facebook/contriever",
        document_encoder_path="facebook/contriever",
        batch_size=16
    )
    
    contriever = Contriever(config_instance)
    similarity_measure = CosScore()
    
    corpus_size = len(corpus_list)
    corpus_ids = [doc.id() for doc in corpus_list]
    
    cached_data = None
    if use_cache:
        cached_data = load_cached_embeddings(corpus_size)
    
    start_time = time.time()
    
    if cached_data is not None and cached_data['corpus_size'] == corpus_size:
        print("   Using cached corpus embeddings...")
        corpus_embeddings = cached_data['embeddings']
        
        query_embeddings = contriever.encode_queries(queries, batch_size=config_instance.batch_size)
        
        device = query_embeddings.device
        corpus_embeddings = corpus_embeddings.to(device)
        
        cos_scores = similarity_measure.evaluate(query_embeddings, corpus_embeddings)
        
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores, 
            min(TOP_K_RETRIEVAL + 1, len(cos_scores[1])), 
            dim=1, 
            largest=True, 
            sorted=True
        )
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        response = {}
        for idx, q in enumerate(queries):
            response[q.id()] = {}
            for index, doc_idx in enumerate(cos_scores_top_k_idx[idx]):
                response[q.id()][corpus_ids[doc_idx]] = float(cos_scores_top_k_values[idx][index])
    else:
        print("   Encoding corpus and queries (this takes a while)...")
        
        corpus_embeddings = contriever.encode_corpus(corpus_list)
        
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_path = get_cache_path(corpus_size)
            print(f"   Saving embeddings to cache: {cache_path}")
            torch.save({
                'embeddings': corpus_embeddings.cpu(),
                'corpus_ids': corpus_ids,
                'corpus_size': corpus_size
            }, cache_path)
        
        query_embeddings = contriever.encode_queries(queries, batch_size=config_instance.batch_size)
        
        cos_scores = similarity_measure.evaluate(query_embeddings, corpus_embeddings)
        
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores, 
            min(TOP_K_RETRIEVAL + 1, len(cos_scores[1])), 
            dim=1, 
            largest=True, 
            sorted=True
        )
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        response = {}
        for idx, q in enumerate(queries):
            response[q.id()] = {}
            for index, doc_idx in enumerate(cos_scores_top_k_idx[idx]):
                response[q.id()][corpus_ids[doc_idx]] = float(cos_scores_top_k_values[idx][index])
    
    elapsed = time.time() - start_time
    print(f"   Retrieval complete in {elapsed:.1f}s")
    
    return response


# ============================================================================
# HARD NEGATIVES SELECTION
# ============================================================================

def get_hard_negatives(
    num_hard_negs: int,
    top_k: int,
    retrieved_doc_ids: List[str],
    golden_doc_ids: Set[str],
    corpus: Dict,
    debug: bool = False
) -> Tuple[List[str], List[str]]:
    hard_neg_texts = []
    hard_neg_ids = []
    
    # Start from position k 
    candidate_start = top_k
    
    if debug:
        print(f"\n      [DEBUG] Selecting hard negatives:")
        print(f"      - Need: {num_hard_negs} hard negatives")
        print(f"      - Start position: {candidate_start} (after top-{top_k})")
        print(f"      - Golden doc IDs: {golden_doc_ids}")
        print(f"      - Total retrieved: {len(retrieved_doc_ids)}")
    
    candidates_checked = 0
    for i in range(candidate_start, len(retrieved_doc_ids)):
        if len(hard_neg_ids) >= num_hard_negs:
            break
            
        doc_id = retrieved_doc_ids[i]
        candidates_checked += 1
        
        # Skip if this is a golden document
        if doc_id in golden_doc_ids:
            if debug:
                print(f"      - Position {i}: {doc_id[:30]}... SKIPPED (golden doc)")
            continue
        
        # Get document text
        if doc_id in corpus:
            doc = corpus[doc_id]
            if hasattr(doc, 'text'):
                text = doc.text() if callable(doc.text) else doc.text
            else:
                text = str(doc)
            
            hard_neg_texts.append(text)
            hard_neg_ids.append(doc_id)
            
            if debug:
                print(f"      - Position {i}: {doc_id[:30]}... SELECTED as hard negative")
    
    if debug:
        print(f"      - Candidates checked: {candidates_checked}")
        print(f"      - Hard negatives selected: {len(hard_neg_ids)}")
    
    return hard_neg_texts, hard_neg_ids


def get_context_with_hard_negatives(
    query_id: str,
    retrieval_results: Dict,
    corpus: Dict,
    golden_doc_ids: Set[str],
    k: int,
    total_docs: int,
    debug: bool = False
) -> Tuple[str, int, int, List[str], List[str]]:
    if query_id not in retrieval_results:
        return "", 0, 0, [], []
    
    # Get all retrieved doc IDs sorted by score 
    doc_scores = retrieval_results[query_id]
    # Sort by score descending to ensure consistent ordering
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    retrieved_doc_ids = [doc_id for doc_id, score in sorted_docs]
    
    # Get top-k relevant doc IDs
    top_k_doc_ids = retrieved_doc_ids[:k]
    
    if debug:
        print(f"\n   [DEBUG] Query: {query_id}")
        print(f"   [DEBUG] Top-{k} retrieved docs (relevant):")
        for i, doc_id in enumerate(top_k_doc_ids):
            score = doc_scores[doc_id]
            is_golden = "GOLDEN" if doc_id in golden_doc_ids else ""
            print(f"      {i+1}. {doc_id[:40]}... (score: {score:.4f}) {is_golden}")
    
    # Get relevant document texts
    relevant_texts = []
    relevant_doc_ids_actual = []
    for doc_id in top_k_doc_ids:
        if doc_id in corpus:
            doc = corpus[doc_id]
            if hasattr(doc, 'text'):
                text = doc.text() if callable(doc.text) else doc.text
            else:
                text = str(doc)
            relevant_texts.append(text)
            relevant_doc_ids_actual.append(doc_id)
    
    # Calculate number of hard negatives needed to reach total_docs
    num_hard_negs = total_docs - len(relevant_texts)
    if num_hard_negs < 0:
        num_hard_negs = 0
    
    # Cap total documents to avoid exceeding context limits
    if total_docs > MAX_TOTAL_DOCS:
        num_hard_negs = MAX_TOTAL_DOCS - len(relevant_texts)
        if num_hard_negs < 0:
            num_hard_negs = 0
    
    # Get hard negatives from positions k+1 onwards, excluding golden docs
    hard_neg_texts, hard_neg_ids = get_hard_negatives(
        num_hard_negs=num_hard_negs,
        top_k=k,
        retrieved_doc_ids=retrieved_doc_ids,
        golden_doc_ids=golden_doc_ids,
        corpus=corpus,
        debug=debug
    )
    
    # Combine documents: hard negatives first, then relevant docs
    all_texts = hard_neg_texts + relevant_texts
    
    combined_context = "\n\n".join(all_texts)
    
    if debug:
        print(f"\n   [DEBUG] Final document composition:")
        print(f"      - Relevant docs (top-k): {len(relevant_texts)}")
        print(f"      - Hard negatives: {len(hard_neg_texts)}")
        print(f"      - Total docs in context: {len(all_texts)}")
        print(f"      - Context length: {len(combined_context)} chars")
    
    return combined_context, len(relevant_texts), len(hard_neg_texts), relevant_doc_ids_actual, hard_neg_ids


# ============================================================================
# GENERATION WITH LLM
# ============================================================================


def run_generation_with_hard_negatives(
    k: int,
    total_docs: int,
    queries: List,
    answers: List,
    retrieval_results: Dict,
    corpus: Dict,
    golden_docs_per_query: Dict,
    llm_engine,
    checkpoint_dir: str,
    debug: bool = False,
    raw_data: List[Dict] = None
) -> Dict:
    config_name = f"k{k}_total{total_docs}"
    num_hard_negs = total_docs - k
    
    print(f"\n{'='*60}")
    print(f"GENERATION: k={k}, total_docs={total_docs}")
    print(f"   ({k} relevant + {num_hard_negs} hard negatives = {total_docs} total docs per query)")
    print(f"{'='*60}")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"hardneg_checkpoint_{config_name}.json")
    
    # Use new checkpoint validation to find missing questions
    indices_to_process, matches, total_processed, results, checkpoint = get_indices_to_process(
        checkpoint_path, len(queries), verbose=True
    )
    
    # Track golden document stats
    all_gold_retrieved = 0  # Count of questions where all gold docs are in top-k
    all_gold_correct = 0  # Count of correct answers when all gold docs are in top-k
    gold_recall_scores = []  # For computing average gold doc recall
    
    # If all questions are already processed, just return the existing results
    if not indices_to_process:
        em_score = matches / total_processed if total_processed > 0 else 0
        print(f"   All questions already processed. EM: {em_score:.4f}")
        return {
            "k": k,
            "total_docs": total_docs,
            "num_hard_negs": num_hard_negs,
            "exact_match": em_score,
            "matches": matches,
            "total": total_processed,
            "errors": 0,
            "all_gold_retrieved_count": 0,
            "all_gold_correct_count": 0,
            "gold_recall_rate": 0.0,
            "gold_em_when_retrieved": 0.0
        }
    
    errors = 0
    processed_count = 0
    
    use_tqdm = TQDM_AVAILABLE and len(indices_to_process) > 0 and not debug
    if use_tqdm:
        iterator = tqdm(
            indices_to_process,
            desc=f"   k={k},t={total_docs}",
            unit="q",
            ncols=85,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        iterator = indices_to_process
    
    for idx in iterator:
        query = queries[idx]
        gold_answer = answers[idx]
        
        question_id = query.id() if hasattr(query, 'id') else str(idx)
        question_text = query.text() if hasattr(query, 'text') else str(query)
        gold_text = gold_answer.text() if hasattr(gold_answer, 'text') else str(gold_answer)
        
        # Get golden document IDs for this query
        golden_doc_ids = golden_docs_per_query.get(str(question_id), set())
        
        if debug:
            print(f"\n{'='*60}")
            print(f"   Question {idx+1}: {question_text[:80]}...")
            print(f"   Gold answer: {gold_text}")
        
        # Get context with hard negatives
        context, num_relevant, num_hard_negs_actual, rel_ids, hard_neg_ids = get_context_with_hard_negatives(
            str(question_id),
            retrieval_results,
            corpus,
            golden_doc_ids,
            k,
            total_docs,
            debug=debug
        )
        
        if not context:
            if TQDM_AVAILABLE and not debug:
                tqdm.write(f"   No context for question {idx+1}, skipping")
            else:
                print(f"   No context for question {idx+1}, skipping")
            errors += 1
            continue
        
        # Compute golden doc stats - how many gold docs in top-k retrieved
        gold_in_top_k = len(set(rel_ids) & golden_doc_ids) if rel_ids else 0
        gold_total = len(golden_doc_ids)
        all_gold_in_topk = gold_in_top_k >= gold_total if gold_total > 0 else True
        gold_recall_scores.append(gold_in_top_k / gold_total if gold_total > 0 else 0)
        
        # Create prompt
        system_prompt, user_prompt = create_rag_prompt(question_text, context)
        
        try:
            # Generate answer
            response = llm_engine.get_chat_completion(user_prompt, system_prompt)
            
            # Extract and evaluate
            predicted = extract_final_answer(response)
            is_match = cover_exact_match(predicted, gold_text)
            
            if debug:
                print(f"\n   [DEBUG] LLM Response:")
                print(f"      Predicted: {predicted[:100]}...")
                print(f"      Match: {'YES' if is_match else 'NO'}")
            
            # Track golden doc stats
            if all_gold_in_topk:
                all_gold_retrieved += 1
                if is_match:
                    all_gold_correct += 1
            
            # Create result entry
            new_result = {
                "idx": idx,
                "question_id": question_id,
                "question": question_text,
                "gold_answer": gold_text,
                "predicted": predicted,
                "is_match": is_match,
                "num_relevant_docs": num_relevant,
                "num_hard_neg_docs": num_hard_negs_actual,
                "relevant_doc_ids": rel_ids,
                "hard_neg_doc_ids": hard_neg_ids,
                "gold_in_top_k": gold_in_top_k,
                "gold_total": gold_total,
                "all_gold_retrieved": all_gold_in_topk,
                "full_response": response[:500]
            }
            
            if not checkpoint:
                checkpoint = {"k": k, "total_docs": total_docs, "results": []}
            checkpoint = merge_result_into_checkpoint(checkpoint, new_result)
            results = checkpoint.get("results", [])
            matches = checkpoint.get("matches", 0)
            total_processed = checkpoint.get("total", 0)
            
            processed_count += 1
            
            if use_tqdm and hasattr(iterator, 'set_postfix') and not debug:
                current_em = matches / total_processed if total_processed > 0 else 0
                iterator.set_postfix(EM=f"{current_em:.3f}", correct=matches)
            elif (not TQDM_AVAILABLE or debug) and processed_count % 10 == 0:
                current_em = matches / total_processed if total_processed > 0 else 0
                print(f"   [Processed {processed_count}] EM: {current_em:.4f} ({matches}/{total_processed})")
            
            if processed_count % CHECKPOINT_FREQUENCY == 0:
                checkpoint["k"] = k
                checkpoint["total_docs"] = total_docs
                save_checkpoint(checkpoint_path, checkpoint, silent=TQDM_AVAILABLE and not debug)
                if TQDM_AVAILABLE and not debug:
                    tqdm.write(f"  Checkpoint saved: {checkpoint_path}")
            
            time.sleep(MIN_DELAY_BETWEEN_REQUESTS)
            
        except Exception as e:
            errors += 1
            error_msg = str(e)
            if TQDM_AVAILABLE and not debug:
                tqdm.write(f"   Error at question {idx+1}: {error_msg[:80]}")
            else:
                print(f"   Error at question {idx+1}: {error_msg}")
            time.sleep(1)
    
    if checkpoint:
        checkpoint["k"] = k
        checkpoint["total_docs"] = total_docs
        save_checkpoint(checkpoint_path, checkpoint, silent=True)
    
    matches = checkpoint.get("matches", 0) if checkpoint else matches
    total_processed = checkpoint.get("total", 0) if checkpoint else total_processed
    results = checkpoint.get("results", []) if checkpoint else results
    em_score = matches / total_processed if total_processed > 0 else 0
    
    gold_recall_rate = sum(gold_recall_scores) / len(gold_recall_scores) if gold_recall_scores else 0
    gold_em_when_retrieved = all_gold_correct / all_gold_retrieved if all_gold_retrieved > 0 else 0
    
    final_results = {
        "k": k,
        "total_docs": total_docs,
        "num_hard_negs": num_hard_negs,
        "exact_match": em_score,
        "matches": matches,
        "total": total_processed,
        "errors": errors,
        "all_gold_retrieved_count": all_gold_retrieved,
        "all_gold_correct_count": all_gold_correct,
        "gold_recall_rate": gold_recall_rate,
        "gold_em_when_retrieved": gold_em_when_retrieved,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    final_path = os.path.join(checkpoint_dir, f"hardneg_results_{config_name}.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n   k={k}, total_docs={total_docs} Complete!")
    print(f"   Exact Match: {em_score:.4f} ({matches}/{total_processed})")
    print(f"   Context: {k} relevant + {num_hard_negs} hard negatives = {total_docs} docs")
    print(f"   Gold Recall: {gold_recall_rate:.2%} | EM@Gold: {gold_em_when_retrieved:.4f}")
    print(f"   Errors: {errors}")
    print(f"   Results saved: {final_path}")
    
    return {
        "k": k,
        "total_docs": total_docs,
        "num_hard_negs": num_hard_negs,
        "exact_match": em_score,
        "matches": matches,
        "total": total_processed,
        "errors": errors,
        "all_gold_retrieved_count": all_gold_retrieved,
        "all_gold_correct_count": all_gold_correct,
        "gold_recall_rate": gold_recall_rate,
        "gold_em_when_retrieved": gold_em_when_retrieved
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Experiment 4: Hard Negatives RAG')
    parser.add_argument('--quick', action='store_true', 
                        help=f'Quick mode: only process first {QUICK_MODE_QUESTIONS} questions for validation')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable corpus embedding caching')
    args = parser.parse_args()
    
    print("="*60)
    print("EXPERIMENT 4: Hard Negatives RAG")
    print("(Impact of Semantically Similar Non-Relevant Documents)")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        print(f"\nQUICK MODE: Processing {QUICK_MODE_QUESTIONS} questions for validation")
        num_questions = QUICK_MODE_QUESTIONS
    else:
        num_questions = NUM_QUESTIONS
    
    # Print experiment configuration
    print(f"\nConfiguration:")
    print(f"   K values: {K_VALUES}")
    print(f"   Total docs per query: {TOTAL_DOCS_VALUES}")
    print(f"   Total configurations: {len(K_VALUES) * len(TOTAL_DOCS_VALUES)}")
    print(f"   Random seed: {RANDOM_SEED}")
    print(f"   Max total docs per query: {MAX_TOTAL_DOCS} (to avoid context limits)")
    
    # Show document breakdown for each config
    print(f"\nDocument breakdown:")
    for total_docs in TOTAL_DOCS_VALUES:
        breakdown = []
        for k in K_VALUES:
            hard_negs = total_docs - k
            breakdown.append(f"k={k}→{hard_negs} hard negs")
        print(f"   Total {total_docs}: {', '.join(breakdown)}")
    
    # Check for Groq API key
    paid_api_key = os.environ.get("PAID_API_KEY", "")
    use_openai = os.environ.get("OPENAI_KEY") and os.environ.get("OPENAI_KEY") != "your_openai_api_key_here"
    
    if (not paid_api_key or paid_api_key == "your_paid_groq_api_key_here") and not use_openai:
        print("\nERROR: No LLM API key configured!")
        print("Set PAID_API_KEY for Groq or OPENAI_KEY for OpenAI in .env file")
        return
    
    # Initialize LLM
    if paid_api_key and paid_api_key != "your_paid_groq_api_key_here":
        from dexter.llms.groq_engine import GroqEngine
        print("\nUsing Groq with Llama 3.1 8B Instant")
        llm_engine = GroqEngine(
            data="",
            model_name="llama-3.1-8b-instant",
            temperature=0.3
        )
    else:
        print("\nUsing OpenAI (PAID) with GPT-3.5-turbo")
        from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
        orchestrator = LLMEngineOrchestrator()
        llm_engine = orchestrator.get_llm_engine(
            data="",
            llm_class="openai",
            model_name="gpt-3.5-turbo"
        )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nResults will be saved to: {OUTPUT_DIR}")
    
    random.seed(RANDOM_SEED)
    
    queries, qrels, corpus_list, corpus_dict, answers, raw_data, all_doc_ids, golden_docs_per_query = load_data(num_questions)
    
    print(f"\nCorpus Statistics:")
    print(f"   Total documents in corpus: {len(all_doc_ids)}")
    print(f"   Questions with golden docs: {sum(1 for q in golden_docs_per_query if golden_docs_per_query[q])}")
    
    use_cache = not args.no_cache
    retrieval_results = run_contriever_retrieval(queries, corpus_list, qrels, use_cache=use_cache)
    
    results_summary = {}
    config_count = 0
    total_configs = len(K_VALUES) * len(TOTAL_DOCS_VALUES)
    
    for k in K_VALUES:
        for total_docs in TOTAL_DOCS_VALUES:
            config_count += 1
            num_hard_negs = total_docs - k
            print(f"\n{'#'*60}")
            print(f"# Configuration {config_count}/{total_configs}: k={k}, total_docs={total_docs} ({k} rel + {num_hard_negs} hard negs)")
            print(f"{'#'*60}")
            
            result = run_generation_with_hard_negatives(
                k=k,
                total_docs=total_docs,
                queries=queries,
                answers=answers,
                retrieval_results=retrieval_results,
                corpus=corpus_dict,
                golden_docs_per_query=golden_docs_per_query,
                llm_engine=llm_engine,
                checkpoint_dir=OUTPUT_DIR,
                debug=False,
                raw_data=raw_data
            )
            
            config_key = f"k{k}_total{total_docs}"
            results_summary[config_key] = result
    
    # Print final summary
    print("\n" + "="*90)
    print("EXPERIMENT 4 RESULTS SUMMARY")
    print("="*90)
    print(f"\nDataset: MusiqueQA (first {num_questions} questions)")
    print(f"Retriever: Contriever (facebook/contriever)")
    print(f"LLM: {'Llama 3.1 8B (Groq)' if paid_api_key else 'GPT-3.5-turbo (OpenAI)'}")
    print(f"\nHard Negatives Injection Results:")
    print("-" * 90)
    print(f"{'Config':<18} {'k':<4} {'HNeg':<6} {'Total':<6} {'EM Score':<12} {'Gold Recall':<12} {'EM@Gold':<10}")
    print("-" * 90)
    
    for config_key, result in results_summary.items():
        k = result['k']
        total_docs = result['total_docs']
        num_hard_negs = result['num_hard_negs']
        em = result['exact_match']
        gold_rate = result.get('gold_recall_rate', 0)
        gold_em = result.get('gold_em_when_retrieved', 0)
        print(f"{config_key:<18} {k:<4} {num_hard_negs:<6} {total_docs:<6} {em:.4f} ({em*100:.1f}%)  {gold_rate:.2%}       {gold_em:.4f}")
    
    print("-" * 90)
    
    # Compare performance across total docs for each k
    print("\nPerformance by Total Documents:")
    print("-" * 60)
    for k in K_VALUES:
        print(f"\n  k={k} relevant docs:")
        for total_docs in TOTAL_DOCS_VALUES:
            config_key = f"k{k}_total{total_docs}"
            if config_key in results_summary:
                em = results_summary[config_key]['exact_match']
                gold_rate = results_summary[config_key].get('gold_recall_rate', 0)
                num_hard_negs = total_docs - k
                print(f"    {total_docs} total ({num_hard_negs} hard negs): EM={em:.4f} | GoldRecall={gold_rate:.2%}")
    
    suffix = "_quick" if args.quick else ""
    summary_path = os.path.join(OUTPUT_DIR, f"experiment4_hardneg_summary{suffix}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "Experiment 4: Hard Negatives RAG",
            "description": "Impact of semantically similar non-relevant documents on LLM performance",
            "num_questions": num_questions,
            "retriever": "Contriever (facebook/contriever)",
            "llm": "llama-3.1-8b-instant" if paid_api_key else "gpt-3.5-turbo",
            "k_values": K_VALUES,
            "total_docs_values": TOTAL_DOCS_VALUES,
            "random_seed": RANDOM_SEED,
            "results": results_summary,
            "quick_mode": args.quick,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    print(f"\nExperiment 4 Complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
