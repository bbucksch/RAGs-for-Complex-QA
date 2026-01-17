import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Import Groq LLM
from dexter.llms.groq_engine import GroqEngine


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of questions to evaluate 
NUM_QUESTIONS = 1200


QUICK_MODE_QUESTIONS = 50

K_VALUES = [1, 3, 5]

MIN_DELAY_BETWEEN_REQUESTS = 0.1

CHECKPOINT_FREQUENCY = 50

# For --best mode: retrieve top N documents to find golden doc rankings
BEST_MODE_TOP_K = 100

# Cache directory for embeddings
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


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


# ============================================================================
# CORPUS AND RETRIEVAL FOR --best MODE
# ============================================================================

def load_corpus_and_data(num_questions: int) -> Tuple[List[Dict], Dict, Dict, List]:
    """Load dev data and corpus for retrieval-based best oracle mode."""
    import torch
    from dexter.data.datastructures.evidence import Evidence
    from dexter.data.datastructures.question import Question
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load dev data
    dev_path = os.path.join(base_path, "data", "dev.json")
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # Load corpus
    corpus_path = os.path.join(base_path, "wiki_musique_corpus.json")
    print(f"   Loading corpus from: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_raw = json.load(f)
    print(f"   Corpus loaded: {len(corpus_raw)} documents")
    
    # Build title -> doc_ids mapping
    title_to_docids = {}
    for doc_id, doc_data in corpus_raw.items():
        title = doc_data.get("title", "")
        if title not in title_to_docids:
            title_to_docids[title] = []
        title_to_docids[title].append(doc_id)
    
    # Convert corpus to Evidence objects
    corpus_list = []
    corpus_dict = {}
    
    for doc_id, doc_data in corpus_raw.items():
        title = doc_data.get("title", "")
        text = doc_data.get("text", "")
        evidence = Evidence(text=text, idx=doc_id, title=title)
        corpus_list.append(evidence)
        corpus_dict[doc_id] = evidence
    
    # Prepare queries
    queries = []
    for item in dev_data[:num_questions]:
        qid = item["_id"]
        question = Question(item["question"], idx=qid)
        queries.append(question)
    
    return dev_data[:num_questions], corpus_list, corpus_dict, queries, title_to_docids


def get_cache_path(corpus_size: int) -> str:
    """Get path to cached embeddings."""
    return os.path.join(CACHE_DIR, f"contriever_corpus_{corpus_size}.pt")


def load_cached_embeddings(corpus_size: int):
    """Load cached corpus embeddings if available."""
    import torch
    cache_path = get_cache_path(corpus_size)
    if os.path.exists(cache_path):
        print(f"   Loading cached embeddings from: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    return None


def run_retrieval_for_best_mode(queries: List, corpus_list: List) -> Dict:
    """Run Contriever retrieval to get top-100 documents per query."""
    import torch
    from dexter.retriever.dense.Contriever import Contriever
    from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
    from dexter.utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
    
    print(f"\n   Running Contriever retrieval for --best mode")
    print(f"   Retrieving top-{BEST_MODE_TOP_K} documents for {len(queries)} queries")
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
    
    # Try to load cached embeddings
    cached_data = load_cached_embeddings(corpus_size)
    
    start_time = time.time()
    
    if cached_data is not None and cached_data['corpus_size'] == corpus_size:
        print("   Using cached corpus embeddings...")
        corpus_embeddings = cached_data['embeddings']
        
        # Encode queries
        query_embeddings = contriever.encode_queries(queries, batch_size=config_instance.batch_size)
        
        # Move corpus embeddings to same device
        device = query_embeddings.device
        corpus_embeddings = corpus_embeddings.to(device)
        
        # Compute similarity scores
        cos_scores = similarity_measure.evaluate(query_embeddings, corpus_embeddings)
        
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores, 
            min(BEST_MODE_TOP_K + 1, len(cos_scores[1])), 
            dim=1, 
            largest=True, 
            sorted=True
        )
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        # Build response dictionary
        response = {}
        for idx, q in enumerate(queries):
            response[q.id()] = {}
            for index, doc_idx in enumerate(cos_scores_top_k_idx[idx]):
                response[q.id()][corpus_ids[doc_idx]] = float(cos_scores_top_k_values[idx][index])
    else:
        print("   Encoding corpus (this takes a while)...")
        corpus_embeddings = contriever.encode_corpus(corpus_list)
        
        # Save cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = get_cache_path(corpus_size)
        print(f"   Saving embeddings to cache: {cache_path}")
        torch.save({
            'embeddings': corpus_embeddings.cpu(),
            'corpus_ids': corpus_ids,
            'corpus_size': corpus_size
        }, cache_path)
        
        # Encode queries
        query_embeddings = contriever.encode_queries(queries, batch_size=config_instance.batch_size)
        
        # Compute similarity scores
        cos_scores = similarity_measure.evaluate(query_embeddings, corpus_embeddings)
        
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores, 
            min(BEST_MODE_TOP_K + 1, len(cos_scores[1])), 
            dim=1, 
            largest=True, 
            sorted=True
        )
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        # Build response dictionary
        response = {}
        for idx, q in enumerate(queries):
            response[q.id()] = {}
            for index, doc_idx in enumerate(cos_scores_top_k_idx[idx]):
                response[q.id()][corpus_ids[doc_idx]] = float(cos_scores_top_k_values[idx][index])
    
    elapsed = time.time() - start_time
    print(f"   Retrieval complete in {elapsed:.1f}s")
    
    return response


def get_gold_doc_ids(item: Dict, title_to_docids: Dict, corpus_dict: Dict) -> set:
    """Get corpus document IDs that match the specific golden paragraphs (by title and sentence index)."""
    gold_ids = set()
    supporting_facts = item.get("supporting_facts", [])
    
    for fact in supporting_facts:
        if isinstance(fact, list) and len(fact) >= 2:
            title = fact[0]
            sentence_idx = fact[1]
            
            if title in title_to_docids:
                # Find the specific paragraph by sentence index
                doc_ids = title_to_docids[title]
                if sentence_idx < len(doc_ids):
                    # Match by index in the list (paragraphs are ordered)
                    gold_ids.add(doc_ids[sentence_idx])
                else:
                    # Fallback: sentence_idx out of range, add all (shouldn't happen)
                    gold_ids.update(doc_ids)
    
    return gold_ids


def extract_best_oracle_contexts(
    question_data: Dict, 
    retrieval_results: Dict,
    corpus_dict: Dict,
    title_to_docids: Dict,
    k: int = None
) -> Tuple[List[str], Dict]:
    """
    Extract oracle contexts ordered by their retrieval ranking.
    
    Returns:
        - List of oracle context strings (ordered by retrieval rank)
        - Dict with statistics about golden doc rankings
    """
    question_id = question_data['_id']
    
    # Get gold document IDs (specific paragraphs, not all paragraphs from article)
    gold_doc_ids = get_gold_doc_ids(question_data, title_to_docids, corpus_dict)
    
    if not gold_doc_ids:
        return [], {"num_gold_docs": 0, "gold_ranks": [], "gold_found_in_top100": 0}
    
    # Get retrieval results for this question
    query_results = retrieval_results.get(question_id, {})
    
    # Sort by score (descending)
    sorted_docs = sorted(query_results.items(), key=lambda x: x[1], reverse=True)
    
    # Find golden documents and their ranks
    gold_docs_with_ranks = []
    for rank, (doc_id, score) in enumerate(sorted_docs):
        if doc_id in gold_doc_ids:
            gold_docs_with_ranks.append((doc_id, rank, score))
    
    # Sort gold docs by their retrieval rank (best first)
    gold_docs_with_ranks.sort(key=lambda x: x[1])
    
    # Extract context texts for gold docs in rank order
    oracle_contexts = []
    gold_ranks = []
    
    for doc_id, rank, score in gold_docs_with_ranks:
        if doc_id in corpus_dict:
            doc = corpus_dict[doc_id]
            if hasattr(doc, 'text'):
                text = doc.text() if callable(doc.text) else doc.text
            else:
                text = str(doc)
            
            if hasattr(doc, 'title'):
                title = doc.title() if callable(doc.title) else doc.title
                oracle_contexts.append(f"{title}: {text}")
            else:
                oracle_contexts.append(text)
            
            gold_ranks.append(rank)
    
    # Limit to k if specified
    if k is not None and k < len(oracle_contexts):
        oracle_contexts = oracle_contexts[:k]
        gold_ranks = gold_ranks[:k]
    
    stats = {
        "num_gold_docs": len(gold_doc_ids),
        "gold_ranks": gold_ranks,
        "gold_found_in_top100": len(gold_docs_with_ranks),
        "avg_gold_rank": sum(gold_ranks) / len(gold_ranks) if gold_ranks else -1
    }
    
    return oracle_contexts, stats


# ============================================================================
# STANDARD ORACLE FUNCTIONS
# ============================================================================

def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_path = os.path.join(base_path, "data", "dev.json")
    
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    return dev_data


def extract_oracle_contexts(question_data: Dict, k: int = None) -> List[str]:

    context_map = {}
    for title, sentences in question_data.get('context', []):
        context_map[title] = sentences
    
    oracle_contexts = []
    seen = set()  
    
    for title, sent_idx in question_data.get('supporting_facts', []):
        key = (title, sent_idx)
        if key in seen:
            continue
        seen.add(key)
        
        if title in context_map:
            sentences = context_map[title]
            if 0 <= sent_idx < len(sentences):
                paragraph_text = f"{title}: {sentences[sent_idx]}"
                oracle_contexts.append(paragraph_text)
    
    if k is not None and k < len(oracle_contexts):
        oracle_contexts = oracle_contexts[:k]
    
    return oracle_contexts


def run_oracle_generation_for_k(
    k: int,
    questions: List[Dict],
    llm: GroqEngine,
    results_dir: str,
    k_offset: int = 0
) -> Dict[str, Any]:
    k_label = k if k is not None else "all"
    print(f"\n{'='*60}")
    print(f"GENERATION WITH k={k_label} ORACLE CONTEXTS")
    print(f"{'='*60}")
    
    errors = 0
    
    k_suffix = f"_k{k}" if k is not None else "_all"
    checkpoint_path = os.path.join(results_dir, f"oracle_checkpoint{k_suffix}.json")
    
    indices_to_process, matches, total, results_list, checkpoint = get_indices_to_process(
        checkpoint_path, len(questions), verbose=True
    )
    
    if not indices_to_process:
        em_score = matches / total if total > 0 else 0
        print(f"   All questions already processed. EM: {em_score:.4f}")
        return {
            "k": k,
            "exact_match": em_score,
            "matches": matches,
            "total": total,
            "errors": 0,
            "results": results_list
        }
    
    processed_count = 0
    
    use_tqdm = TQDM_AVAILABLE and len(indices_to_process) > 0
    if use_tqdm:
        iterator = tqdm(
            indices_to_process,
            desc=f"   k={k_label}",
            unit="q",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        iterator = indices_to_process
    
    for i in iterator:
        q_data = questions[i]
        question = q_data['question']
        answer = q_data['answer']
        
        # Extract oracle contexts
        oracle_contexts = extract_oracle_contexts(q_data, k)
        
        if not oracle_contexts:
            predicted = "I don't know"
        else:
            # Create prompt and generate
            system_prompt, user_prompt = create_rag_prompt(question, oracle_contexts)
            
            try:
                response = llm.get_chat_completion(user_prompt, system_prompt)
                predicted = extract_final_answer(response)
            except Exception as e:
                error_msg = str(e)
                if use_tqdm:
                    tqdm.write(f"   Error at question {i}: {error_msg[:50]}")
                else:
                    print(f"   Error at question {i}: {error_msg[:50]}")
                
                # Retry with increasing delays for transient API errors
                predicted = "ERROR"
                for retry in range(3):
                    wait_time = 2 * (retry + 1) 
                    time.sleep(wait_time)
                    try:
                        response = llm.get_chat_completion(user_prompt, system_prompt)
                        predicted = extract_final_answer(response)
                        break 
                    except Exception as retry_e:
                        if retry == 2:  
                            errors += 1
                            print(f"   Failed after 3 retries: {str(retry_e)[:50]}")
        
        # Evaluate
        is_correct = cover_exact_match(predicted, answer)
        
        # Create result entry
        new_result = {
            'idx': i,
            'question_id': q_data['_id'],
            'question': question,
            'answer': answer,
            'predicted': predicted,
            'num_oracle_contexts': len(oracle_contexts),
            'is_match': is_correct
        }
        
        if not checkpoint:
            checkpoint = {"k": k, "results": []}
        checkpoint = merge_result_into_checkpoint(checkpoint, new_result)
        results_list = checkpoint.get("results", [])
        matches = checkpoint.get("matches", 0)
        total = checkpoint.get("total", 0)
        
        processed_count += 1
        
        if use_tqdm and hasattr(iterator, 'set_postfix'):
            current_em = matches / total if total > 0 else 0
            iterator.set_postfix(EM=f"{current_em:.3f}", correct=matches)
        elif not TQDM_AVAILABLE and processed_count % 10 == 0:
            em_score = matches / total if total > 0 else 0
            print(f"   [Processed {processed_count}] EM: {em_score:.4f} ({matches}/{total})")
        
        if processed_count % CHECKPOINT_FREQUENCY == 0:
            checkpoint["k"] = k
            save_checkpoint(checkpoint_path, checkpoint, silent=TQDM_AVAILABLE)
            if TQDM_AVAILABLE:
                tqdm.write(f"  Checkpoint saved: {checkpoint_path}")
        
        time.sleep(MIN_DELAY_BETWEEN_REQUESTS)
    
    if checkpoint:
        checkpoint["k"] = k
        save_checkpoint(checkpoint_path, checkpoint, silent=True)
    
    matches = checkpoint.get("matches", 0) if checkpoint else matches
    total = checkpoint.get("total", 0) if checkpoint else total
    results_list = checkpoint.get("results", []) if checkpoint else results_list
    em_score = matches / total if total > 0 else 0
    
    final_results = {
        "k": k,
        "exact_match": em_score,
        "matches": matches,
        "total": total,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
        "results": results_list
    }
    
    final_path = os.path.join(results_dir, f"oracle_results{k_suffix}.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n   k={k_label} Complete!")
    print(f"   Exact Match: {em_score:.4f} ({matches}/{total})")
    print(f"   Errors: {errors}")
    print(f"   Results saved: {final_path}")
    
    return final_results


# ============================================================================
# BEST MODE: Oracle with Retrieval-based Ranking
# ============================================================================

def run_best_oracle_generation_for_k(
    k: int,
    questions: List[Dict],
    llm: GroqEngine,
    results_dir: str,
    retrieval_results: Dict,
    corpus_dict: Dict,
    title_to_docids: Dict
) -> Dict[str, Any]:
    """
    Run oracle generation but with golden documents ordered by retrieval ranking.
    This gives the LLM the best (highest-ranked) golden documents first.
    """
    k_label = k if k is not None else "all"
    print(f"\n{'='*60}")
    print(f"BEST ORACLE: k={k_label} (gold docs ordered by retrieval rank)")
    print(f"{'='*60}")
    
    errors = 0
    
    k_suffix = f"_k{k}" if k is not None else "_all"
    checkpoint_path = os.path.join(results_dir, f"best_oracle_checkpoint{k_suffix}.json")
    
    indices_to_process, matches, total, results_list, checkpoint = get_indices_to_process(
        checkpoint_path, len(questions), verbose=True
    )
    
    if not indices_to_process:
        em_score = matches / total if total > 0 else 0
        print(f"   All questions already processed. EM: {em_score:.4f}")
        return {
            "k": k,
            "exact_match": em_score,
            "matches": matches,
            "total": total,
            "errors": 0,
            "results": results_list
        }
    
    processed_count = 0
    total_gold_found = 0
    total_gold_possible = 0
    rank_sum = 0
    rank_count = 0
    
    use_tqdm = TQDM_AVAILABLE and len(indices_to_process) > 0
    if use_tqdm:
        iterator = tqdm(
            indices_to_process,
            desc=f"   k={k_label}",
            unit="q",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        iterator = indices_to_process
    
    for i in iterator:
        q_data = questions[i]
        question = q_data['question']
        answer = q_data['answer']
        
        # Extract best oracle contexts (ordered by retrieval rank)
        oracle_contexts, gold_stats = extract_best_oracle_contexts(
            q_data, retrieval_results, corpus_dict, title_to_docids, k
        )
        
        # Track statistics
        total_gold_possible += gold_stats['num_gold_docs']
        total_gold_found += gold_stats['gold_found_in_top100']
        for rank in gold_stats['gold_ranks']:
            rank_sum += rank
            rank_count += 1
        
        if not oracle_contexts:
            predicted = "I don't know"
        else:
            system_prompt, user_prompt = create_rag_prompt(question, oracle_contexts)
            
            try:
                response = llm.get_chat_completion(user_prompt, system_prompt)
                predicted = extract_final_answer(response)
            except Exception as e:
                error_msg = str(e)
                if use_tqdm:
                    tqdm.write(f"   Error at question {i}: {error_msg[:50]}")
                else:
                    print(f"   Error at question {i}: {error_msg[:50]}")
                
                predicted = "ERROR"
                for retry in range(3):
                    wait_time = 2 * (retry + 1)
                    time.sleep(wait_time)
                    try:
                        response = llm.get_chat_completion(user_prompt, system_prompt)
                        predicted = extract_final_answer(response)
                        break
                    except Exception as retry_e:
                        if retry == 2:
                            errors += 1
                            print(f"   Failed after 3 retries: {str(retry_e)[:50]}")
        
        is_correct = cover_exact_match(predicted, answer)
        
        new_result = {
            'idx': i,
            'question_id': q_data['_id'],
            'question': question,
            'answer': answer,
            'predicted': predicted,
            'num_oracle_contexts': len(oracle_contexts),
            'is_match': is_correct,
            'gold_ranks': gold_stats['gold_ranks'],
            'gold_found_in_top100': gold_stats['gold_found_in_top100'],
            'num_gold_docs': gold_stats['num_gold_docs']
        }
        
        if not checkpoint:
            checkpoint = {"k": k, "mode": "best", "results": []}
        checkpoint = merge_result_into_checkpoint(checkpoint, new_result)
        results_list = checkpoint.get("results", [])
        matches = checkpoint.get("matches", 0)
        total = checkpoint.get("total", 0)
        
        processed_count += 1
        
        if use_tqdm and hasattr(iterator, 'set_postfix'):
            current_em = matches / total if total > 0 else 0
            iterator.set_postfix(EM=f"{current_em:.3f}", correct=matches)
        elif not TQDM_AVAILABLE and processed_count % 10 == 0:
            em_score = matches / total if total > 0 else 0
            print(f"   [Processed {processed_count}] EM: {em_score:.4f} ({matches}/{total})")
        
        if processed_count % CHECKPOINT_FREQUENCY == 0:
            checkpoint["k"] = k
            checkpoint["mode"] = "best"
            save_checkpoint(checkpoint_path, checkpoint, silent=TQDM_AVAILABLE)
            if TQDM_AVAILABLE:
                tqdm.write(f"  Checkpoint saved: {checkpoint_path}")
        
        time.sleep(MIN_DELAY_BETWEEN_REQUESTS)
    
    if checkpoint:
        checkpoint["k"] = k
        checkpoint["mode"] = "best"
        save_checkpoint(checkpoint_path, checkpoint, silent=True)
    
    matches = checkpoint.get("matches", 0) if checkpoint else matches
    total = checkpoint.get("total", 0) if checkpoint else total
    results_list = checkpoint.get("results", []) if checkpoint else results_list
    em_score = matches / total if total > 0 else 0
    avg_rank = rank_sum / rank_count if rank_count > 0 else -1
    
    final_results = {
        "k": k,
        "mode": "best",
        "exact_match": em_score,
        "matches": matches,
        "total": total,
        "errors": errors,
        "avg_gold_rank": avg_rank,
        "gold_found_rate": total_gold_found / total_gold_possible if total_gold_possible > 0 else 0,
        "timestamp": datetime.now().isoformat(),
        "results": results_list
    }
    
    final_path = os.path.join(results_dir, f"best_oracle_results{k_suffix}.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n   k={k_label} Complete!")
    print(f"   Exact Match: {em_score:.4f} ({matches}/{total})")
    print(f"   Avg Gold Rank: {avg_rank:.1f}")
    print(f"   Gold Found Rate: {total_gold_found}/{total_gold_possible} ({total_gold_found/total_gold_possible*100:.1f}%)")
    print(f"   Errors: {errors}")
    print(f"   Results saved: {final_path}")
    
    return final_results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Experiment 2: Oracle Context RAG')
    parser.add_argument('--quick', action='store_true', 
                        help=f'Quick mode: only process first {QUICK_MODE_QUESTIONS} questions')
    parser.add_argument('--best', action='store_true',
                        help='Best mode: retrieve top-100 docs and give best-ranked golden docs')
    args = parser.parse_args()
    
    if args.best:
        print("=" * 60)
        print("EXPERIMENT 2: Best Oracle RAG (Retrieval-Ranked Gold Docs)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXPERIMENT 2: Oracle Context RAG (Perfect Retrieval)")
        print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        print(f"\nQUICK MODE: Processing only {QUICK_MODE_QUESTIONS} questions")
        num_questions = QUICK_MODE_QUESTIONS
    else:
        num_questions = NUM_QUESTIONS
    
    print("\nUsing Groq with Llama 3.1 8B Instant")
    llm = GroqEngine(
        data=None,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp2")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")
    
    # ========================================================================
    # BEST MODE: Use retrieval to rank golden documents
    # ========================================================================
    if args.best:
        print("\n" + "=" * 60)
        print("BEST MODE: Loading corpus and running retrieval...")
        print("=" * 60)
        
        questions, corpus_list, corpus_dict, queries, title_to_docids = load_corpus_and_data(num_questions)
        print(f"   Questions: {len(questions)}")
        print(f"   Corpus: {len(corpus_list)} documents")
        
        # Run retrieval
        retrieval_results = run_retrieval_for_best_mode(queries, corpus_list)
        
        results_summary = {}
        all_results = {}
        
        for k in K_VALUES:
            result = run_best_oracle_generation_for_k(
                k=k,
                questions=questions,
                llm=llm,
                results_dir=results_dir,
                retrieval_results=retrieval_results,
                corpus_dict=corpus_dict,
                title_to_docids=title_to_docids
            )
            
            results_summary[k] = result['exact_match']
            all_results[k] = result
        
        # Print final summary
        print("\n" + "=" * 60)
        print("EXPERIMENT 2 (BEST MODE) RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nDataset: MusiqueQA (first {num_questions} questions)")
        print(f"Method: Best Oracle (gold docs ordered by retrieval rank)")
        print(f"LLM: Llama 3.1 8B (Groq)")
        print(f"\nExact Match Scores:")
        print("-" * 30)
        for k in K_VALUES:
            em = results_summary[k]
            avg_rank = all_results[k].get('avg_gold_rank', -1)
            print(f"  k={k}: {em:.4f} ({em*100:.2f}%) | Avg Gold Rank: {avg_rank:.1f}")
        print("-" * 30)
        
        summary = {
            'experiment': 'Experiment 2: Best Oracle RAG (Retrieval-Ranked Gold Docs)',
            'description': 'Using oracle/gold contexts ordered by retrieval ranking',
            'model': 'llama-3.1-8b-instant (Groq)',
            'num_questions': num_questions,
            'retrieval_top_k': BEST_MODE_TOP_K,
            'k_values': K_VALUES,
            'results': {str(k): {
                'exact_match': all_results[k]['exact_match'],
                'avg_gold_rank': all_results[k].get('avg_gold_rank', -1),
                'gold_found_rate': all_results[k].get('gold_found_rate', 0)
            } for k in K_VALUES},
            'timestamp': datetime.now().isoformat(),
            'quick_mode': args.quick
        }
        
        suffix = "_quick" if args.quick else ""
        summary_path = os.path.join(results_dir, f"experiment2_best_oracle_summary{suffix}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        print(f"\nExperiment 2 (Best Mode) Complete!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return
    
    # ========================================================================
    # STANDARD MODE: Perfect oracle (ground truth order)
    # ========================================================================
    print("\nLoading dataset...")
    dev_data = load_data()
    print(f"   Dev questions loaded: {len(dev_data)}")
    
    questions = dev_data[:num_questions]
    print(f"   Using first {num_questions} questions for evaluation")
    
    oracle_counts = []
    for q in questions:
        oracle_ctx = extract_oracle_contexts(q)
        oracle_counts.append(len(oracle_ctx))
    
    avg_oracle = sum(oracle_counts) / len(oracle_counts)
    print(f"\nOracle Context Statistics:")
    print(f"   Average oracle contexts per question: {avg_oracle:.2f}")
    print(f"   Min: {min(oracle_counts)}, Max: {max(oracle_counts)}")
    
    results_summary = {}
    all_results = {}
    
    for k in K_VALUES:
        result = run_oracle_generation_for_k(
            k=k,
            questions=questions,
            llm=llm,
            results_dir=results_dir
        )
        
        results_summary[k] = result['exact_match']
        all_results[k] = result
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nDataset: MusiqueQA (first {num_questions} questions)")
    print(f"Method: Oracle Contexts (Perfect Retrieval)")
    print(f"LLM: Llama 3.1 8B (Groq)")
    print(f"\nExact Match Scores:")
    print("-" * 30)
    for k in K_VALUES:
        print(f"  k={k}: {results_summary[k]:.4f} ({results_summary[k]*100:.2f}%)")
    print("-" * 30)
    
    summary = {
        'experiment': 'Experiment 2: Oracle Context RAG (Perfect Retrieval)',
        'description': 'Using oracle/gold contexts at different k values',
        'model': 'llama-3.1-8b-instant (Groq)',
        'num_questions': num_questions,
        'avg_oracle_contexts': avg_oracle,
        'k_values': K_VALUES,
        'results': {str(k): v for k, v in results_summary.items()},
        'timestamp': datetime.now().isoformat(),
        'quick_mode': args.quick
    }
    
    suffix = "_quick" if args.quick else ""
    summary_path = os.path.join(results_dir, f"experiment2_oracle_summary{suffix}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nExperiment 2 Complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

