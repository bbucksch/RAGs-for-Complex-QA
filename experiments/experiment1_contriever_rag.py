import os
import sys
import json
import time
import re
import argparse
import joblib
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Optional
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
from checkpoint_utils import load_checkpoint, save_checkpoint, get_indices_to_process, merge_result_into_checkpoint
from rag_prompt import create_rag_prompt, extract_final_answer, normalize_answer, cover_exact_match


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of questions to evaluate 
NUM_QUESTIONS = 1200

QUICK_MODE_QUESTIONS = 100

QUICK_MODE_CORPUS = 1000

K_VALUES = [1, 3, 5]

TOP_K_RETRIEVAL = 100

CACHE_DIR = os.path.join(project_root, "experiments", "cache")

MIN_DELAY_BETWEEN_REQUESTS = 0.1
CHECKPOINT_FREQUENCY = 50
OUTPUT_DIR = os.path.join(project_root, "experiments", "results")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(num_questions: int = NUM_QUESTIONS, corpus_limit: int = None):
    print("\nLoading dataset and corpus...")
    print("   This may take a few minutes for the large corpus...")
    
    # Load dev.json 
    dev_path = os.path.join(project_root, "data", "dev.json")
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    print(f"   Dev questions loaded: {len(dev_data)}")
    
    # Load corpus
    corpus_path = os.path.join(project_root, "wiki_musique_corpus.json")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_raw = json.load(f)
    
    print(f"   Corpus loaded: {len(corpus_raw)} documents")
    
    # Apply corpus limit 
    if corpus_limit is not None and corpus_limit < len(corpus_raw):
        corpus_raw = dict(list(corpus_raw.items())[:corpus_limit])
        print(f"   Corpus limited to first {corpus_limit} documents (quick mode)")
    
    title_to_docids = {}
    for doc_id, doc_data in corpus_raw.items():
        title = doc_data.get("title", "")
        if title not in title_to_docids:
            title_to_docids[title] = []
        title_to_docids[title].append(doc_id)
    
    print(f"   Built title index: {len(title_to_docids)} unique titles")
    
    # Convert corpus to list of Evidence objects for retriever
    from dexter.data.datastructures.evidence import Evidence
    
    corpus_list = []  # List for retriever
    corpus_dict = {}  # Dict for lookup by ID
    
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
    
    print(f"   Corpus converted: {len(corpus_list)} Evidence objects")
    
    # Prepare queries and answers
    from dexter.data.datastructures.question import Question
    from dexter.data.datastructures.answer import Answer
    
    queries = []
    answers = []
    qrels = {} 
    
    for item in dev_data[:num_questions]:
        qid = item["_id"]
        question = Question(item["question"], idx=qid)
        answer = Answer(item["answer"])
        
        queries.append(question)
        answers.append(answer)
        
        qrels[str(qid)] = {}
    
    print(f"   Using first {num_questions} questions for evaluation")
    
    return queries, qrels, corpus_list, corpus_dict, answers, dev_data[:num_questions], title_to_docids


def get_gold_doc_ids(item: dict, title_to_docids: dict) -> set:
    gold_ids = set()
    supporting_facts = item.get("supporting_facts", [])
    
    for fact in supporting_facts:
        # Each fact is [title, paragraph_index]
        if isinstance(fact, list) and len(fact) >= 1:
            title = fact[0]
            # Find all doc_ids with this title
            if title in title_to_docids:
                gold_ids.update(title_to_docids[title])
    
    return gold_ids


def compute_gold_doc_stats(question_id: str, retrieval_results: dict, gold_doc_ids: set, k: int) -> dict:
    if not gold_doc_ids:
        return {
            "num_gold_docs": 0,
            "gold_docs_in_topk": 0,
            "all_gold_in_topk": True,  # Vacuously true
            "gold_doc_ids": [],
            "retrieved_gold_ids": []
        }
    
    # Get top-k retrieved doc_ids
    qid_results = retrieval_results.get(question_id, {})
    sorted_results = sorted(qid_results.items(), key=lambda x: x[1], reverse=True)[:k]
    topk_doc_ids = set(doc_id for doc_id, _ in sorted_results)
    
    # Check which gold docs are in top-k
    retrieved_gold = gold_doc_ids & topk_doc_ids
    
    return {
        "num_gold_docs": len(gold_doc_ids),
        "gold_docs_in_topk": len(retrieved_gold),
        "all_gold_in_topk": len(retrieved_gold) == len(gold_doc_ids),
        "gold_doc_ids": list(gold_doc_ids),
        "retrieved_gold_ids": list(retrieved_gold)
    }


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


def save_embeddings_cache(corpus_size: int, embeddings, corpus_ids: List[str]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path(corpus_size)
    print(f"   Saving embeddings to cache: {cache_path}")
    torch.save({
        'embeddings': embeddings,
        'corpus_ids': corpus_ids,
        'corpus_size': corpus_size
    }, cache_path)
    print(f"   Cache saved successfully")


def run_contriever_retrieval(queries: List, corpus_list: List, qrels: Dict, use_cache: bool = True) -> Dict:
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
    
    # Try to load cached embeddings
    cached_data = None
    if use_cache:
        cached_data = load_cached_embeddings(corpus_size)
    
    start_time = time.time()
    
    if cached_data is not None and cached_data['corpus_size'] == corpus_size:
        # Use cached corpus embeddings
        print("   Encoding queries and using cached corpus embeddings...")
        corpus_embeddings = cached_data['embeddings']
        
        # Encode queries
        query_embeddings = contriever.encode_queries(queries, batch_size=config_instance.batch_size)
        
        # Move corpus embeddings to same device as query embeddings
        device = query_embeddings.device
        corpus_embeddings = corpus_embeddings.to(device)
        
        # Compute similarity scores
        cos_scores = similarity_measure.evaluate(query_embeddings, corpus_embeddings)
        
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores, 
            min(TOP_K_RETRIEVAL + 1, len(cos_scores[1])), 
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
        # No cache - do full encoding
        print("   Encoding corpus and queries (this takes a while for large corpus)...")
        
        # Encode corpus first
        corpus_embeddings = contriever.encode_corpus(corpus_list)
        
        if use_cache:
            save_embeddings_cache(corpus_size, corpus_embeddings.cpu(), corpus_ids)
        
        # Encode queries
        query_embeddings = contriever.encode_queries(queries, batch_size=config_instance.batch_size)
        
        # Compute similarity scores
        cos_scores = similarity_measure.evaluate(query_embeddings, corpus_embeddings)
        
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores, 
            min(TOP_K_RETRIEVAL + 1, len(cos_scores[1])), 
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
    
    # Evaluate retrieval metrics
    print("\nRetrieval Metrics (before generation):")
    
    has_valid_qrels = any(len(docs) > 0 for docs in qrels.values())
    
    if has_valid_qrels:
        metrics = RetrievalMetrics(k_values=K_VALUES)
        try:
            retrieval_scores = metrics.evaluate_retrieval(qrels=qrels, results=response)
            print(f"   {retrieval_scores}")
        except Exception as e:
            print(f"   Could not compute retrieval metrics: {e}")
    else:
        print("   Skipping metrics: No relevance judgments available in qrels")
    
    return response


# ============================================================================
# GENERATION WITH LLM
# ============================================================================

def get_context_for_query(query_id: str, retrieval_results: Dict, corpus: Dict, k: int) -> str:
    if query_id not in retrieval_results:
        return ""
    
    # Get top-k doc IDs 
    doc_scores = retrieval_results[query_id]
    top_k_docs = list(doc_scores.keys())[:k]
    
    contexts = []
    for doc_id in top_k_docs:
        if doc_id in corpus:
            doc = corpus[doc_id]
            # Extract text from document
            if hasattr(doc, 'text'):
                text = doc.text() if callable(doc.text) else doc.text
            else:
                text = str(doc)
            contexts.append(text)
    
    return "\n\n".join(contexts)


def run_generation_for_k(
    k: int,
    queries: List,
    answers: List,
    retrieval_results: Dict,
    corpus: Dict,
    llm_engine,
    checkpoint_dir: str,
    raw_data: List = None,
    title_to_docids: Dict = None,
    k_offset: int = 0
) -> dict:
    print(f"\n{'='*60}")
    print(f"GENERATION WITH k={k}")
    print(f"{'='*60}")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_k{k}.json")
    total_questions = len(queries)
    
    # Get indices to process 
    indices_to_process, matches, total, existing_results, checkpoint = get_indices_to_process(
        checkpoint_path, total_questions, verbose=True
    )
    
    if not indices_to_process:
        print(f"   All {total_questions} questions already processed!")
        # Calculate stats from existing checkpoint
        em_matches = sum(1 for r in existing_results if r.get("is_match", False))
        total_processed = len(existing_results)
        em_score = em_matches / total_processed if total_processed else 0.0
        
        # Load checkpoint for golden doc stats
        all_gold_retrieved = checkpoint.get("all_gold_retrieved_count", 0) if checkpoint else 0
        all_gold_correct = checkpoint.get("all_gold_correct_count", 0) if checkpoint else 0
        gold_recall_rate = all_gold_retrieved / total_processed if total_processed > 0 else 0
        gold_em_when_retrieved = all_gold_correct / all_gold_retrieved if all_gold_retrieved > 0 else 0
        
        return {
            "exact_match": em_score,
            "matches": em_matches,
            "total": total_processed,
            "all_gold_retrieved_count": all_gold_retrieved,
            "all_gold_correct_count": all_gold_correct,
            "gold_recall_rate": gold_recall_rate,
            "gold_em_when_retrieved": gold_em_when_retrieved
        }
    
    # Initialize checkpoint if needed
    if not checkpoint:
        checkpoint = {
            "k": k, 
            "results": [], 
            "matches": 0, 
            "total": 0,
            "all_gold_retrieved_count": 0,
            "all_gold_correct_count": 0
        }
    errors = 0
    
    use_tqdm = TQDM_AVAILABLE
    if use_tqdm:
        iterator = tqdm(
            indices_to_process,
            desc=f"   k={k}",
            unit="q",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        iterator = indices_to_process
    
    processed_count = 0
    for idx in iterator:
        query = queries[idx]
        gold_answer = answers[idx]
        
        question_id = query.id() if hasattr(query, 'id') else str(idx)
        question_text = query.text() if hasattr(query, 'text') else str(query)
        gold_text = gold_answer.text() if hasattr(gold_answer, 'text') else str(gold_answer)
        
        # Compute golden document stats if we have the data
        gold_stats = {}
        if raw_data and title_to_docids and idx < len(raw_data):
            gold_doc_ids = get_gold_doc_ids(raw_data[idx], title_to_docids)
            gold_stats = compute_gold_doc_stats(str(question_id), retrieval_results, gold_doc_ids, k)
        
        # Get context
        context = get_context_for_query(str(question_id), retrieval_results, corpus, k)
        
        if not context:
            if TQDM_AVAILABLE:
                tqdm.write(f"   No context for question {idx+1}, skipping")
            else:
                print(f"   No context for question {idx+1}, skipping")
            errors += 1
            continue
        
        # Create prompt
        system_prompt, user_prompt = create_rag_prompt(question_text, context)
        
        try:
            # Generate answer 
            response = llm_engine.get_chat_completion(user_prompt, system_prompt)
            
            # Extract the final answer 
            predicted = extract_final_answer(response)
            is_match = cover_exact_match(predicted, gold_text)
            
            # Store result with golden document tracking
            new_result = {
                "idx": idx,
                "question_id": question_id,
                "question": question_text,
                "gold_answer": gold_text,
                "predicted": predicted,
                "is_match": is_match,
                "raw_response": response,
                # Golden document stats
                "num_gold_docs": gold_stats.get("num_gold_docs", 0),
                "gold_docs_in_topk": gold_stats.get("gold_docs_in_topk", 0),
                "all_gold_in_topk": gold_stats.get("all_gold_in_topk", False)
            }
            
            if gold_stats.get("all_gold_in_topk", False):
                checkpoint["all_gold_retrieved_count"] = checkpoint.get("all_gold_retrieved_count", 0) + 1
                if is_match:
                    checkpoint["all_gold_correct_count"] = checkpoint.get("all_gold_correct_count", 0) + 1
            
            checkpoint = merge_result_into_checkpoint(checkpoint, new_result)
            
            matches = checkpoint.get("matches", 0)
            total = checkpoint.get("total", 0)
            
            processed_count += 1
            
            if use_tqdm and hasattr(iterator, 'set_postfix'):
                current_em = matches / total if total > 0 else 0
                iterator.set_postfix(EM=f"{current_em:.3f}", correct=matches)
            elif not use_tqdm and (idx + 1) % 10 == 0:
                current_em = matches / total if total > 0 else 0
                print(f"   [{idx+1}/{len(queries)}] EM: {current_em:.4f} ({matches}/{total})")
            
            if processed_count % CHECKPOINT_FREQUENCY == 0:
                checkpoint["k"] = k
                save_checkpoint(checkpoint_path, checkpoint, silent=use_tqdm)
                if use_tqdm:
                    tqdm.write(f"  Checkpoint saved: {checkpoint_path}")
            
            time.sleep(MIN_DELAY_BETWEEN_REQUESTS)
            
        except Exception as e:
            errors += 1
            error_msg = str(e)
            if TQDM_AVAILABLE:
                tqdm.write(f"   Error at question {idx+1}: {error_msg[:80]}")
            else:
                print(f"   Error at question {idx+1}: {error_msg}")
            
            time.sleep(1)
    
    results = checkpoint.get("results", [])
    matches = checkpoint.get("matches", 0)
    total = checkpoint.get("total", 0)
    em_score = matches / total if total > 0 else 0
    
    all_gold_retrieved = checkpoint.get("all_gold_retrieved_count", 0)
    all_gold_correct = checkpoint.get("all_gold_correct_count", 0)
    gold_recall_rate = all_gold_retrieved / total if total > 0 else 0
    gold_em_when_retrieved = all_gold_correct / all_gold_retrieved if all_gold_retrieved > 0 else 0
    
    final_results = {
        "k": k,
        "exact_match": em_score,
        "matches": matches,
        "total": total,
        "errors": errors,
        "all_gold_retrieved_count": all_gold_retrieved,
        "all_gold_correct_count": all_gold_correct,
        "gold_recall_rate": gold_recall_rate,
        "gold_em_when_retrieved": gold_em_when_retrieved,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    final_path = os.path.join(checkpoint_dir, f"results_k{k}.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n   k={k} Complete!")
    print(f"   Exact Match: {em_score:.4f} ({matches}/{total})")
    print(f"   All Gold Docs Retrieved: {all_gold_retrieved}/{total} ({gold_recall_rate:.2%})")
    print(f"   EM when All Gold Retrieved: {gold_em_when_retrieved:.4f} ({all_gold_correct}/{all_gold_retrieved})")
    print(f"   Errors: {errors}")
    print(f"   Results saved: {final_path}")
    
    # Return dict with all stats
    return {
        "exact_match": em_score,
        "matches": matches,
        "total": total,
        "all_gold_retrieved_count": all_gold_retrieved,
        "all_gold_correct_count": all_gold_correct,
        "gold_recall_rate": gold_recall_rate,
        "gold_em_when_retrieved": gold_em_when_retrieved
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment_loop(args, queries, answers, retrieval_results, corpus_dict, 
                        llm_engine, raw_data, title_to_docids, 
                        using_groq=True):
    results_summary = {}
    num_questions = len(queries)
    
    for ki, k in enumerate(K_VALUES):
        # Calculate offset: questions processed in prior k values
        k_offset = 0
        for prev_k in K_VALUES[:ki]:
            prev_checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_k{prev_k}.json")
            if os.path.exists(prev_checkpoint_path):
                try:
                    with open(prev_checkpoint_path, 'r', encoding='utf-8') as f:
                        prev_ckpt = json.load(f)
                        k_offset += len(prev_ckpt.get("results", []))
                except:
                    pass
        
        k_results = run_generation_for_k(
            k=k,
            queries=queries,
            answers=answers,
            retrieval_results=retrieval_results,
            corpus=corpus_dict,
            llm_engine=llm_engine,
            checkpoint_dir=OUTPUT_DIR,
            raw_data=raw_data,
            title_to_docids=title_to_docids,
            k_offset=k_offset
        )
        results_summary[k] = k_results
    
    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT 1 RESULTS SUMMARY")
    print("="*60)
    print(f"\nDataset: MusiqueQA (first {num_questions} questions)")
    print(f"Retriever: Contriever (facebook/contriever)")
    print(f"LLM: {'Llama 3.1 8B (Groq)' if using_groq else 'GPT-3.5-turbo (OpenAI)'}")
    print(f"\nExact Match Scores:")
    print("-" * 50)
    for k, stats in results_summary.items():
        em = stats["exact_match"]
        gold_rate = stats["gold_recall_rate"]
        gold_em = stats["gold_em_when_retrieved"]
        print(f"  k={k}: EM={em:.4f} ({em*100:.2f}%) | Gold Recall={gold_rate:.2%} | EM@Gold={gold_em:.4f}")
    print("-" * 50)
    
    suffix = "_quick" if args.quick else ""
    summary_path = os.path.join(OUTPUT_DIR, f"experiment1_summary{suffix}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "Experiment 1: Standard RAG Baseline",
            "num_questions": num_questions,
            "retriever": "Contriever (facebook/contriever)",
            "llm": "llama-3.1-8b-instant" if using_groq else "gpt-3.5-turbo",
            "k_values": K_VALUES,
            "results": {str(k): v for k, v in results_summary.items()},
            "quick_mode": args.quick,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Standard RAG with Contriever')
    parser.add_argument('--quick', action='store_true', 
                        help=f'Quick mode: only process first {QUICK_MODE_QUESTIONS} questions with {QUICK_MODE_CORPUS} corpus docs')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable corpus embedding caching')
    parser.add_argument('--use-full-corpus', action='store_true',
                        help='Use full corpus with cached embeddings even in quick mode (faster if cache exists)')
    args = parser.parse_args()
    
    print("="*60)
    print("EXPERIMENT 1: Standard RAG Baseline with Contriever")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        num_questions = QUICK_MODE_QUESTIONS
        if args.use_full_corpus:
            print(f"\nQUICK MODE (Full Corpus): Processing {QUICK_MODE_QUESTIONS} questions with FULL cached corpus")
            corpus_limit = None
        else:
            print(f"\nQUICK MODE: Processing {QUICK_MODE_QUESTIONS} questions with {QUICK_MODE_CORPUS} corpus docs")
            corpus_limit = QUICK_MODE_CORPUS
    else:
        num_questions = NUM_QUESTIONS
        corpus_limit = None
    
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
    
    queries, qrels, corpus_list, corpus_dict, answers, raw_data, title_to_docids = load_data(num_questions, corpus_limit)
    
    use_cache = not args.no_cache
    retrieval_results = run_contriever_retrieval(queries, corpus_list, qrels, use_cache=use_cache)
    
    using_groq = paid_api_key and paid_api_key != "your_paid_groq_api_key_here"
    
    run_experiment_loop(
        args=args,
        queries=queries,
        answers=answers,
        retrieval_results=retrieval_results,
        corpus_dict=corpus_dict,
        llm_engine=llm_engine,
        raw_data=raw_data,
        title_to_docids=title_to_docids,
        using_groq=using_groq
    )


if __name__ == "__main__":
    main()
