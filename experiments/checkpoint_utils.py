import os
import json
import shutil
from typing import Optional, Dict, List, Set, Tuple, Any
from datetime import datetime


def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"   Warning: Failed to load checkpoint: {e}")
            # Try to load backup if exists
            backup_path = checkpoint_path + ".backup"
            if os.path.exists(backup_path):
                print(f"   Attempting to load backup checkpoint...")
                with open(backup_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
    return None


def save_checkpoint(checkpoint_path: str, data: dict, silent: bool = False):
    if os.path.exists(checkpoint_path):
        backup_path = checkpoint_path + ".backup"
        try:
            shutil.copy2(checkpoint_path, backup_path)
        except IOError:
            pass 
    
    # Write new checkpoint
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    if not silent:
        print(f"  Checkpoint saved: {checkpoint_path}")


def get_processed_question_indices(checkpoint: dict) -> Set[int]:
    processed = set()
    for result in checkpoint.get("results", []):
        idx = result.get("idx")
        if idx is not None:
            processed.add(idx)
    return processed


def find_missing_questions(checkpoint: dict, total_questions: int) -> List[int]:
    processed = get_processed_question_indices(checkpoint)
    expected = set(range(total_questions))
    missing = expected - processed
    return sorted(missing)


def validate_checkpoint(
    checkpoint_path: str, 
    total_questions: int,
    verbose: bool = True
) -> Tuple[bool, Optional[dict], List[int]]:
    checkpoint = load_checkpoint(checkpoint_path)
    
    if checkpoint is None:
        if verbose:
            print(f"   No existing checkpoint found at: {checkpoint_path}")
        return False, None, list(range(total_questions))
    
    # Get processed indices
    processed = get_processed_question_indices(checkpoint)
    missing = find_missing_questions(checkpoint, total_questions)
    
    if verbose:
        print(f"   Found checkpoint with {len(processed)} processed questions")
        if missing:
            print(f"   Missing {len(missing)} questions: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        else:
            print(f"   Checkpoint is complete!")
    
    is_complete = len(missing) == 0
    return is_complete, checkpoint, missing


def merge_result_into_checkpoint(
    checkpoint: dict, 
    new_result: dict,
    update_counters: bool = True
) -> dict:
    results = checkpoint.get("results", [])
    
    existing_idx = new_result.get("idx")
    found = False
    for i, r in enumerate(results):
        if r.get("idx") == existing_idx:
            results[i] = new_result
            found = True
            break
    
    if not found:
        results.append(new_result)
    
    # Sort by index for consistency
    results.sort(key=lambda r: r.get("idx", 0))
    
    checkpoint["results"] = results
    
    if update_counters:
        matches = sum(1 for r in results if r.get("is_match", False))
        total = len(results)
        checkpoint["matches"] = matches
        checkpoint["total"] = total
        
        if results:
            checkpoint["last_processed_idx"] = max(r.get("idx", 0) for r in results)
    
    checkpoint["last_updated"] = datetime.now().isoformat()
    
    return checkpoint


def prepare_checkpoint_for_resume(
    checkpoint_path: str,
    total_questions: int,
    verbose: bool = True
) -> Tuple[dict, List[int], int, int, List[dict]]:
    is_complete, checkpoint, missing = validate_checkpoint(
        checkpoint_path, total_questions, verbose=verbose
    )
    
    if checkpoint is None:
        # No existing checkpoint - start fresh
        return {}, list(range(total_questions)), 0, 0, []
    
    if is_complete:
        # Checkpoint is complete 
        matches = checkpoint.get("matches", 0)
        total = checkpoint.get("total", 0)
        results = checkpoint.get("results", [])
        return checkpoint, [], matches, total, results
    
    # Checkpoint exists but has missing questions
    matches = checkpoint.get("matches", 0)
    total = checkpoint.get("total", 0)
    results = checkpoint.get("results", [])
    
    return checkpoint, missing, matches, total, results


def get_indices_to_process(
    checkpoint_path: str,
    total_questions: int,
    verbose: bool = True
) -> Tuple[List[int], int, int, List[dict], dict]:
    checkpoint, missing, matches, total, results = prepare_checkpoint_for_resume(
        checkpoint_path, total_questions, verbose=verbose
    )
    
    if not checkpoint:
        # Fresh start
        return list(range(total_questions)), 0, 0, [], {}
    
    if not missing:
        # Complete
        if verbose:
            print(f"   All {total_questions} questions already processed!")
        return [], matches, total, results, checkpoint
    
    # Return missing indices for processing
    if verbose:
        print(f"   Will process {len(missing)} missing questions")
    
    return missing, matches, total, results, checkpoint