import os
import sys
import json
import time
import argparse
import logging
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Matplotlib for loss graphs
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: Install matplotlib for loss graphs: pip install matplotlib")

# Transformers for model loading
from transformers import AutoTokenizer, AutoModel

# Import utilities from the project
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
    from checkpoint_utils import (
        load_checkpoint, save_checkpoint,
        get_indices_to_process, merge_result_into_checkpoint
    )
    from rag_prompt import (
        create_rag_prompt, extract_final_answer,
        normalize_answer, cover_exact_match
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

# Device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data configuration
NUM_TRAIN_QUESTIONS = None  # Use all training data (or set a limit)
NUM_TEST_QUESTIONS = 1200   # First 1200 questions from dev.json

# Quick mode configuration
QUICK_TRAIN_BATCHES = 5     # Number of batches for quick training
QUICK_TEST_QUESTIONS = 10   # Number of questions for quick testing

# Training hyperparameters
BATCH_SIZE = 8              # Batch size (reduce if OOM)
LEARNING_RATE = 5e-6        # Learning rate (paper: 5e-6)
NUM_EPOCHS = 100            # Max epochs (Ctrl+C to stop early)
WARMUP_STEPS = 100          # Warmup steps for scheduler
GRADIENT_ACCUMULATION = 1   # Gradient accumulation steps
PATIENCE = 1000               # Early stopping patience (epochs without improvement)
BJORN_PATIENCE = 10         # Save adore_bjorn after this many checkpoints without improvement
SAVE_EVERY_N_STEPS = 2000    # Save checkpoint every N steps

# Hard negative mining
NUM_HARD_NEGATIVES = 7      # Number of hard negatives per query
TOP_K_CANDIDATES = 100      # Top-K candidates for hard negative mining
MARGIN = 0.1                # Margin for ranking loss (optional)

# InfoNCE loss temperature
TEMPERATURE = 0.05

# Evaluation
K_VALUES = [1, 3, 5]        # Top-K values for evaluation

# Checkpointing
CHECKPOINT_FREQUENCY = 50
MIN_DELAY_BETWEEN_REQUESTS = 0.1
CHECKPOINT_DIR = os.path.join(project_root, "experiments", "checkpoints")
RESULTS_DIR = os.path.join(project_root, "experiments", "results")
CACHE_DIR = os.path.join(project_root, "experiments", "cache")

# Model configuration
BASE_MODEL = "facebook/contriever"  # Pre-trained model for initialization

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingSample:
    query_id: str
    query_text: str
    gold_doc_ids: List[str]
    answer: str


class ADOREDataset(Dataset):
    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# QUERY ENCODER
# ============================================================================

class QueryEncoder(nn.Module):
    def __init__(self, model_name: str = BASE_MODEL, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.to(device)
        
        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size
        
        logger.info(f"QueryEncoder initialized from {model_name}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Device: {device}")
    
    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Encode
        outputs = self.encoder(**encoded)
        
        # Mean pooling
        embeddings = self.mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
        
        return embeddings
    
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.forward(batch)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.encoder.state_dict(),
            'model_name': BASE_MODEL,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


# ============================================================================
# HARD NEGATIVE MINING
# ============================================================================

class HardNegativeMiner:
    def __init__(
        self,
        corpus_embeddings: torch.Tensor,
        corpus_ids: List[str],
        device: torch.device = DEVICE
    ):
        self.device = device
        self.corpus_ids = corpus_ids
        self.id_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
        
        # Move embeddings to device
        self.corpus_embeddings = corpus_embeddings.to(device)
        
        # Normalize embeddings for cosine similarity
        self.corpus_embeddings_normalized = F.normalize(self.corpus_embeddings, p=2, dim=1)
        
        logger.info(f"HardNegativeMiner initialized with {len(corpus_ids)} documents")
        logger.info(f"  Embedding shape: {self.corpus_embeddings.shape}")
    
    def mine(
        self,
        query_embeddings: torch.Tensor,
        gold_doc_ids_batch: List[List[str]],
        num_negatives: int = NUM_HARD_NEGATIVES,
        top_k: int = TOP_K_CANDIDATES
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        batch_size = query_embeddings.shape[0]
        
        # Normalize query embeddings
        query_embeddings_normalized = F.normalize(query_embeddings, p=2, dim=1)
        
        # Compute similarity scores with all documents
        # [batch_size, num_docs]
        similarity_scores = torch.matmul(
            query_embeddings_normalized, 
            self.corpus_embeddings_normalized.t()
        )
        
        # Get top-k candidates
        top_k_scores, top_k_indices = torch.topk(similarity_scores, top_k, dim=1)
        
        gold_embeddings_list = []
        negative_embeddings_list = []
        negative_indices_list = []
        
        for i in range(batch_size):
            gold_ids = set(gold_doc_ids_batch[i])
            
            # Get gold embedding (use first gold doc if multiple)
            gold_idx = None
            for gold_id in gold_ids:
                if gold_id in self.id_to_idx:
                    gold_idx = self.id_to_idx[gold_id]
                    break
            
            if gold_idx is None:
                # Fallback: use a random document as "gold" (shouldn't happen often)
                gold_idx = 0
                logger.warning(f"No gold document found for query {i}, using fallback")
            
            gold_embeddings_list.append(self.corpus_embeddings[gold_idx])
            
            # Find hard negatives (high-ranked but not gold)
            candidate_indices = top_k_indices[i].tolist()
            negatives = []
            neg_indices = []
            
            for cand_idx in candidate_indices:
                if len(negatives) >= num_negatives:
                    break
                cand_id = self.corpus_ids[cand_idx]
                if cand_id not in gold_ids:
                    negatives.append(self.corpus_embeddings[cand_idx])
                    neg_indices.append(cand_idx)
            
            # Pad if not enough negatives
            while len(negatives) < num_negatives:
                # Use random documents as padding
                rand_idx = torch.randint(0, len(self.corpus_ids), (1,)).item()
                if self.corpus_ids[rand_idx] not in gold_ids:
                    negatives.append(self.corpus_embeddings[rand_idx])
                    neg_indices.append(rand_idx)
            
            negative_embeddings_list.append(torch.stack(negatives))
            negative_indices_list.append(neg_indices)
        
        gold_embeddings = torch.stack(gold_embeddings_list)
        negative_embeddings = torch.stack(negative_embeddings_list)
        
        return gold_embeddings, negative_embeddings, negative_indices_list


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        batch_size = query_embeddings.shape[0]
        num_negatives = negative_embeddings.shape[1]
        
        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=1)
        negative_norm = F.normalize(negative_embeddings, p=2, dim=2)
        
        # Compute positive similarity
        # [batch_size]
        positive_sim = torch.sum(query_norm * positive_norm, dim=1) / self.temperature
        
        # Compute negative similarities
        # [batch_size, num_negatives]
        negative_sim = torch.bmm(
            negative_norm,
            query_norm.unsqueeze(2)
        ).squeeze(2) / self.temperature
        
        # Concatenate: [batch_size, 1 + num_negatives]
        # Positive is at index 0
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


# ============================================================================
# DATA LOADING
# ============================================================================

def load_corpus_embeddings(cache_path: Optional[str] = None) -> Tuple[torch.Tensor, List[str]]:
    if cache_path is None:
        # Look for existing cache
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("contriever_corpus_")]
        if not cache_files:
            raise FileNotFoundError(
                f"No corpus embeddings found in {CACHE_DIR}. "
                "Please run experiment1_contriever_rag.py first to generate embeddings."
            )
        cache_path = os.path.join(CACHE_DIR, cache_files[0])
    
    logger.info(f"Loading corpus embeddings from: {cache_path}")
    
    cached_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    embeddings = cached_data['embeddings']
    corpus_ids = cached_data['corpus_ids']
    
    logger.info(f"  Loaded {len(corpus_ids)} document embeddings")
    logger.info(f"  Embedding shape: {embeddings.shape}")
    
    return embeddings, corpus_ids


def load_corpus_dict() -> Dict[str, Dict[str, str]]:
    corpus_path = os.path.join(project_root, "wiki_musique_corpus.json")
    
    logger.info(f"Loading corpus from: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    logger.info(f"  Loaded {len(corpus)} documents")
    
    return corpus


def load_training_data(limit: Optional[int] = None) -> List[TrainingSample]:
    train_path = os.path.join(project_root, "data", "train.json")
    
    logger.info(f"Loading training data from: {train_path}")
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    if limit:
        train_data = train_data[:limit]
    
    # Build title to doc_id mapping
    corpus = load_corpus_dict()
    title_to_docids = {}
    for doc_id, doc_data in corpus.items():
        title = doc_data.get("title", "")
        if title not in title_to_docids:
            title_to_docids[title] = []
        title_to_docids[title].append(doc_id)
    
    samples = []
    for item in train_data:
        # Extract gold document IDs from supporting_facts
        gold_doc_ids = []
        for fact in item.get("supporting_facts", []):
            if isinstance(fact, list) and len(fact) >= 1:
                title = fact[0]
                if title in title_to_docids:
                    gold_doc_ids.extend(title_to_docids[title])
        
        sample = TrainingSample(
            query_id=item.get("_id", item.get("id", str(len(samples)))),
            query_text=item.get("question", ""),
            gold_doc_ids=list(set(gold_doc_ids)),  # Deduplicate
            answer=item.get("answer", "")
        )
        samples.append(sample)
    
    logger.info(f"  Loaded {len(samples)} training samples")
    
    # Filter samples with no gold documents
    valid_samples = [s for s in samples if len(s.gold_doc_ids) > 0]
    if len(valid_samples) < len(samples):
        logger.warning(f"  Filtered {len(samples) - len(valid_samples)} samples without gold docs")
    
    return valid_samples


def load_dev_data(limit: int = NUM_TEST_QUESTIONS) -> Tuple[List[Dict], Dict[str, List[str]]]:
    dev_path = os.path.join(project_root, "data", "dev.json")
    
    logger.info(f"Loading dev data from: {dev_path}")
    
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    dev_data = dev_data[:limit]
    
    # Build title to doc_id mapping
    corpus = load_corpus_dict()
    title_to_docids = {}
    for doc_id, doc_data in corpus.items():
        title = doc_data.get("title", "")
        if title not in title_to_docids:
            title_to_docids[title] = []
        title_to_docids[title].append(doc_id)
    
    logger.info(f"  Loaded {len(dev_data)} dev questions")
    
    return dev_data, title_to_docids


def load_validation_data(
    num_samples: int = 1000,
    title_to_docids: Optional[Dict[str, List[str]]] = None
) -> Tuple[List[TrainingSample], Dict[str, List[str]]]:
    # Use dev.json for validation (test.json has no ground truth)
    val_path = os.path.join(project_root, "data", "dev.json")
    
    logger.info(f"Loading validation data from: {val_path}")
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # Use the LAST num_samples questions to avoid overlap with training
    total_questions = len(val_data)
    if num_samples and num_samples < total_questions:
        val_data = val_data[-num_samples:]  # Take from the END
        logger.info(f"  Using last {num_samples} questions (indices {total_questions - num_samples} to {total_questions})")
    
    # Build title to doc_id mapping if not provided
    if title_to_docids is None:
        corpus = load_corpus_dict()
        title_to_docids = {}
        for doc_id, doc_data in corpus.items():
            title = doc_data.get("title", "")
            if title not in title_to_docids:
                title_to_docids[title] = []
            title_to_docids[title].append(doc_id)
    
    samples = []
    for item in val_data:
        # Extract gold document IDs from supporting_facts
        gold_doc_ids = []
        for fact in item.get("supporting_facts", []):
            if isinstance(fact, list) and len(fact) >= 1:
                title = fact[0]
                if title in title_to_docids:
                    gold_doc_ids.extend(title_to_docids[title])
        
        sample = TrainingSample(
            query_id=item.get("_id", item.get("id", str(len(samples)))),
            query_text=item.get("question", ""),
            gold_doc_ids=list(set(gold_doc_ids)),  # Deduplicate
            answer=item.get("answer", "")
        )
        samples.append(sample)
    
    # Filter samples with no gold documents
    valid_samples = [s for s in samples if len(s.gold_doc_ids) > 0]
    if len(valid_samples) < len(samples):
        logger.warning(f"  Filtered {len(samples) - len(valid_samples)} validation samples without gold docs")
    
    logger.info(f"  Loaded {len(valid_samples)} validation samples")
    
    return valid_samples, title_to_docids


def compute_validation_loss(
    query_encoder: QueryEncoder,
    miner: 'HardNegativeMiner',
    criterion: 'InfoNCELoss',
    val_samples: List[TrainingSample],
    batch_size: int = BATCH_SIZE,
    quick_mode: bool = False
) -> float:
    query_encoder.encoder.eval()
    
    # In quick mode, use only a subset
    if quick_mode:
        val_samples = val_samples[:min(50, len(val_samples))]
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i:i + batch_size]
            
            queries = [s.query_text for s in batch]
            gold_doc_ids = [s.gold_doc_ids for s in batch]
            
            # Encode queries
            query_embeddings = query_encoder.forward(queries)
            
            # Mine hard negatives
            gold_embeddings, negative_embeddings, _ = miner.mine(
                query_embeddings,
                gold_doc_ids,
                num_negatives=NUM_HARD_NEGATIVES
            )
            
            # Compute loss (need to temporarily enable grad for loss computation)
            # We'll compute it manually without backward
            batch_size_actual = query_embeddings.shape[0]
            num_negatives = negative_embeddings.shape[1]
            
            # Normalize embeddings
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            positive_norm = F.normalize(gold_embeddings, p=2, dim=1)
            negative_norm = F.normalize(negative_embeddings, p=2, dim=2)
            
            # Compute positive similarity
            positive_sim = torch.sum(query_norm * positive_norm, dim=1) / criterion.temperature
            
            # Compute negative similarities
            negative_sim = torch.bmm(
                negative_norm,
                query_norm.unsqueeze(2)
            ).squeeze(2) / criterion.temperature
            
            # Concatenate
            logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
            
            # Labels
            labels = torch.zeros(batch_size_actual, dtype=torch.long, device=query_embeddings.device)
            
            # Cross entropy loss
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    query_encoder.encoder.train()
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def generate_loss_graphs(
    train_losses: List[float],
    val_losses: List[float],
    epochs: List[int],
    save_path: str
):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Cannot generate loss graphs.")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation loss alone (for better visibility)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_losses, 'r-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    # Mark best validation loss
    if val_losses:
        best_idx = val_losses.index(min(val_losses))
        plt.axvline(x=epochs[best_idx], color='g', linestyle='--', alpha=0.7, label=f'Best (Epoch {epochs[best_idx]})')
        plt.scatter([epochs[best_idx]], [val_losses[best_idx]], color='g', s=100, zorder=5)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Loss graph saved to: {save_path}")


# ============================================================================
# TRAINING
# ============================================================================

def train_adore(
    args,
    query_encoder: QueryEncoder,
    corpus_embeddings: torch.Tensor,
    corpus_ids: List[str],
    train_samples: List[TrainingSample],
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    quick_mode: bool = False,
    resume_from: Optional[str] = None
):
    logger.info("="*60)
    logger.info("ADORE TRAINING")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Max epochs: {num_epochs} (Ctrl+C to stop early)")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Hard negatives per query: {NUM_HARD_NEGATIVES}")
    logger.info(f"Early stopping patience: {PATIENCE} epochs")
    logger.info(f"Bjorn checkpoint patience: {BJORN_PATIENCE} epochs after first best")
    logger.info(f"Quick mode: {quick_mode}")
    logger.info("")
    logger.info("Press Ctrl+C anytime to stop training gracefully.")
    logger.info("   Best checkpoint (by TEST loss) saved at: checkpoints/adore_best.pt")
    logger.info("=" * 60)
    
    miner = HardNegativeMiner(corpus_embeddings, corpus_ids, DEVICE)
    criterion = InfoNCELoss(temperature=TEMPERATURE)
    
    logger.info("Loading validation data for checkpoint evaluation...")
    val_samples_count = 100 if quick_mode else 1000
    val_samples, _ = load_validation_data(num_samples=val_samples_count)
    logger.info(f"  Validation samples for evaluation: {len(val_samples)}")
    
    optimizer = AdamW(query_encoder.encoder.parameters(), lr=learning_rate)
    
    dataset = ADOREDataset(train_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: x
    )
    
    total_steps = len(dataloader) * num_epochs
    if quick_mode:
        total_steps = QUICK_TRAIN_BATCHES
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    global_step = 0
    best_test_loss = float('inf')
    epochs_without_improvement = 0
    training_interrupted = False
    start_epoch = 0
    
    # Bjorn checkpoint tracking
    first_best_saved = False
    epochs_since_first_best = 0
    bjorn_saved = False
    
    # Loss logging
    train_losses = []
    test_losses = []
    epoch_numbers = []
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming training from: {resume_from}")
        query_encoder.load(resume_from)
        # Try to extract epoch number from filename
        try:
            if 'epoch_' in resume_from:
                start_epoch = int(resume_from.split('epoch_')[1].split('.')[0])
                logger.info(f"  Resuming from epoch {start_epoch}")
        except:
            pass
    
    try:
      for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        query_encoder.encoder.train()
        
        if TQDM_AVAILABLE:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            pbar = dataloader
        
        for batch_idx, batch in enumerate(pbar):
            # Quick mode: only train for a few batches
            if quick_mode and batch_idx >= QUICK_TRAIN_BATCHES:
                break
            
            # Extract batch data
            queries = [s.query_text for s in batch]
            gold_doc_ids = [s.gold_doc_ids for s in batch]
            
            # Encode queries
            query_embeddings = query_encoder.forward(queries)
            
            # Mine hard negatives
            with torch.no_grad():
                gold_embeddings, negative_embeddings, _ = miner.mine(
                    query_embeddings.detach(),
                    gold_doc_ids,
                    num_negatives=NUM_HARD_NEGATIVES
                )
            
            # Compute loss
            loss = criterion(query_embeddings, gold_embeddings, negative_embeddings)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(query_encoder.encoder.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if TQDM_AVAILABLE:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/num_batches:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            elif batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx}: loss={loss.item():.4f}")
            
            if global_step % SAVE_EVERY_N_STEPS == 0:
                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR, 
                    f"adore_step_{global_step}.pt"
                )
                query_encoder.save(checkpoint_path)
        
        # Epoch summary - compute both training and validation loss
        avg_train_loss = epoch_loss / max(num_batches, 1)
        
        # Compute validation loss for checkpoint selection
        logger.info(f"  Computing validation loss...")
        val_loss = compute_validation_loss(
            query_encoder, miner, criterion, 
            val_samples, batch_size, quick_mode
        )
        
        # Log losses
        train_losses.append(avg_train_loss)
        test_losses.append(val_loss)
        epoch_numbers.append(epoch + 1)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_test_loss:
            improvement = best_test_loss - val_loss
            best_test_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "adore_best.pt")
            query_encoder.save(best_path)
            logger.info(f"  New best model (by TEST loss)! Loss: {best_test_loss:.4f} (improved by {improvement:.4f})")
            epochs_without_improvement = 0
            
            if not first_best_saved:
                first_best_saved = True
                epochs_since_first_best = 0
                logger.info(f"  First best model saved - Bjorn countdown starts")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement}/{PATIENCE} epochs (Val Loss: {val_loss:.4f} vs Best: {best_test_loss:.4f})")
            
            if first_best_saved:
                epochs_since_first_best += 1
                
                if epochs_since_first_best == BJORN_PATIENCE and not bjorn_saved:
                    bjorn_path = os.path.join(CHECKPOINT_DIR, "adore_bjorn.pt")
                    query_encoder.save(bjorn_path)
                    bjorn_saved = True
                    logger.info(f"  Bjorn checkpoint saved after {BJORN_PATIENCE} epochs without improvement")
                    logger.info(f"     Path: {bjorn_path}")
        
        epoch_path = os.path.join(CHECKPOINT_DIR, f"adore_epoch_{epoch+1}.pt")
        query_encoder.save(epoch_path)
        
        if train_losses and test_losses:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            graph_path = os.path.join(RESULTS_DIR, "adore_loss_graph.png")
            generate_loss_graphs(train_losses, test_losses, epoch_numbers, graph_path)
        
        old_epoch = epoch + 1 - 3
        if old_epoch > 0:
            old_path = os.path.join(CHECKPOINT_DIR, f"adore_epoch_{old_epoch}.pt")
            if os.path.exists(old_path):
                os.remove(old_path)
        
        if quick_mode:
            logger.info("Quick mode: stopping after 1 epoch")
            break
        
        if epochs_without_improvement >= PATIENCE:
            logger.info(f"")
            logger.info(f"⏹️  Early stopping triggered after {PATIENCE} epochs without improvement")
            logger.info(f"   Best validation loss achieved: {best_test_loss:.4f}")
            break
    
    except KeyboardInterrupt:
        training_interrupted = True
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING INTERRUPTED BY USER (Ctrl+C)")
        logger.info("=" * 60)
    
    final_path = os.path.join(CHECKPOINT_DIR, "adore_final.pt")
    query_encoder.save(final_path)
    
    loss_log = {
        "epochs": epoch_numbers,
        "train_losses": train_losses,
        "val_losses": test_losses,
        "best_val_loss": best_test_loss,
        "best_epoch": epoch_numbers[test_losses.index(min(test_losses))] if test_losses else None,
        "bjorn_saved": bjorn_saved,
        "training_interrupted": training_interrupted
    }
    loss_log_path = os.path.join(RESULTS_DIR, "adore_training_log.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(loss_log_path, 'w') as f:
        json.dump(loss_log, f, indent=2)
    logger.info(f"Training log saved to: {loss_log_path}")
    
    # Generate loss graphs
    if train_losses and test_losses:
        graph_path = os.path.join(RESULTS_DIR, "adore_loss_graph.png")
        generate_loss_graphs(train_losses, test_losses, epoch_numbers, graph_path)
    
    logger.info("")
    logger.info("=" * 60)
    if training_interrupted:
        logger.info("TRAINING STOPPED BY USER")
    else:
        logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_test_loss:.4f}")
    logger.info(f"Best model: {os.path.join(CHECKPOINT_DIR, 'adore_best.pt')}")
    if bjorn_saved:
        logger.info(f"Bjorn model: {os.path.join(CHECKPOINT_DIR, 'adore_bjorn.pt')}")
    logger.info(f"Final model: {final_path}")
    logger.info(f"Checkpoints: {CHECKPOINT_DIR}")
    logger.info(f"Loss graph: {os.path.join(RESULTS_DIR, 'adore_loss_graph.png')}")
    logger.info("=" * 60)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_retrieval(
    query_encoder: QueryEncoder,
    corpus_embeddings: torch.Tensor,
    corpus_ids: List[str],
    dev_data: List[Dict],
    title_to_docids: Dict[str, List[str]],
    k_values: List[int] = K_VALUES,
    quick_mode: bool = False
) -> Dict[str, float]:
    logger.info("="*60)
    logger.info("RETRIEVAL EVALUATION")
    logger.info("="*60)
    
    if quick_mode:
        dev_data = dev_data[:QUICK_TEST_QUESTIONS]
        logger.info(f"Quick mode: evaluating on {len(dev_data)} questions")
    
    query_encoder.encoder.eval()
    
    # Move corpus embeddings to device and normalize
    corpus_emb = corpus_embeddings.to(DEVICE)
    corpus_emb_norm = F.normalize(corpus_emb, p=2, dim=1)
    
    # Prepare queries and gold docs
    queries = [item['question'] for item in dev_data]
    
    # Extract gold doc IDs for each question
    gold_doc_ids_list = []
    for item in dev_data:
        gold_ids = set()
        for fact in item.get("supporting_facts", []):
            if isinstance(fact, list) and len(fact) >= 1:
                title = fact[0]
                if title in title_to_docids:
                    gold_ids.update(title_to_docids[title])
        gold_doc_ids_list.append(gold_ids)
    
    # Encode queries
    logger.info("Encoding queries...")
    with torch.no_grad():
        query_embeddings = query_encoder.encode_queries(queries, batch_size=32)
        query_emb_norm = F.normalize(query_embeddings, p=2, dim=1)
    
    # Compute similarities
    logger.info("Computing similarities...")
    similarities = torch.matmul(query_emb_norm, corpus_emb_norm.t())
    
    # Evaluate for each k
    results = {}
    
    for k in k_values:
        # Get top-k predictions
        top_k_scores, top_k_indices = torch.topk(similarities, k, dim=1)
        
        # Compute recall
        hits = 0
        for i, gold_ids in enumerate(gold_doc_ids_list):
            if not gold_ids:
                continue
            
            predicted_ids = [corpus_ids[idx] for idx in top_k_indices[i].tolist()]
            
            # Check if any gold doc is in top-k
            if any(pid in gold_ids for pid in predicted_ids):
                hits += 1
        
        recall = hits / len(dev_data) if dev_data else 0
        results[f"recall@{k}"] = recall
        logger.info(f"  Recall@{k}: {recall:.4f} ({hits}/{len(dev_data)})")
    
    return results


def run_rag_evaluation(
    args,
    query_encoder: QueryEncoder,
    corpus_embeddings: torch.Tensor,
    corpus_ids: List[str],
    corpus_dict: Dict[str, Dict],
    dev_data: List[Dict],
    title_to_docids: Dict[str, List[str]],
    k_values: List[int] = K_VALUES,
    quick_mode: bool = False
) -> Dict[str, Any]:
    if quick_mode:
        dev_data = dev_data[:QUICK_TEST_QUESTIONS]
    
    num_questions = len(dev_data)
    
    # Check for Groq API key
    paid_api_key = os.environ.get("PAID_API_KEY", "")
    
    if not paid_api_key or paid_api_key == "your_paid_groq_api_key_here":
        logger.warning("No Groq API key found. Skipping LLM evaluation.")
        logger.info("Set PAID_API_KEY in .env file for LLM-based evaluation.")
        return evaluate_retrieval(
            query_encoder, corpus_embeddings, corpus_ids, 
            dev_data, title_to_docids, k_values, quick_mode=False
        )
    
    # Initialize LLM
    from dexter.llms.groq_engine import GroqEngine
    print("\n   Using Groq API with Llama 3.1 8B Instant")
    llm_engine = GroqEngine(
        data="",
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )
    
    query_encoder.encoder.eval()
    
    # Pre-compute retrieval results for all queries
    print(f"\nPre-computing retrieval for {num_questions} questions...")
    
    # Move corpus embeddings to device and normalize
    corpus_emb = corpus_embeddings.to(DEVICE)
    corpus_emb_norm = F.normalize(corpus_emb, p=2, dim=1)
    
    # Prepare queries
    queries = [item['question'] for item in dev_data]
    
    # Encode queries
    print("   Encoding queries...")
    with torch.no_grad():
        query_embeddings = query_encoder.encode_queries(queries, batch_size=32)
        query_emb_norm = F.normalize(query_embeddings, p=2, dim=1)
    
    # Compute similarities
    print("   Computing similarities...")
    similarities = torch.matmul(query_emb_norm, corpus_emb_norm.t())
    
    # Get top-k for maximum k
    max_k = max(k_values)
    top_k_scores, top_k_indices = torch.topk(similarities, max_k, dim=1)
    top_k_indices = top_k_indices.cpu()
    
    retrieval_results = {}
    for idx, item in enumerate(dev_data):
        qid = item.get("_id", str(idx))
        retrieval_results[str(qid)] = {}
        for rank, doc_idx in enumerate(top_k_indices[idx].tolist()):
            doc_id = corpus_ids[doc_idx]
            score = float(top_k_scores[idx][rank])
            retrieval_results[str(qid)][doc_id] = score
    
    print(f"   Retrieval complete for {num_questions} queries")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results_summary = {}
    
    for k in k_values:
        k_results = run_generation_for_k(
            k=k,
            dev_data=dev_data,
            retrieval_results=retrieval_results,
            corpus_dict=corpus_dict,
            title_to_docids=title_to_docids,
            llm_engine=llm_engine,
            checkpoint_dir=RESULTS_DIR,
            quick_mode=quick_mode
        )
        results_summary[k] = k_results
    
    print("\n" + "="*60)
    print("EXPERIMENT 5 RESULTS SUMMARY")
    print("="*60)
    print(f"\nDataset: MusiqueQA (first {num_questions} questions)")
    print(f"Retriever: ADORE-trained Query Encoder (facebook/contriever base)")
    print(f"LLM: Llama 3.1 8B (Groq)")
    print(f"\nExact Match Scores:")
    print("-" * 50)
    for k, stats in results_summary.items():
        em = stats["exact_match"]
        gold_rate = stats["gold_recall_rate"]
        gold_em = stats["gold_em_when_retrieved"]
        print(f"  k={k}: EM={em:.4f} ({em*100:.2f}%) | Gold Recall={gold_rate:.2%} | EM@Gold={gold_em:.4f}")
    print("-" * 50)
    
    suffix = "_quick" if quick_mode else ""
    summary_path = os.path.join(RESULTS_DIR, f"experiment5_summary{suffix}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "Experiment 5: ADORE Dense Retrieval Training",
            "num_questions": num_questions,
            "retriever": "ADORE-trained Query Encoder (facebook/contriever base)",
            "llm": "llama-3.1-8b-instant",
            "k_values": k_values,
            "results": {str(k): v for k, v in results_summary.items()},
            "quick_mode": quick_mode,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    
    return results_summary


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


def run_generation_for_k(
    k: int,
    dev_data: List[Dict],
    retrieval_results: Dict,
    corpus_dict: Dict[str, Dict],
    title_to_docids: Dict[str, List[str]],
    llm_engine,
    checkpoint_dir: str,
    quick_mode: bool = False
) -> dict:
    print(f"\n{'='*60}")
    print(f"GENERATION WITH k={k}")
    print(f"{'='*60}")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"exp5_checkpoint_k{k}.json")
    total_questions = len(dev_data)
    
    # Get indices to process (validates checkpoint completeness)
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
    
    if TQDM_AVAILABLE:
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
        item = dev_data[idx]
        question_id = item.get("_id", str(idx))
        question_text = item['question']
        gold_text = item['answer']
        
        # Compute golden document stats
        gold_doc_ids = get_gold_doc_ids(item, title_to_docids)
        gold_stats = compute_gold_doc_stats(str(question_id), retrieval_results, gold_doc_ids, k)
        
        # Get context from retrieved docs
        qid_results = retrieval_results.get(str(question_id), {})
        sorted_docs = sorted(qid_results.items(), key=lambda x: x[1], reverse=True)[:k]
        
        contexts = []
        for doc_id, score in sorted_docs:
            if doc_id in corpus_dict:
                doc = corpus_dict[doc_id]
                text = doc.get('text', '')
                title = doc.get('title', '')
                if title:
                    contexts.append(f"{title}: {text}")
                else:
                    contexts.append(text)
        
        context = "\n\n".join(contexts)
        
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
            
            if TQDM_AVAILABLE and hasattr(iterator, 'set_postfix'):
                current_em = matches / total if total > 0 else 0
                iterator.set_postfix(EM=f"{current_em:.3f}", correct=matches)
            elif not TQDM_AVAILABLE and (idx + 1) % 10 == 0:
                current_em = matches / total if total > 0 else 0
                print(f"   [{idx+1}/{len(dev_data)}] EM: {current_em:.4f} ({matches}/{total})")
            
            if processed_count % CHECKPOINT_FREQUENCY == 0:
                checkpoint["k"] = k
                save_checkpoint(checkpoint_path, checkpoint, silent=TQDM_AVAILABLE)
                if TQDM_AVAILABLE:
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
    
    checkpoint["k"] = k
    save_checkpoint(checkpoint_path, checkpoint, silent=True)
    
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
    
    final_path = os.path.join(checkpoint_dir, f"exp5_results_k{k}.json")
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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5: ADORE - Dense Retrieval Training with Hard Negatives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train (100 epochs max, Ctrl+C to stop early):
    python experiment5_adore_training.py --train
    
  Train with custom epochs:
    python experiment5_adore_training.py --train --epochs 50
    
  Resume training from best checkpoint:
    python experiment5_adore_training.py --train --resume
    
  Test with best model:
    python experiment5_adore_training.py --test
    
  Quick smoke test (CPU debugging):
    python experiment5_adore_training.py --train --quick
    python experiment5_adore_training.py --test --quick
    
  Train and then test:
    python experiment5_adore_training.py --train --test
    
Checkpoints saved to: experiments/checkpoints/
  - adore_best.pt   : Best model (use this one!)
  - adore_final.pt  : Final model after training
  - adore_epoch_N.pt: Last 3 epoch checkpoints
        """
    )
    
    parser.add_argument('--train', action='store_true',
                        help='Train the query encoder using ADORE method')
    parser.add_argument('--test', action='store_true',
                        help='Evaluate on dev.json (first 1200 questions)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode for smoke testing on CPU')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to load for testing')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the best checkpoint (adore_best.pt)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM evaluation in test mode (retrieval metrics only)')
    
    args = parser.parse_args()
    
    if not args.train and not args.test:
        parser.print_help()
        print("\nError: Please specify --train and/or --test mode")
        return
    
    # Use argument values (override defaults)
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    
    print("="*60)
    print("EXPERIMENT 5: ADORE Dense Retrieval Training")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Quick mode: {args.quick}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    
    # Load corpus embeddings
    try:
        corpus_embeddings, corpus_ids = load_corpus_embeddings()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Initialize query encoder
    query_encoder = QueryEncoder(BASE_MODEL, DEVICE)
    
    # Load checkpoint if specified
    if args.checkpoint:
        query_encoder.load(args.checkpoint)
    elif args.test and not args.train:
        # Try to load best model for testing
        best_path = os.path.join(CHECKPOINT_DIR, "adore_best.pt")
        final_path = os.path.join(CHECKPOINT_DIR, "adore_final.pt")
        
        if os.path.exists(best_path):
            query_encoder.load(best_path)
        elif os.path.exists(final_path):
            query_encoder.load(final_path)
        else:
            logger.warning("No trained checkpoint found. Using base model.")
    
    # Training mode
    if args.train:
        train_samples = load_training_data(limit=None if not args.quick else 100)
        
        # Determine resume checkpoint
        resume_path = None
        if args.resume:
            best_path = os.path.join(CHECKPOINT_DIR, "adore_best.pt")
            if os.path.exists(best_path):
                resume_path = best_path
                logger.info(f"Will resume training from: {resume_path}")
            else:
                logger.info("No best checkpoint found, starting fresh training")
        
        train_adore(
            args=args,
            query_encoder=query_encoder,
            corpus_embeddings=corpus_embeddings,
            corpus_ids=corpus_ids,
            train_samples=train_samples,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            quick_mode=args.quick,
            resume_from=resume_path
        )
    
    # Testing mode
    if args.test:
        dev_data, title_to_docids = load_dev_data(
            limit=QUICK_TEST_QUESTIONS if args.quick else NUM_TEST_QUESTIONS
        )
        
        if args.no_llm:
            # Retrieval-only evaluation
            evaluate_retrieval(
                query_encoder=query_encoder,
                corpus_embeddings=corpus_embeddings,
                corpus_ids=corpus_ids,
                dev_data=dev_data,
                title_to_docids=title_to_docids,
                k_values=K_VALUES,
                quick_mode=args.quick
            )
        else:
            # Full RAG evaluation with LLM
            corpus_dict = load_corpus_dict()
            run_rag_evaluation(
                args=args,
                query_encoder=query_encoder,
                corpus_embeddings=corpus_embeddings,
                corpus_ids=corpus_ids,
                corpus_dict=corpus_dict,
                dev_data=dev_data,
                title_to_docids=title_to_docids,
                k_values=K_VALUES,
                quick_mode=args.quick
            )
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
