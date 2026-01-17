from transformers import AutoTokenizer, RobertaForQuestionAnswering
import torch
from ..BaseRetriever import BaseRetriver
from tqdm import tqdm
import logging
import heapq

class ADORERetriever(BaseRetriver):
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32, chunk_size: int = 200000):
        super().__init__()
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('zycao/star')
        self.model = RobertaForQuestionAnswering.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.model.config.output_hidden_states = True  # Enable hidden states
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

    def encode_queries(self, queries):
        # Extract text from each query object
        query_texts = [query.text() for query in queries]
        
        # Process in batches
        embeddings_list = []
        for i in range(0, len(query_texts), self.batch_size):
            batch_texts = query_texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(self.device)
            
            # Forward pass to get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the hidden states
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            batch_embeddings = hidden_states.mean(dim=1)  # Mean pooling across tokens
            embeddings_list.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings

    def encode_context(self, contexts):
        # Extract text from each evidence object
        context_texts = [evidence.text() for evidence in contexts]
        
        # Process in batches
        embeddings_list = []
        pbar = tqdm(total=len(context_texts))
        self.logger.info("Starting encoding of contexts...")
        
        for i in range(0, len(context_texts), self.batch_size):
            batch_texts = context_texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(self.device)
            
            # Forward pass to get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the hidden states
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            batch_embeddings = hidden_states.mean(dim=1)  # Mean pooling across tokens
            embeddings_list.append(batch_embeddings)
            
            pbar.update(len(batch_texts))
        
        pbar.close()
        # Concatenate all batches
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings

    def train(self, queries, contexts, labels):
        # Implement training loop here if needed
        pass

    def retrieve_in_chunks(self, corpus, queries, max_k, similarity_measure):
        self.logger.info("Starting retrieval in chunks...")
        corpus_ids = [doc.id() for doc in corpus]
        query_embeddings = self.encode_queries(queries).to(self.device)
        query_ids = [query.id() for query in queries]
        result_heaps = {qid: [] for qid in query_ids}
        
        # Process corpus in chunks to avoid memory issues
        for chunk_start in range(0, len(corpus), self.chunk_size):
            self.logger.info(f"Processing chunk {chunk_start//self.chunk_size + 1}/{(len(corpus)-1)//self.chunk_size + 1}")
            chunk_end = min(chunk_start + self.chunk_size, len(corpus))
            
            # Encode chunk of corpus
            chunk_corpus = corpus[chunk_start:chunk_end]
            chunk_embeddings = self.encode_context(chunk_corpus).to(self.device)
            
            # Process similarity computation in batches
            similarity_batch_size = 128  # Adjust based on GPU memory
            for i in range(0, len(queries), similarity_batch_size):
                batch_end = min(i + similarity_batch_size, len(queries))
                batch_query_embeddings = query_embeddings[i:batch_end]
                
                # Compute similarities for this batch
                cos_sim = torch.nn.functional.cosine_similarity(
                    batch_query_embeddings.unsqueeze(1),
                    chunk_embeddings.unsqueeze(0),
                    dim=2
                )
                
                # Get top-k values for this batch
                chunk_top_k = min(max_k, cos_sim.size(1))
                cos_scores_top_k_values, cos_scores_top_k_idx = cos_sim.topk(chunk_top_k, dim=1)
                
                # Move to CPU for processing
                cos_scores_top_k_values = cos_scores_top_k_values.cpu()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu()
                
                # Update result heaps for this batch
                for batch_idx, query_idx in enumerate(range(i, batch_end)):
                    query_id = query_ids[query_idx]
                    for score_idx, corpus_idx in enumerate(cos_scores_top_k_idx[batch_idx]):
                        corpus_id = corpus_ids[chunk_start + corpus_idx.item()]
                        score = cos_scores_top_k_values[batch_idx][score_idx].item()
                        
                        if len(result_heaps[query_id]) < max_k:
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
                
                # Clear GPU memory
                del cos_sim, cos_scores_top_k_values, cos_scores_top_k_idx
                torch.cuda.empty_cache()
            
            # Clear chunk embeddings from GPU memory
            del chunk_embeddings
            torch.cuda.empty_cache()
        
        # Prepare final results
        results = {}
        for query_id in query_ids:
            results[query_id] = {}
            for score, corpus_id in sorted(result_heaps[query_id], reverse=True):
                results[query_id][corpus_id] = score
                
        return results

    def retrieve(self, corpus, queries, max_k, similarity_measure):
        return self.retrieve_in_chunks(corpus, queries, max_k, similarity_measure)
