"""
Enhanced Discord RAG Bot - Improved RAG Agent with Better Accuracy
Implements advanced techniques for more relevant and accurate responses
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedDocumentChunker:
    """Enhanced document chunking with better context preservation."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Larger chunks with more overlap for better context.
        
        Args:
            chunk_size: Increased to 800 for more context
            chunk_overlap: Increased to 200 to preserve connections
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        logger.info(f"ImprovedChunker initialized (size={chunk_size}, overlap={chunk_overlap})")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and common artifacts
        text = re.sub(r'\n\d+\n', '\n', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        return text.strip()
    
    def chunk_documents(self, documents: List[str], doc_names: Optional[List[str]] = None) -> List[Dict]:
        """Split documents with preprocessing."""
        all_chunks = []
        
        if doc_names is None:
            doc_names = [f"Document_{i}" for i in range(len(documents))]
        
        for doc_id, doc in enumerate(documents):
            # Preprocess document
            cleaned_doc = self.preprocess_text(doc)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_doc)
            
            for chunk_id, chunk in enumerate(chunks):
                # Add contextual information
                all_chunks.append({
                    'text': chunk,
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'doc_name': doc_names[doc_id] if doc_id < len(doc_names) else f"Document_{doc_id}",
                    'char_count': len(chunk),
                    'word_count': len(chunk.split())
                })
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


class ImprovedEmbeddingModel:
    """Enhanced embedding with query optimization."""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Use a better embedding model for improved accuracy.
        
        all-mpnet-base-v2 is more accurate than all-MiniLM-L6-v2
        Dimension: 768 (vs 384)
        Better semantic understanding
        """
        logger.info(f"Loading improved embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Convert texts to embeddings with normalization."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better comparison
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query with preprocessing.
        Add instructions for better semantic matching.
        """
        # Expand query for better matching
        enhanced_query = f"Query: {query}"
        
        embedding = self.model.encode(
            [enhanced_query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        return embedding


class HybridVectorStore:
    """Enhanced vector store with hybrid search (dense + keyword)."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Use Inner Product for normalized vectors
        self.chunks = []
        logger.info(f"HybridVectorStore initialized with dimension {dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Add embeddings to the index."""
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")
    
    def keyword_score(self, query: str, chunk_text: str) -> float:
        """Calculate keyword-based relevance score."""
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & chunk_words
        union = query_words | chunk_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def search(self, query_embedding: np.ndarray, query_text: str, k: int = 5) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query_embedding: Dense vector representation
            query_text: Original query text for keyword matching
            k: Number of results (retrieve more, then re-rank)
        """
        if len(self.chunks) == 0:
            logger.warning("Vector store is empty!")
            return []
        
        # Retrieve more candidates for re-ranking
        k_retrieval = min(k * 3, len(self.chunks))
        
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, k_retrieval)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk_copy = self.chunks[idx].copy()
                
                # Semantic similarity (cosine via inner product)
                semantic_score = float(distances[0][i])
                
                # Keyword similarity
                keyword_score = self.keyword_score(query_text, chunk_copy['text'])
                
                # Hybrid score (weighted combination)
                hybrid_score = 0.7 * semantic_score + 0.3 * keyword_score
                
                chunk_copy['semantic_score'] = semantic_score
                chunk_copy['keyword_score'] = keyword_score
                chunk_copy['hybrid_score'] = hybrid_score
                chunk_copy['rank'] = i + 1
                
                results.append(chunk_copy)
        
        # Re-rank by hybrid score
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Return top k
        final_results = results[:k]
        
        # Update ranks
        for i, result in enumerate(final_results):
            result['rank'] = i + 1
        
        logger.info(f"Retrieved {len(final_results)} chunks with hybrid search")
        return final_results
    
    def save(self, index_path: str = 'faiss_index.bin', chunks_path: str = 'chunks.pkl'):
        """Save index and chunks to disk."""
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved index to {index_path}")
    
    def load(self, index_path: str = 'faiss_index.bin', chunks_path: str = 'chunks.pkl'):
        """Load index and chunks from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        logger.info(f"Loaded index with {len(self.chunks)} chunks")


class ImprovedLLMClient:
    """Enhanced LLM client with better prompting."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        import requests
        self.base_url = base_url
        self.model = model
        self.requests = requests
        logger.info(f"Improved LLM client initialized (model={model})")
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate answer with improved prompting technique.
        
        Uses:
        - Chain-of-thought prompting
        - Explicit instructions
        - Context formatting
        - Grounding techniques
        """
        # Format context with better structure
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(
                f"[Document: {chunk.get('doc_name', 'Unknown')}]\n"
                f"Relevance Score: {chunk.get('hybrid_score', 0):.2f}\n"
                f"Content: {chunk['text']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Improved prompt with chain-of-thought
        prompt = f"""You are a helpful AI assistant for an AI Engineering Bootcamp. Your job is to answer questions accurately using ONLY the information provided in the context below.

CONTEXT DOCUMENTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Read the context documents carefully
2. Identify which documents contain relevant information for the question
3. Synthesize the information from relevant documents
4. Provide a clear, accurate answer based ONLY on the context
5. If the context doesn't contain enough information, say: "I don't have enough information in the provided documents to fully answer this question."
6. Cite which document(s) you used (e.g., "According to the Project Overview...")
7. Be concise but complete
8. Use a helpful, professional tone

ANSWER:"""
        
        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more focused answers
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 600,  # Allow longer responses
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Ollama returned status {response.status_code}"
        
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error connecting to Ollama: {str(e)}"


class ImprovedAzureLLMClient:
    """Enhanced Azure client with better prompting."""
    
    def __init__(self, endpoint: str, api_key: str, deployment_name: str):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )
        self.deployment = deployment_name
        logger.info(f"Improved Azure LLM client initialized")
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate with improved prompting."""
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(
                f"[Document: {chunk.get('doc_name', 'Unknown')} | "
                f"Relevance: {chunk.get('hybrid_score', 0):.2f}]\n"
                f"{chunk['text']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        system_prompt = """You are a precise AI assistant for an AI Engineering Bootcamp. You provide accurate, helpful answers based strictly on the provided documentation. You cite your sources and admit when you don't have enough information."""

        user_prompt = f"""CONTEXT DOCUMENTS:
{context}

QUESTION: {query}

Provide a clear, accurate answer using ONLY the information above. Cite which document you're referencing. If the information isn't available, say so clearly.

ANSWER:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower for more accuracy
                max_tokens=600,
                top_p=0.9
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Azure LLM error: {e}")
            return f"Error generating response: {str(e)}"


class ImprovedRAGAgent:
    """Enhanced RAG agent with better accuracy and relevance."""
    
    def __init__(self, embedding_model, vector_store, llm_client):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.chunker = ImprovedDocumentChunker()
        logger.info("Improved RAG Agent initialized")
    
    def ingest_documents(self, documents: List[str], doc_names: Optional[List[str]] = None):
        """Ingest documents with improved processing."""
        logger.info(f"Starting document ingestion for {len(documents)} documents")
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents, doc_names)

        if not chunks:
            logger.warning("No chunks created. Skipping embedding and indexing.")
            return
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(chunk_texts)
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, chunks)
        
        logger.info("Document ingestion complete")
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Answer query with improved retrieval and generation.
        
        Key improvements:
        - More retrieved documents (5 instead of 3)
        - Hybrid search (semantic + keyword)
        - Better prompting
        - Lower temperature for accuracy
        """
        logger.info(f"Processing query: {question}")
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(question)
        
        # Hybrid search (semantic + keyword)
        retrieved_chunks = self.vector_store.search(
            query_embedding, 
            question,  # Pass original query for keyword matching
            k=top_k
        )
        
        # Log retrieval quality
        if retrieved_chunks:
            avg_score = sum(c['hybrid_score'] for c in retrieved_chunks) / len(retrieved_chunks)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks, avg score: {avg_score:.3f}")
        
        # Generate answer
        answer = self.llm_client.generate_answer(question, retrieved_chunks)
        
        return {
            'answer': answer,
            'sources': retrieved_chunks,
            'query': question,
            'num_sources': len(retrieved_chunks),
            'avg_relevance': sum(c['hybrid_score'] for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0
        }
    
    def save(self, index_path: str = 'faiss_index_improved.bin', 
             chunks_path: str = 'chunks_improved.pkl'):
        """Save vector store."""
        self.vector_store.save(index_path, chunks_path)
    
    def load(self, index_path: str = 'faiss_index_improved.bin', 
             chunks_path: str = 'chunks_improved.pkl'):
        """Load vector store."""
        self.vector_store.load(index_path, chunks_path)
