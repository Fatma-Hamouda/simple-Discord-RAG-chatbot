"""
IMPROVED Document Ingestion Script
"""

import os
import sys
from pathlib import Path
from rag_agent import (
    ImprovedEmbeddingModel,
    HybridVectorStore,
    ImprovedLLMClient,
    ImprovedAzureLLMClient,
    ImprovedRAGAgent
)
from dotenv import load_dotenv
import logging

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pdf_documents(pdf_directory='docs'):
    """Load PDF documents from a directory."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Install with: pip install pypdf")
        sys.exit(1)
    
    documents = []
    doc_names = []
    
    pdf_dir = Path(pdf_directory)
    
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    if not pdf_files:
        logger.warning(f"No PDF files found in: {pdf_dir.resolve()}")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading: {pdf_file.name}")
            reader = PdfReader(str(pdf_file))
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                # Add page markers for better context
                text += f"\n[Page {page_num + 1}]\n{page_text}\n"
            
            if text.strip():
                documents.append(text)
                doc_names.append(pdf_file.stem)
                logger.info(f"  ✓ Loaded {len(text)} characters from {len(reader.pages)} pages")
            else:
                logger.warning(f"  ✗ No text extracted from {pdf_file.name}")
                
        except Exception as e:
            logger.error(f"  ✗ Error loading {pdf_file.name}: {e}")

    
    return documents, doc_names


def main():
    """Main ingestion function with IMPROVED processing."""
    print("=" * 80)
    print("IMPROVED DISCORD RAG BOT - DOCUMENT INGESTION")
    print("Using better embeddings and hybrid search for higher accuracy")
    print("=" * 80)
    print()
    
    # Load documents
    logger.info("Loading documents...")
    documents, doc_names = load_pdf_documents()
    
    logger.info(f"✓ Loaded {len(documents)} documents:")
    for i, name in enumerate(doc_names, 1):
        doc_length = len(documents[i-1])
        logger.info(f"  {i}. {name} ({doc_length:,} characters)")
    print()

    if not documents:
        logger.error("No documents were loaded. Add PDFs to the docs/ folder and try again.")
        return
    
    # Initialize IMPROVED components
    logger.info("Initializing IMPROVED RAG components...")
    logger.info("  • Using all-mpnet-base-v2 (768-dim) instead of all-MiniLM-L6-v2 (384-dim)")
    logger.info("  • Larger chunks (800 chars) with more overlap (200 chars)")
    logger.info("  • Hybrid search (semantic + keyword matching)")
    
    embedding_model = ImprovedEmbeddingModel(model_name='all-mpnet-base-v2')
    vector_store = HybridVectorStore(dimension=embedding_model.dimension)
    
    # Initialize LLM
    llm_type = os.getenv('LLM_TYPE', 'ollama')
    
    if llm_type == 'azure':
        llm_client = ImprovedAzureLLMClient(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            api_key=os.getenv('AZURE_API_KEY'),
            deployment_name=os.getenv('AZURE_DEPLOYMENT')
        )
    else:
        llm_client = ImprovedLLMClient(
            base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'),
            model=os.getenv('OLLAMA_MODEL', 'llama2')
        )
    
    # Create IMPROVED RAG agent
    rag_agent = ImprovedRAGAgent(embedding_model, vector_store, llm_client)
    print()
    
    # Ingest documents
    rag_agent.ingest_documents(documents, doc_names)
    
    # Save index with improved suffix
    logger.info("Saving IMPROVED vector index...")
    rag_agent.save('faiss_index_improved.bin', 'chunks_improved.pkl')
    logger.info("✓ Index saved successfully!")
    print()
    
    # Test queries
    print("=" * 80)
    print("TESTING IMPROVED RAG SYSTEM")
    print("=" * 80)
    print()
    
    test_queries = [
        "What are the two phases for data scientist role of Discord RAG FQA chatbot assignment?",
        "What is this AI internship about ?",
        "How should I handle errors in the Discord bot?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Q: {query}")
        print('='*80)
        result = rag_agent.query(query, top_k=5)
        print(f"\nA: {result['answer']}")
        print(f"\nRelevance Score: {result.get('avg_relevance', 0):.3f}")
        print(f"Sources: {result['num_sources']}")
        print()


if __name__ == "__main__":
    main()
