"""
Discord RAG Bot - Flask API Backend (IMPROVED VERSION)
Uses enhanced RAG techniques for better accuracy
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Import IMPROVED RAG components
from rag_agent import (
    ImprovedEmbeddingModel, 
    HybridVectorStore, 
    ImprovedAzureLLMClient, 
    ImprovedLLMClient,
    ImprovedRAGAgent
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global RAG agent
rag_agent = None

# Metrics
metrics = {
    'total_queries': 0,
    'successful_queries': 0,
    'failed_queries': 0,
    'feedback_received': 0,
    'avg_relevance_scores': [],
    'start_time': datetime.now()
}


def initialize_rag_agent():
    """Initialize the IMPROVED RAG agent."""
    global rag_agent
    
    logger.info("=" * 70)
    logger.info("Initializing IMPROVED RAG Agent...")
    logger.info("=" * 70)
    
    try:
        # Initialize IMPROVED embedding model (better model)
        embedding_model = ImprovedEmbeddingModel(model_name='all-mpnet-base-v2')
        
        # Initialize HYBRID vector store
        vector_store = HybridVectorStore(dimension=embedding_model.dimension)
        
        # Try to load existing index
        try:
            vector_store.load('faiss_index_improved.bin', 'chunks_improved.pkl')
            logger.info("✓ Loaded existing IMPROVED vector store")
        except Exception as e:
            logger.warning(f"Could not load improved index: {e}")
            logger.info("Trying to load old index for migration...")
            try:
                vector_store.load('faiss_index.bin', 'chunks.pkl')
                logger.info("✓ Loaded old index - please re-run ingestion for best results")
            except:
                logger.info("No index found - please run ingest_documents_improved.py")
        
        # Initialize IMPROVED LLM client
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
        logger.info("✓ IMPROVED RAG Agent initialized successfully!")
        logger.info("=" * 70)
        logger.info("IMPROVEMENTS ENABLED:")
        logger.info("  ✓ Better embedding model (all-mpnet-base-v2)")
        logger.info("  ✓ Hybrid search (semantic + keyword)")
        logger.info("  ✓ Larger chunks with more overlap")
        logger.info("  ✓ Improved prompting techniques")
        logger.info("  ✓ Lower temperature for accuracy")
        logger.info("  ✓ More retrieved contexts (5 vs 3)")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG agent: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Discord RAG Bot API (IMPROVED)',
        'agent_initialized': rag_agent is not None,
        'version': '2.0-improved',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/query', methods=['POST'])
def query():
    """
    Main query endpoint with IMPROVED RAG.
    
    Now returns relevance scores and uses hybrid search.
    """
    global metrics
    
    try:
        if rag_agent is None:
            logger.error("RAG agent not initialized")
            return jsonify({
                'error': 'RAG agent not initialized. Please run document ingestion first.'
            }), 500
        
        # Get request data
        data = request.json
        question = data.get('question')
        top_k = data.get('top_k', 5)  # Increased default from 3 to 5
        user_id = data.get('user_id', 'anonymous')
        
        # Validate input
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        logger.info(f"Query from {user_id}: {question}")
        
        # Update metrics
        metrics['total_queries'] += 1
        
        # Query the IMPROVED RAG agent
        result = rag_agent.query(question, top_k=top_k)
        result['timestamp'] = datetime.now().isoformat()
        
        # Track relevance score
        if 'avg_relevance' in result:
            metrics['avg_relevance_scores'].append(result['avg_relevance'])
        
        # Update metrics
        metrics['successful_queries'] += 1
        
        logger.info(f"Query successful. Answer length: {len(result['answer'])} chars, "
                   f"Avg relevance: {result.get('avg_relevance', 0):.3f}")
        
        return jsonify(result), 200
    
    except Exception as e:
        metrics['failed_queries'] += 1
        logger.error(f"Query error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Receive user feedback on answers."""
    global metrics
    
    try:
        data = request.json
        
        # Log feedback
        logger.info(f"Feedback received:")
        logger.info(f"  User: {data.get('user_id', 'anonymous')}")
        logger.info(f"  Query: {data.get('query', 'N/A')}")
        logger.info(f"  Rating: {data.get('feedback', 'N/A')}")
        logger.info(f"  Comment: {data.get('comment', 'None')}")
        logger.info(f"  Relevance: {data.get('relevance', 'N/A')}")
        
        # Update metrics
        metrics['feedback_received'] += 1
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded'
        }), 200
    
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get statistics about the RAG system."""
    try:
        if rag_agent is None:
            return jsonify({'error': 'RAG agent not initialized'}), 500
        
        num_chunks = len(rag_agent.vector_store.chunks)
        uptime = (datetime.now() - metrics['start_time']).total_seconds()
        
        # Calculate average relevance
        avg_relevance = 0
        if metrics['avg_relevance_scores']:
            avg_relevance = sum(metrics['avg_relevance_scores']) / len(metrics['avg_relevance_scores'])
        
        return jsonify({
            'vector_store': {
                'total_chunks': num_chunks,
                'embedding_dimension': rag_agent.vector_store.dimension,
                'model': 'all-mpnet-base-v2 (IMPROVED)',
                'search_type': 'hybrid (semantic + keyword)'
            },
            'metrics': {
                'total_queries': metrics['total_queries'],
                'successful_queries': metrics['successful_queries'],
                'failed_queries': metrics['failed_queries'],
                'feedback_received': metrics['feedback_received'],
                'uptime_seconds': uptime,
                'average_relevance': round(avg_relevance, 3),
                'success_rate': f"{(metrics['successful_queries'] / max(metrics['total_queries'], 1) * 100):.1f}%"
            },
            'improvements': {
                'better_embeddings': 'all-mpnet-base-v2 (768-dim)',
                'hybrid_search': 'semantic + keyword matching',
                'chunk_size': '800 chars (vs 500)',
                'chunk_overlap': '200 chars (vs 50)',
                'retrieval_count': '5 docs (vs 3)',
                'temperature': '0.3 (vs 0.7) for accuracy'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ingest', methods=['POST'])
def ingest():
    """Ingest new documents (admin endpoint)."""
    try:
        if rag_agent is None:
            return jsonify({'error': 'RAG agent not initialized'}), 500
        
        data = request.json
        documents = data.get('documents')
        doc_names = data.get('doc_names')
        
        if not documents or not isinstance(documents, list):
            return jsonify({'error': 'Documents must be a non-empty list'}), 400
        
        logger.info(f"Ingesting {len(documents)} new documents with IMPROVED processing")
        
        # Ingest documents
        rag_agent.ingest_documents(documents, doc_names)
        
        # Save the updated index
        rag_agent.save()
        
        logger.info("Document ingestion complete")
        
        return jsonify({
            'status': 'success',
            'message': f'Ingested {len(documents)} documents with improved processing'
        }), 200
    
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize on startup
    initialize_rag_agent()
    
    # Run the API
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info("\n" + "=" * 70)
    logger.info("Discord RAG Bot API Starting (IMPROVED VERSION)...")
    logger.info("=" * 70)
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    logger.info("\nAPI Endpoints:")
    logger.info("  GET  /health          - Health check")
    logger.info("  POST /api/query       - Query the IMPROVED RAG system")
    logger.info("  POST /api/feedback    - Submit user feedback")
    logger.info("  POST /api/ingest      - Ingest new documents")
    logger.info("  GET  /api/stats       - Get system statistics")
    logger.info("=" * 70 + "\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
