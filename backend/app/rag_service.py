from typing import List, Dict, Optional
from .qdrant_service import QdrantService
from .embeddings import EmbeddingService


class RAGService:
    """Retrieval Augmented Generation service"""
    
    def __init__(self):
        self.qdrant = QdrantService()
        self.embeddings = EmbeddingService()
    
    async def search_knowledge_base(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict]:
        """Search knowledge base for relevant information"""
        # Generate query embedding
        query_embedding = await self.embeddings.generate_embedding(query)
        
        # Search Qdrant
        results = self.qdrant.search(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return results
    
    async def build_context(
        self, 
        query: str, 
        max_results: int = 3
    ) -> str:
        """Build context from knowledge base for RAG"""
        results = await self.search_knowledge_base(query, limit=max_results)
        
        if not results:
            return ""
        
        # Format results into context
        context_parts = []
        for idx, result in enumerate(results, 1):
            text = result["text"]
            score = result["score"]
            context_parts.append(f"[Sumber {idx}] (Relevansi: {score:.2f})\n{text}")
        
        context = "\n\n".join(context_parts)
        return f"Informasi dari knowledge base:\n\n{context}"
    
    async def add_to_knowledge_base(
        self, 
        documents: List[Dict[str, str]]
    ) -> int:
        """
        Add documents to knowledge base
        Each document should have: {text, metadata (optional)}
        """
        # Generate embeddings for all documents
        texts = [doc["text"] for doc in documents]
        embeddings = await self.embeddings.generate_embeddings(texts)
        
        # Prepare documents with embeddings
        docs_with_embeddings = []
        for doc, embedding in zip(documents, embeddings):
            docs_with_embeddings.append({
                "text": doc["text"],
                "embedding": embedding,
                "metadata": doc.get("metadata", {})
            })
        
        # Upsert to Qdrant
        count = self.qdrant.upsert_documents(docs_with_embeddings)
        return count
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return self.qdrant.get_collection_info()
    
    def initialize_collection(self):
        """Initialize the Qdrant collection"""
        self.qdrant.init_collection(dimension=self.embeddings.dimension)

