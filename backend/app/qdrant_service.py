from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional
import os
import hashlib


class QdrantService:
    """Service for managing Qdrant vector database"""
    
    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.client = QdrantClient(url=self.url)
        self.collection_name = "knowledge_base"
        self.embedding_dimension = 768  # Ollama nomic-embed-text
    
    def init_collection(self, dimension: int = 768):
        """Initialize the knowledge base collection"""
        self.embedding_dimension = dimension
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")
    
    def upsert_documents(self, documents: List[Dict]) -> int:
        """
        Upsert documents into Qdrant
        Each document should have: {id, text, embedding, metadata}
        """
        points = []
        for doc in documents:
            # Generate unique ID if not provided
            doc_id = doc.get("id") or self._generate_id(doc["text"])
            
            point = PointStruct(
                id=doc_id,
                vector=doc["embedding"],
                payload={
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {})
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        return len(points)
    
    def search(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """Search for similar documents"""
        # Validate embedding dimension before searching
        if not query_embedding or len(query_embedding) != self.embedding_dimension:
            error_msg = f"Invalid query embedding dimension: expected {self.embedding_dimension}, got {len(query_embedding) if query_embedding else 0}"
            print(f"Qdrant search error: {error_msg}")
            raise ValueError(error_msg)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {})
            }
            for result in results
        ]
    
    def delete_by_id(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id]
            )
            return True
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID from text"""
        return hashlib.md5(text.encode()).hexdigest()

