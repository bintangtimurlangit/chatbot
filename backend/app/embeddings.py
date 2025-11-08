import httpx
from typing import List
import os


class EmbeddingService:
    """Service for generating text embeddings using Ollama (self-hosted)"""
    
    def __init__(self):
        self.api_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = "nomic-embed-text"  # Fast, multilingual embedding model
        self.dimension = 768
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for text in texts:
                    response = await client.post(
                        f"{self.api_url}/api/embeddings",
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": self.model,
                            "prompt": text
                        }
                    )
                    
                    if response.status_code != 200:
                        print(f"Embedding API error: {response.text}")
                        embeddings.append([0.0] * self.dimension)
                    else:
                        result = response.json()
                        embeddings.append(result["embedding"])
                
                return embeddings
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * self.dimension for _ in texts]

