import httpx
from typing import List
import os


class EmbeddingService:
    """Service for generating text embeddings using Ollama (self-hosted)"""
    
    def __init__(self):
        self.api_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = "nomic-embed-text"  # Fast, multilingual embedding model
        self.dimension = 768
    
    async def check_ollama_ready(self) -> bool:
        """Check if Ollama is ready and model is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Ollama is responding
                response = await client.get(f"{self.api_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    # Check if model exists (with or without :latest tag)
                    model_found = any(
                        name == self.model or 
                        name == f"{self.model}:latest" or 
                        name.startswith(f"{self.model}:")
                        for name in model_names
                    )
                    if model_found:
                        return True
                    else:
                        print(f"Warning: Model '{self.model}' not found in Ollama. Available models: {model_names}")
                        return False
                return False
        except Exception as e:
            print(f"Ollama health check failed: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text])
        # Always return a valid vector, never an empty list
        if embeddings and len(embeddings) > 0 and len(embeddings[0]) == self.dimension:
            return embeddings[0]
        else:
            print(f"Warning: Invalid embedding generated, returning zero vector. Embeddings: {embeddings}")
            return [0.0] * self.dimension
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        try:
            # Quick check if Ollama is ready (non-blocking, just for logging)
            ollama_ready = await self.check_ollama_ready()
            if not ollama_ready:
                print(f"Warning: Ollama may not be ready or model '{self.model}' not available. Proceeding with embedding generation...")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                for text in texts:
                    try:
                        response = await client.post(
                            f"{self.api_url}/api/embeddings",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": self.model,
                                "prompt": text
                            }
                        )
                        
                        if response.status_code != 200:
                            print(f"Embedding API error (status {response.status_code}): {response.text}")
                            print(f"  Request text length: {len(text)} chars")
                            embeddings.append([0.0] * self.dimension)
                        else:
                            result = response.json()
                            embedding = result.get("embedding", [])
                            
                            # Validate embedding dimension
                            if not embedding or len(embedding) != self.dimension:
                                print(f"Warning: Invalid embedding dimension. Expected {self.dimension}, got {len(embedding) if embedding else 0}")
                                print(f"  Response keys: {result.keys()}")
                                embeddings.append([0.0] * self.dimension)
                            else:
                                embeddings.append(embedding)
                    except httpx.TimeoutException:
                        print(f"Timeout generating embedding for text (length: {len(text)} chars)")
                        embeddings.append([0.0] * self.dimension)
                    except httpx.RequestError as e:
                        print(f"Request error generating embedding: {e}")
                        embeddings.append([0.0] * self.dimension)
                    except Exception as e:
                        print(f"Unexpected error generating embedding: {e}")
                        embeddings.append([0.0] * self.dimension)
                
                return embeddings
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * self.dimension for _ in texts]

