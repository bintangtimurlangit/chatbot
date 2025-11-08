from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
import httpx
import os
from datetime import datetime
from contextlib import asynccontextmanager

from .database import get_db, init_db
from .context_manager import ContextManager
from .rag_service import RAGService
from .models import User, Conversation


# Initialize RAG service
rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown"""
    # Startup
    print("Starting chatbot backend...")
    init_db()
    print("Database initialized")
    
    try:
        rag_service.initialize_collection()
        print("Qdrant collection initialized")
    except Exception as e:
        print(f"Qdrant initialization warning: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down chatbot backend...")


app = FastAPI(
    title="Chatbot Backend API",
    description="Backend for WhatsApp and Instagram chatbot with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://agentrouter.org/v1")


# Pydantic models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str
    platform: str  # 'whatsapp' or 'instagram'
    message: str
    use_knowledge_base: bool = True


class ChatResponse(BaseModel):
    response: str
    timestamp: str
    sources_used: int = 0


class WebhookMessage(BaseModel):
    """Message from n8n webhook (WhatsApp/Instagram)"""
    user_id: str
    platform: str
    message: str
    metadata: Optional[Dict] = None


class KnowledgeDocument(BaseModel):
    text: str
    metadata: Optional[Dict] = None


class KnowledgeBatchRequest(BaseModel):
    documents: List[KnowledgeDocument]


# Root and health endpoints
@app.get("/")
async def root():
    return {
        "service": "Chatbot Backend API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
@app.head("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Check database
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check Qdrant
    try:
        qdrant_info = rag_service.get_knowledge_base_stats()
        qdrant_status = "healthy" if "error" not in qdrant_info else "unhealthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "services": {
            "database": db_status,
            "qdrant": qdrant_status,
            "deepseek": "configured" if DEEPSEEK_API_KEY else "not_configured",
        },
        "timestamp": datetime.now().isoformat()
    }


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Main chat endpoint with RAG support"""
    try:
        context_manager = ContextManager(db)
        
        # Get conversation history
        history = context_manager.get_conversation_history(
            user_id=request.user_id,
            platform=request.platform,
            limit=8  # Last 8 messages for context
        )
        
        # Build system prompt
        system_prompt = """Kamu adalah asisten chatbot yang ramah dan membantu. 
Jawab semua pertanyaan dalam Bahasa Indonesia dengan sopan dan jelas.
Jika ada informasi dari knowledge base, gunakan informasi tersebut untuk menjawab dengan akurat.
Jika tidak ada informasi yang relevan, jawab berdasarkan pengetahuanmu sendiri."""
        
        # Search knowledge base if enabled
        knowledge_context = ""
        sources_count = 0
        if request.use_knowledge_base:
            try:
                knowledge_context = await rag_service.build_context(
                    query=request.message,
                    max_results=3
                )
                if knowledge_context:
                    sources_count = len(await rag_service.search_knowledge_base(request.message, limit=3))
                    system_prompt += f"\n\n{knowledge_context}"
            except Exception as e:
                print(f"RAG search error: {e}")
        
        # Build messages for AI
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": request.message})
        
        # Call DeepSeek API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{DEEPSEEK_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Deepseek API error: {response.text}"
                )
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
        
        # Store conversation in database
        context_manager.add_message(request.user_id, request.platform, "user", request.message)
        context_manager.add_message(request.user_id, request.platform, "assistant", ai_response)
        
        return ChatResponse(
            response=ai_response,
            timestamp=datetime.now().isoformat(),
            sources_used=sources_count
        )
        
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Webhook endpoint for n8n (WhatsApp/Instagram)
@app.post("/webhook/message")
async def webhook_message(webhook_data: WebhookMessage, db: Session = Depends(get_db)):
    """
    Webhook endpoint for n8n to send messages from WhatsApp/Instagram
    n8n will call this endpoint when a message is received
    """
    try:
        # Create chat request
        chat_request = ChatRequest(
            user_id=webhook_data.user_id,
            platform=webhook_data.platform,
            message=webhook_data.message,
            use_knowledge_base=True
        )
        
        # Process the message
        response = await chat(chat_request, db)
        
        return {
            "status": "success",
            "user_id": webhook_data.user_id,
            "platform": webhook_data.platform,
            "response": response.response,
            "sources_used": response.sources_used,
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook error: {str(e)}")


# Knowledge base management endpoints
@app.post("/knowledge/add")
async def add_knowledge(request: KnowledgeBatchRequest):
    """Add documents to knowledge base (called by n8n from Google Sheets)"""
    try:
        documents = [
            {
                "text": doc.text,
                "metadata": doc.metadata or {}
            }
            for doc in request.documents
        ]
        
        count = await rag_service.add_to_knowledge_base(documents)
        
        return {
            "status": "success",
            "documents_added": count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding knowledge: {str(e)}")


@app.get("/knowledge/search")
async def search_knowledge(query: str, limit: int = 5):
    """Search knowledge base"""
    try:
        results = await rag_service.search_knowledge_base(query, limit=limit)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/knowledge/stats")
async def knowledge_stats():
    """Get knowledge base statistics"""
    try:
        stats = rag_service.get_knowledge_base_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Context management endpoints
@app.get("/context/{user_id}/{platform}")
async def get_context(user_id: str, platform: str, db: Session = Depends(get_db)):
    """Get conversation history for a user"""
    try:
        context_manager = ContextManager(db)
        history = context_manager.get_conversation_history(user_id, platform, limit=20)
        return {
            "user_id": user_id,
            "platform": platform,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.delete("/context/{user_id}/{platform}")
async def clear_context(user_id: str, platform: str, db: Session = Depends(get_db)):
    """Clear conversation history for a user"""
    try:
        context_manager = ContextManager(db)
        deleted = context_manager.clear_user_history(user_id, platform)
        return {
            "status": "success",
            "user_id": user_id,
            "platform": platform,
            "messages_deleted": deleted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/context/{user_id}/{platform}/stats")
async def get_user_stats(user_id: str, platform: str, db: Session = Depends(get_db)):
    """Get user statistics"""
    try:
        context_manager = ContextManager(db)
        stats = context_manager.get_user_stats(user_id, platform)
        if not stats:
            raise HTTPException(status_code=404, detail="User not found")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Admin endpoints
@app.get("/users")
async def list_users(db: Session = Depends(get_db)):
    """List all users"""
    try:
        users = db.query(User).all()
        return {
            "users": [
                {
                    "user_id": user.user_id,
                    "platform": user.platform,
                    "created_at": user.created_at.isoformat()
                }
                for user in users
            ],
            "count": len(users)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
