from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
import httpx
import os
import hashlib
import redis
from datetime import datetime
from contextlib import asynccontextmanager

from .database import get_db, init_db
from .context_manager import ContextManager
from .rag_service import RAGService
from .models import User, Conversation


# Initialize RAG service
rag_service = RAGService()

# Initialize Redis for deduplication
redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
try:
    redis_client = redis.from_url(redis_url, decode_responses=True)
    redis_client.ping()
    print("Redis connected for deduplication")
except Exception as e:
    print(f"Redis connection warning: {e}. Deduplication disabled.")
    redis_client = None


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
    use_knowledge_base: bool = True  # Always enforced in strict mode


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
    print(f"Chat request received: user_id={request.user_id}, platform={request.platform}, message_length={len(request.message)}")
    try:
        context_manager = ContextManager(db)
        
        # Get conversation history from last 24 hours
        history = context_manager.get_recent_context(
            user_id=request.user_id,
            platform=request.platform,
            hours=24  # Last 24 hours of context
        )
        
        # Build system prompt - STRICT mode: Only use knowledge base
        system_prompt = """Kamu adalah asisten chatbot resmi untuk layanan KSJPS dan JPD Kota Yogyakarta.

ATURAN PENTING:
- HANYA jawab pertanyaan tentang KSJPS, JPD, dan layanan Dinsosnakertrans Kota Yogyakarta
- WAJIB gunakan informasi dari knowledge base yang diberikan
- Jika pertanyaan di luar topik KSJPS/JPD, katakan: "Maaf, saya hanya dapat membantu pertanyaan seputar KSJPS dan JPD. Silakan hubungi Dinsosnakertrans Kota Yogyakarta untuk informasi lainnya."
- Jangan jawab pertanyaan umum, chitchat, atau topik lain
- Selalu sopan dan profesional dalam Bahasa Indonesia"""
        
        # ALWAYS search knowledge base (enforced for strict mode)
        knowledge_context = ""
        sources_count = 0
        try:
            # Search for relevant knowledge (only once!)
            search_results = await rag_service.search_knowledge_base(request.message, limit=3)
            sources_count = len(search_results)
            
            # If NO relevant knowledge found, reject the question
            if sources_count == 0:
                return ChatResponse(
                    response="Maaf, saya hanya dapat membantu pertanyaan seputar KSJPS (Keluarga Sasaran Jaminan Perlindungan Sosial) dan JPD (Jaminan Pendidikan Daerah) di Kota Yogyakarta. Untuk informasi lainnya, silakan hubungi Dinsosnakertrans Kota Yogyakarta di nomor telepon atau datang langsung ke kantor.",
                    timestamp=datetime.now().isoformat(),
                    sources_used=0
                )
            
            # Build context from search results (reuse results, don't search again!)
            if search_results:
                context_parts = []
                for idx, result in enumerate(search_results, 1):
                    text = result["text"]
                    score = result["score"]
                    context_parts.append(f"[Sumber {idx}] (Relevansi: {score:.2f})\n{text}")
                
                knowledge_context = "\n\n".join(context_parts)
                system_prompt += f"\n\nInformasi dari knowledge base:\n\n{knowledge_context}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"RAG search error: {e}")
            print(f"RAG search error details:\n{error_details}")
            # If RAG fails, also reject to be safe
            return ChatResponse(
                response="Maaf, sistem sedang mengalami kendala. Silakan coba lagi atau hubungi Dinsosnakertrans Kota Yogyakarta.",
                timestamp=datetime.now().isoformat(),
                sources_used=0
            )
        
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
    print(f"Webhook received: user_id={webhook_data.user_id}, platform={webhook_data.platform}, message='{webhook_data.message[:50] if len(webhook_data.message) > 50 else webhook_data.message}...'")
    try:
        # Deduplication: Check if we've processed this exact message recently (within 30 seconds)
        cache_key = None
        if redis_client:
            message_fingerprint = hashlib.md5(
                f"{webhook_data.user_id}:{webhook_data.platform}:{webhook_data.message}".encode()
            ).hexdigest()
            cache_key = f"msg:{message_fingerprint}"
            
            # Check if message was recently processed (using SETNX to handle race conditions)
            cached_response = redis_client.get(cache_key)
            if cached_response and cached_response != "processing":
                print(f"Duplicate message detected for {webhook_data.user_id}, returning cached response")
                import json
                try:
                    return json.loads(cached_response)
                except:
                    pass  # If JSON parsing fails, continue processing
            
            # Try to acquire lock (set if not exists) - prevents concurrent processing
            lock_acquired = redis_client.set(cache_key, "processing", ex=30, nx=True)
            if not lock_acquired:
                # Another request is processing, wait a bit and check again
                import asyncio
                await asyncio.sleep(0.5)
                cached_response = redis_client.get(cache_key)
                if cached_response and cached_response != "processing":
                    print(f"Duplicate message detected (after wait) for {webhook_data.user_id}, returning cached response")
                    import json
                    try:
                        return json.loads(cached_response)
                    except:
                        pass  # If JSON parsing fails, continue processing
        
        # Create chat request (knowledge base always enforced)
        chat_request = ChatRequest(
            user_id=webhook_data.user_id,
            platform=webhook_data.platform,
            message=webhook_data.message,
            use_knowledge_base=True  # Always true for strict mode
        )
        
        # Process the message
        response = await chat(chat_request, db)
        
        result = {
            "status": "success",
            "user_id": webhook_data.user_id,
            "platform": webhook_data.platform,
            "response": response.response,
            "sources_used": response.sources_used,
            "timestamp": response.timestamp
        }
        
        # Cache the response for deduplication (30 seconds)
        if redis_client and cache_key:
            import json
            redis_client.setex(cache_key, 30, json.dumps(result))
        
        return result
        
    except Exception as e:
        # Clear processing flag on error so it can be retried
        if redis_client and cache_key:
            redis_client.delete(cache_key)
        print(f"Webhook error: {e}")
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


# Admin/Maintenance endpoints
@app.post("/admin/cleanup-old-conversations")
async def cleanup_old_conversations(days: int = 90, db: Session = Depends(get_db)):
    """
    Delete conversations older than specified days (default: 90 days)
    This helps with GDPR compliance and database management
    """
    try:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Count first
        count = db.query(Conversation).filter(
            Conversation.timestamp < cutoff_date
        ).count()
        
        # Delete
        deleted = db.query(Conversation).filter(
            Conversation.timestamp < cutoff_date
        ).delete()
        
        db.commit()
        
        return {
            "status": "success",
            "deleted_count": deleted,
            "cutoff_date": cutoff_date.isoformat(),
            "retention_days": days
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")


@app.post("/admin/clear-all-contexts")
async def clear_all_contexts(db: Session = Depends(get_db)):
    """
    Delete ALL conversation history for all users
    Keeps user records intact, only clears conversation messages
    Useful for testing or complete system reset
    """
    try:
        # Count total conversations before deletion
        total_conversations = db.query(Conversation).count()
        total_users = db.query(User).count()
        
        # Delete all conversations (CASCADE will handle relationships)
        deleted = db.query(Conversation).delete()
        
        db.commit()
        
        return {
            "status": "success",
            "message": "All conversation contexts cleared",
            "conversations_deleted": deleted,
            "users_retained": total_users,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Clear contexts error: {str(e)}")
