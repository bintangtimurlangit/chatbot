from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .models import User, Conversation


class ContextManager:
    """Manages conversation context for users"""
    
    def __init__(self, db: Session, max_history: int = 10):
        self.db = db
        self.max_history = max_history
    
    def get_or_create_user(self, user_id: str, platform: str) -> User:
        """Get existing user or create new one"""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(user_id=user_id, platform=platform)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user
    
    def add_message(self, user_id: str, platform: str, role: str, message: str) -> Conversation:
        """Add a message to conversation history"""
        # Ensure user exists
        self.get_or_create_user(user_id, platform)
        
        conversation = Conversation(
            user_id=user_id,
            platform=platform,
            role=role,
            message=message
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    def get_conversation_history(
        self, 
        user_id: str, 
        platform: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get conversation history for a user"""
        if limit is None:
            limit = self.max_history
        
        conversations = (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id, Conversation.platform == platform)
            .order_by(Conversation.timestamp.desc())
            .limit(limit)
            .all()
        )
        
        # Reverse to get chronological order
        conversations = list(reversed(conversations))
        
        return [
            {"role": conv.role, "content": conv.message}
            for conv in conversations
        ]
    
    def get_recent_context(
        self, 
        user_id: str, 
        platform: str, 
        hours: int = 24
    ) -> List[Dict[str, str]]:
        """Get conversation history within specific time window"""
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        conversations = (
            self.db.query(Conversation)
            .filter(
                Conversation.user_id == user_id,
                Conversation.platform == platform,
                Conversation.timestamp >= time_threshold
            )
            .order_by(Conversation.timestamp.asc())
            .all()
        )
        
        return [
            {"role": conv.role, "content": conv.message}
            for conv in conversations
        ]
    
    def clear_user_history(self, user_id: str, platform: str) -> int:
        """Clear conversation history for a user"""
        deleted = (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id, Conversation.platform == platform)
            .delete()
        )
        self.db.commit()
        return deleted
    
    def get_user_stats(self, user_id: str, platform: str) -> Dict:
        """Get statistics for a user"""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return None
        
        message_count = (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id, Conversation.platform == platform)
            .count()
        )
        
        last_message = (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id, Conversation.platform == platform)
            .order_by(Conversation.timestamp.desc())
            .first()
        )
        
        return {
            "user_id": user_id,
            "platform": platform,
            "message_count": message_count,
            "first_seen": user.created_at.isoformat(),
            "last_seen": last_message.timestamp.isoformat() if last_message else None
        }

