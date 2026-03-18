from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Boolean, UniqueConstraint, Index, func
from backend.config.database import DatabaseFactory
from backend.utils.timezone import to_china_time

Base = DatabaseFactory.get_base()


class UserMemoryProfile(Base):
    __tablename__ = "user_memory_profiles"
    __table_args__ = (
        UniqueConstraint("user_id", "collection_id", "memory_key", name="uq_user_memory_profile_key"),
        Index("idx_user_memory_profile_user_collection", "user_id", "collection_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False)
    collection_id = Column(String(128), nullable=False, default="default")
    memory_key = Column(String(128), nullable=False)
    memory_value = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=0.6)
    source = Column(String(64), nullable=False, default="chat")
    is_deleted = Column(Boolean, nullable=False, default=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "collection_id": self.collection_id,
            "memory_key": self.memory_key,
            "memory_value": self.memory_value,
            "confidence": self.confidence,
            "source": self.source,
            "is_deleted": self.is_deleted,
            "expires_at": to_china_time(self.expires_at).isoformat() if self.expires_at else None,
            "created_at": to_china_time(self.created_at).isoformat() if self.created_at else None,
            "updated_at": to_china_time(self.updated_at).isoformat() if self.updated_at else None
        }


class UserMemoryEvent(Base):
    __tablename__ = "user_memory_events"
    __table_args__ = (
        Index("idx_user_memory_event_user_collection_created", "user_id", "collection_id", "created_at"),
        Index("idx_user_memory_event_conversation", "conversation_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False)
    conversation_id = Column(String(64), nullable=False)
    collection_id = Column(String(128), nullable=False, default="default")
    role = Column(String(20), nullable=False, default="user")
    content = Column(Text, nullable=False)
    importance = Column(Float, nullable=False, default=0.5)
    source = Column(String(64), nullable=False, default="chat")
    extra_data = Column(JSON, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "collection_id": self.collection_id,
            "role": self.role,
            "content": self.content,
            "importance": self.importance,
            "source": self.source,
            "extra_data": self.extra_data,
            "expires_at": to_china_time(self.expires_at).isoformat() if self.expires_at else None,
            "created_at": to_china_time(self.created_at).isoformat() if self.created_at else None
        }
