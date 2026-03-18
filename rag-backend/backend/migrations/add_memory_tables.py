import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir.parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from backend.config.database import DatabaseFactory


def migrate():
    engine = DatabaseFactory.get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_memory_profiles (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
                memory_key VARCHAR(128) NOT NULL,
                memory_value TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 0.6,
                source VARCHAR(64) NOT NULL DEFAULT 'chat',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                expires_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                CONSTRAINT uq_user_memory_profile_key UNIQUE (user_id, collection_id, memory_key)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_user_memory_profile_user_collection
            ON user_memory_profiles (user_id, collection_id)
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_memory_events (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                conversation_id VARCHAR(64) NOT NULL,
                collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
                role VARCHAR(20) NOT NULL DEFAULT 'user',
                content TEXT NOT NULL,
                importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                source VARCHAR(64) NOT NULL DEFAULT 'chat',
                extra_data JSON NULL,
                expires_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_user_memory_event_user_collection_created
            ON user_memory_events (user_id, collection_id, created_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_user_memory_event_conversation
            ON user_memory_events (conversation_id)
        """))
        conn.commit()
    print("memory tables migration done")


if __name__ == "__main__":
    migrate()
