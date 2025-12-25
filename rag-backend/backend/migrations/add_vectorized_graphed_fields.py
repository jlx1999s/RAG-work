"""添加 is_vectorized 和 is_graphed 字段到 knowledge_documents 表"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.config.database import DatabaseFactory
from sqlalchemy import text


def migrate():
    """执行数据库迁移"""
    db_factory = DatabaseFactory()
    engine = db_factory.get_engine()
    
    with engine.connect() as conn:
        # 检查字段是否已存在
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'knowledge_documents' 
            AND column_name IN ('is_vectorized', 'is_graphed')
        """))
        
        existing_columns = {row[0] for row in result}
        
        # 添加 is_vectorized 字段
        if 'is_vectorized' not in existing_columns:
            print("添加 is_vectorized 字段...")
            conn.execute(text("""
                ALTER TABLE knowledge_documents 
                ADD COLUMN is_vectorized BOOLEAN NOT NULL DEFAULT FALSE 
                COMMENT '是否已向量化'
            """))
            conn.commit()
            print("✅ is_vectorized 字段添加成功")
        else:
            print("⏭️  is_vectorized 字段已存在，跳过")
        
        # 添加 is_graphed 字段
        if 'is_graphed' not in existing_columns:
            print("添加 is_graphed 字段...")
            conn.execute(text("""
                ALTER TABLE knowledge_documents 
                ADD COLUMN is_graphed BOOLEAN NOT NULL DEFAULT FALSE 
                COMMENT '是否已图谱化'
            """))
            conn.commit()
            print("✅ is_graphed 字段添加成功")
        else:
            print("⏭️  is_graphed 字段已存在，跳过")
        
        # 同步旧数据：is_processed = True 的文档，设置两个字段都为 True
        print("\n同步旧数据...")
        result = conn.execute(text("""
            UPDATE knowledge_documents 
            SET is_vectorized = TRUE, is_graphed = TRUE 
            WHERE is_processed = TRUE
        """))
        conn.commit()
        print(f"✅ 已同步 {result.rowcount} 条旧数据")
    
    print("\n🎉 迁移完成！")


if __name__ == "__main__":
    migrate()
