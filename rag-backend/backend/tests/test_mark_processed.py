"""手动标记文档为已处理"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.service.knowledge_library import _mark_document_processed


async def main():
    """标记文档为已处理"""
    document_id = 27  # 血脂文档
    
    print(f"标记文档 {document_id} 为已处理...")
    await _mark_document_processed(document_id)
    print("完成！")


if __name__ == "__main__":
    asyncio.run(main())
