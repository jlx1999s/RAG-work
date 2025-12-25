"""测试血脂文档检索"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.rag.storage.milvus_storage import MilvusStorage
from backend.config.embedding import get_embedding_model


async def main():
    """测试向量检索"""
    collection_id = "kb12_1766670767961"
    query = "血脂是什么"
    
    print(f"测试向量检索: '{query}'")
    print(f"Collection: {collection_id}\n")
    
    # 初始化 Milvus
    milvus = MilvusStorage(
        embedding_function=get_embedding_model(),
        collection_name=collection_id
    )
    
    # 执行检索
    results = milvus.hybrid_search(query, k=5)
    
    print(f"检索结果（共 {len(results)} 条）：\n")
    
    for i, doc in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  - 文档名: {doc.metadata.get('document_name', 'unknown')}")
        print(f"  - 内容: {doc.page_content[:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
