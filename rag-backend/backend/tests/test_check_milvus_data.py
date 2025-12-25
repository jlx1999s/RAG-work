"""检查Milvus数据库中"医疗"知识库的数据"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
backend_dir = Path(__file__).parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from backend.config.database import DatabaseFactory
from backend.model.knowledge_library import KnowledgeLibrary, KnowledgeDocument


async def check_milvus_data():
    """检查医疗知识库的数据状态"""
    
    print("=" * 60)
    print("检查【医疗】知识库的数据状态")
    print("=" * 60)
    
    # 1. 查询MySQL中的知识库和文档信息
    db_factory = DatabaseFactory()
    session = db_factory.create_session()
    
    try:
        # 查找"医疗"知识库
        library = session.query(KnowledgeLibrary).filter(
            KnowledgeLibrary.title == "medic",
            KnowledgeLibrary.is_active == True
        ).first()
        
        if not library:
            print("❌ 未找到【医疗】知识库，可能已被删除或名称不匹配")
            return
        
        print(f"\n✅ 找到知识库:")
        print(f"  - ID: {library.id}")
        print(f"  - 名称: {library.title}")
        print(f"  - Collection ID: {library.collection_id}")
        print(f"  - 是否激活: {library.is_active}")
        
        collection_id = library.collection_id
        
        # 查询该知识库下的所有文档
        documents = session.query(KnowledgeDocument).filter(
            KnowledgeDocument.library_id == library.id
        ).all()
        
        print(f"\n📄 MySQL中的文档列表 (共 {len(documents)} 个):")
        for doc in documents:
            print(f"  [{doc.id}] {doc.name}")
            print(f"      - 文件路径: {doc.file_path}")
            print(f"      - 已处理: {doc.is_processed}")
            print(f"      - 创建时间: {doc.created_at}")
        
        if not documents:
            print("  ⚠️  MySQL中没有文档记录")
            return
        
    finally:
        session.close()
    
    # 2. 检查Milvus中的数据
    print(f"\n{'='*60}")
    print(f"检查 Milvus Collection: {collection_id}")
    print(f"{'='*60}")
    
    try:
        from pymilvus import connections, utility, Collection
        
        # 连接到Milvus
        milvus_uri = os.getenv('MILVUS_URI', 'http://localhost:19530')
        milvus_db = os.getenv('MILVUS_DB_NAME', 'rag')
        
        print(f"\n🔗 连接到 Milvus: {milvus_uri}, 数据库: {milvus_db}")
        
        connections.connect(
            alias="default",
            uri=milvus_uri,
            db_name=milvus_db
        )
        
        # 检查collection是否存在
        if not utility.has_collection(collection_id):
            print(f"\n❌ Milvus中不存在 collection: {collection_id}")
            print("   原因可能是:")
            print("   1. 文档从未成功上传到Milvus")
            print("   2. Collection被删除了")
            print("   3. Collection ID不匹配")
            connections.disconnect("default")
            return
        
        print(f"✅ Collection 存在: {collection_id}")
        
        # 获取collection并查询数据
        collection = Collection(collection_id)
        collection.load()
        
        # 获取总数
        total_count = collection.num_entities
        print(f"\n📊 Collection 统计:")
        print(f"  - 总向量数: {total_count}")
        
        if total_count == 0:
            print("\n❌ Collection中没有任何数据!")
            print("   这意味着文档处理过程中出现了问题，向量没有成功写入")
            connections.disconnect("default")
            return
        
        # 查询每个文档的向量数量
        print(f"\n📑 按文档名称统计向量数量:")
        for doc in documents:
            doc_name = doc.name
            # 使用表达式过滤查询
            expr = f'document_name == "{doc_name}"'
            result = collection.query(
                expr=expr,
                output_fields=["document_name", "chunk_index"],
                limit=1000  # 设置较大的限制以获取所有chunks
            )
            
            chunk_count = len(result)
            if chunk_count > 0:
                print(f"  ✅ [{doc_name}]: {chunk_count} 个 chunks")
            else:
                print(f"  ❌ [{doc_name}]: 0 个 chunks (向量未写入)")
        
        # 随机抽取几条记录看看内容
        print(f"\n🔍 随机抽取 3 条记录查看:")
        sample_results = collection.query(
            expr="chunk_index >= 0",
            output_fields=["document_name", "chunk_index", "chunk_size"],
            limit=3
        )
        
        for i, record in enumerate(sample_results, 1):
            print(f"\n  记录 {i}:")
            print(f"    - document_name: {record.get('document_name', 'N/A')}")
            print(f"    - chunk_index: {record.get('chunk_index', 'N/A')}")
            print(f"    - chunk_size: {record.get('chunk_size', 'N/A')}")
        
        connections.disconnect("default")
        
    except Exception as e:
        print(f"\n❌ 检查Milvus数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 3. 检查LightRAG/Neo4j图谱状态
    print(f"\n{'='*60}")
    print(f"检查 LightRAG Workspace (Neo4j): {collection_id}")
    print(f"{'='*60}")
    
    try:
        from backend.rag.storage.lightrag_storage import LightRAGStorage
        
        lightrag = LightRAGStorage(workspace=collection_id)
        await lightrag.initialize()
        
        # 获取图谱统计
        stats = await lightrag.get_graph_stats()
        print(f"\n📊 图谱统计:")
        print(f"  - Workspace: {stats.get('workspace', 'N/A')}")
        print(f"  - 实体数: {stats.get('entity_count', 'N/A')}")
        print(f"  - 关系数: {stats.get('relation_count', 'N/A')}")
        
        await lightrag.finalize()
        
    except Exception as e:
        print(f"\n❌ 检查LightRAG数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("检查完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(check_milvus_data())
