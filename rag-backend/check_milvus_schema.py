#!/usr/bin/env python3
"""检查Milvus collection的schema"""

import os
import sys
from dotenv import load_dotenv
from pymilvus import MilvusClient, Collection, connections

load_dotenv()

# 连接Milvus
milvus_uri = os.getenv('MILVUS_URI', 'http://localhost:19530')
temp_client = MilvusClient(uri=milvus_uri)

# 列出所有collections
collections = temp_client.list_collections()
print("\n所有collections:")
for c in collections:
    print(f"  - {c}")

# 找chunks collections
chunks_collections = [c for c in collections if c.endswith('_chunks')]
print(f"\n找到 {len(chunks_collections)} 个chunks collection")

if chunks_collections:
    # 检查所有chunks collections找一个有数据的
    for collection_name in sorted(chunks_collections, reverse=True):
        print(f"\n{'='*60}")
        print(f"检查collection: {collection_name}")
        print(f"{'='*60}")
        
        # 获取schema
        try:
            # 使用PyMilvus连接
            connections.connect(
                alias="default",
                uri=milvus_uri
            )
            
            collection = Collection(collection_name)
            schema = collection.schema
            
            print(f"\n字段列表:")
            for field in schema.fields:
                print(f"  - {field.name} ({field.dtype})")
            
            # 查询一条记录看实际数据
            print(f"\n查询一条记录查看字段内容:")
            results = temp_client.query(
                collection_name=collection_name,
                filter="",
                output_fields=['*'],
                limit=1
            )
            
            if results:
                print(f"\n✅ 第一条记录的所有字段:")
                record = results[0]
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}... (长度: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
                break  # 找到一个有数据的就退出
            else:
                print("❌ collection为空")
                
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
else:
    print("\n未找到任何chunks collection")
