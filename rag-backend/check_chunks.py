#!/usr/bin/env python3
"""检查文档分块情况"""

import os
import sys
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()

# 连接Milvus
temp_client = MilvusClient(uri=os.getenv('MILVUS_URI', 'http://localhost:19530'))
collections = temp_client.list_collections()

print("\n所有collections:")
for c in collections:
    print(f"  - {c}")

# 只查找chunks类型的collection
chunks_collections = [c for c in collections if c.endswith('_chunks')]
print(f"\n找到 {len(chunks_collections)} 个chunks collection")

if not chunks_collections:
    print("\n错误：未找到任何chunks collection")
    sys.exit(1)

# 显示所有chunks collection并查询每个collection中是否有"发烧诊疗指南.md"
print("\n查找'发烧诊疗指南.md'...")
found = False

for collection_name in chunks_collections:
    try:
        results = temp_client.query(
            collection_name=collection_name,
            filter='document_name == "发烧诊疗指南.md"',
            output_fields=['document_name', 'chunk_index', 'chunk_size'],
            limit=100
        )
        
        if len(results) > 0:
            found = True
            print(f"\n✅ 在 {collection_name} 中找到文档！")
            print(f"\n文档: 发烧诊疗指南.md")
            print(f"总chunk数: {len(results)}")
            print('\nchunk详情:')
            for r in results:
                print(f"  chunk_index={r.get('chunk_index')}, chunk_size={r.get('chunk_size')}")
            break
    except Exception as e:
        # 跳过查询失败的collection
        continue

if not found:
    print("\n❌ 未在任何collection中找到'发烧诊疗指南.md'")
    print("\n请检查：")
    print("1. 文档是否已上传")
    print("2. 文档名称是否正确")
    print("3. 是否已完成向量化")
