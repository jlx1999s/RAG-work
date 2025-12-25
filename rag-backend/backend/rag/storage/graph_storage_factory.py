"""图谱存储工厂
自动根据环境变量选择 LightRAG 或 GraphRAG
"""
import os
from typing import Union
from backend.rag.storage.lightrag_storage import LightRAGStorage
from backend.rag.storage.graphrag_storage import GraphRAGStorage


def create_graph_storage(workspace: str = "default") -> Union[LightRAGStorage, GraphRAGStorage]:
    """创建图谱存储实例
    
    Args:
        workspace: 工作空间名称
        
    Returns:
        LightRAGStorage 或 GraphRAGStorage 实例
    """
    engine = os.getenv("GRAPH_ENGINE", "lightrag").lower()
    
    if engine == "graphrag":
        return GraphRAGStorage(workspace=workspace)
    else:
        return LightRAGStorage(workspace=workspace)
