import os
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class GraphRAGStorage:
    """Microsoft GraphRAG 存储和检索类
    
    核心特性：
    - 分层社区检测
    - 多级摘要生成
    - 高性能并行处理
    - 支持超大规模知识库
    """

    def __init__(self, workspace: str = "default"):
        """初始化 GraphRAG 存储
        
        Args:
            workspace: 工作空间名称，用于数据隔离
        """
        self.workspace = workspace
        load_dotenv()
        
        # GraphRAG 工作目录
        self.working_dir = Path(__file__).parent / "graphrag_storage" / workspace
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 输出目录
        self.output_dir = self.working_dir / "output"
        self.cache_dir = self.working_dir / "cache"
        
        self.initialized = False

    async def initialize(self) -> None:
        """初始化 GraphRAG 索引"""
        if self.initialized:
            return
        
        # 创建 GraphRAG 配置文件
        await self._create_config()
        self.initialized = True
        logger.info(f"GraphRAG 初始化完成: {self.workspace}")

    async def _create_config(self) -> None:
        """创建 GraphRAG 配置文件 (settings.yaml)"""
        config_content = f"""
encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: {os.getenv('LLM_DASHSCOPE_API_KEY')}
  type: openai_chat
  model: {os.getenv('LLM_DASHSCOPE_CHAT_MODEL', 'qwen-plus')}
  api_base: {os.getenv('LLM_DASHSCOPE_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}
  max_tokens: 4000
  temperature: 0
  top_p: 1

parallelization:
  stagger: 0.3
  num_threads: 4  # 并行线程数

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: {os.getenv('VECTOR_DASHSCOPE_API_KEY')}
    type: openai_embedding
    model: {os.getenv('VECTOR_DASHSCOPE_EMBEDDING_MODEL', 'text-embedding-v4')}
    api_base: {os.getenv('VECTOR_DASHSCOPE_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}

chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

input:
  type: file
  file_type: text
  base_dir: "{self.working_dir / 'input'}"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file
  base_dir: "{self.cache_dir}"

storage:
  type: file
  base_dir: "{self.output_dir}"

reporting:
  type: file
  base_dir: "{self.working_dir / 'reports'}"

entity_extraction:
  entity_types: [organization,person,geo,event,concept,disease,treatment,symptom]
  max_gleanings: 1  # 减少噪声

community_reports:
  max_length: 2000
  max_input_length: 8000

claim_extraction:
  enabled: false  # 禁用声明提取以提升性能

summarize_descriptions:
  max_length: 500
"""
        
        config_path = self.working_dir / "settings.yaml"
        config_path.write_text(config_content, encoding='utf-8')
        logger.info(f"GraphRAG 配置文件已创建: {config_path}")

    async def insert_text(self, text: str, doc_id: str = None) -> None:
        """插入文本到 GraphRAG
        
        Args:
            text: 文本内容
            doc_id: 文档ID（可选）
        """
        if not self.initialized:
            await self.initialize()
        
        # 将文本写入输入目录
        input_dir = self.working_dir / "input"
        input_dir.mkdir(exist_ok=True)
        
        doc_id = doc_id or f"doc_{len(list(input_dir.glob('*.txt')))}"
        input_file = input_dir / f"{doc_id}.txt"
        input_file.write_text(text, encoding='utf-8')
        
        logger.info(f"文本已写入 GraphRAG 输入目录: {input_file}")

    async def insert_texts(self, texts: List[str]) -> None:
        """批量插入文本
        
        Args:
            texts: 文本列表
        """
        for i, text in enumerate(texts):
            await self.insert_text(text, doc_id=f"chunk_{i}")

    async def build_index(self) -> Dict[str, Any]:
        """构建 GraphRAG 索引（异步后台任务）
        
        Returns:
            构建结果统计
        """
        logger.info(f"开始构建 GraphRAG 索引: {self.workspace}")
        
        try:
            # 运行 GraphRAG 索引构建命令
            import subprocess
            
            cmd = [
                "graphrag",
                "index",
                "--root", str(self.working_dir),
                "--verbose"
            ]
            
            # 异步执行（避免阻塞）
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"GraphRAG 索引构建成功: {self.workspace}")
                return {"status": "success", "workspace": self.workspace}
            else:
                logger.error(f"GraphRAG 索引构建失败: {stderr.decode()}")
                return {"status": "error", "message": stderr.decode()}
                
        except Exception as e:
            logger.error(f"构建 GraphRAG 索引异常: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def query(
        self,
        query: str,
        method: str = "global",  # global, local, or drift
        **kwargs
    ) -> str:
        """执行 GraphRAG 查询
        
        Args:
            query: 查询文本
            method: 查询方法
                - global: 全局社区摘要检索
                - local: 本地实体检索
                - drift: 混合检索
            **kwargs: 其他参数
            
        Returns:
            查询结果
        """
        logger.info(f"GraphRAG 查询: {query} (method={method})")
        
        try:
            import subprocess
            
            cmd = [
                "graphrag",
                "query",
                "--root", str(self.working_dir),
                "--method", method,
                "--query", query
            ]
            
            # 执行查询
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"GraphRAG 查询失败: {result.stderr}")
                return f"查询失败: {result.stderr}"
                
        except Exception as e:
            logger.error(f"GraphRAG 查询异常: {str(e)}")
            return f"查询异常: {str(e)}"

    async def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 读取 GraphRAG 输出的统计文件
            stats_file = self.output_dir / "stats.json"
            
            if stats_file.exists():
                import json
                stats = json.loads(stats_file.read_text())
                return stats
            else:
                return {"status": "not_indexed", "workspace": self.workspace}
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}

    async def drop_workspace(self) -> None:
        """删除当前 workspace 的所有数据"""
        import shutil
        
        try:
            if self.working_dir.exists():
                shutil.rmtree(self.working_dir)
                logger.info(f"已删除 GraphRAG workspace: {self.workspace}")
        except Exception as e:
            logger.error(f"删除 workspace 失败: {str(e)}")
