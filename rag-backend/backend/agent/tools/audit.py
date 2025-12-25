"""工具调用审计日志模块

企业级最佳实践：
1. 持久化所有工具调用记录
2. 便于审计、合规和问题排查
3. 支持性能分析和统计
"""

from typing import Optional, Dict, Any
from datetime import datetime
from backend.config.log import get_logger
import json

logger = get_logger(__name__)


class ToolCallAuditLogger:
    """工具调用审计日志器
    
    记录所有工具调用的详细信息，包括：
    - 调用时间
    - 用户信息
    - 工具名称和参数
    - 执行结果
    - 执行时长
    - 错误信息（如果有）
    """
    
    def __init__(self, db_session=None):
        """
        Args:
            db_session: 数据库会话（可选）
                      如果提供，将记录保存到数据库
                      否则只记录到日志文件
        """
        self.db_session = db_session
        self.logger = logger
    
    def log_tool_call(
        self,
        conversation_id: str,
        user_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录工具调用
        
        Args:
            conversation_id: 会话ID
            user_id: 用户ID
            tool_name: 工具名称
            tool_args: 工具参数
            result: 执行结果
            error: 错误信息
            execution_time_ms: 执行时长（毫秒）
            metadata: 额外的元数据
        """
        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            "error": error,
            "execution_time_ms": execution_time_ms,
            "status": "success" if error is None else "error",
            "metadata": metadata or {}
        }
        
        # 记录到日志文件
        self.logger.info(
            f"[TOOL_CALL_AUDIT] {json.dumps(audit_record, ensure_ascii=False)}"
        )
        
        # 如果有数据库会话，保存到数据库
        if self.db_session:
            try:
                self._save_to_database(audit_record)
            except Exception as e:
                self.logger.error(f"保存审计日志到数据库失败: {e}")
    
    def _save_to_database(self, audit_record: Dict[str, Any]) -> None:
        """保存审计记录到数据库
        
        Args:
            audit_record: 审计记录
        """
        # TODO: 实现数据库保存逻辑
        # 示例SQL:
        # INSERT INTO tool_call_audit (
        #     timestamp, conversation_id, user_id, tool_name,
        #     tool_args, result, error, execution_time_ms, status
        # ) VALUES (...)
        
        pass
    
    def get_audit_records(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list:
        """查询审计记录
        
        Args:
            conversation_id: 会话ID过滤
            user_id: 用户ID过滤
            tool_name: 工具名称过滤
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            limit: 返回记录数量限制
            
        Returns:
            审计记录列表
        """
        # TODO: 实现数据库查询逻辑
        # 示例SQL:
        # SELECT * FROM tool_call_audit
        # WHERE conversation_id = ? AND user_id = ?
        # ORDER BY timestamp DESC LIMIT ?
        
        return []


# 全局审计日志器实例
_audit_logger = ToolCallAuditLogger()


def get_audit_logger() -> ToolCallAuditLogger:
    """获取全局审计日志器实例"""
    return _audit_logger
