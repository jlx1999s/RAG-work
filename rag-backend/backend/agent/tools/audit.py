"""工具调用审计日志模块

企业级最佳实践：
1. 持久化所有工具调用记录
2. 便于审计、合规和问题排查
3. 支持性能分析和统计
"""

from typing import Optional, Dict, Any
from datetime import datetime
from backend.config.log import get_logger
from backend.config.database import DatabaseFactory
import json
from sqlalchemy import Table, Column, Integer, String, DateTime, Float, Text, MetaData, select, and_

logger = get_logger(__name__)

_metadata = MetaData()
_tool_call_audit_table = Table(
    "tool_call_audit",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False, index=True),
    Column("conversation_id", String(64), nullable=True, index=True),
    Column("user_id", String(64), nullable=True, index=True),
    Column("tool_name", String(128), nullable=False, index=True),
    Column("tool_args", Text, nullable=True),
    Column("result", Text, nullable=True),
    Column("error", Text, nullable=True),
    Column("execution_time_ms", Float, nullable=True),
    Column("status", String(32), nullable=False, index=True),
    Column("metadata", Text, nullable=True),
)


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
        self._table_ready = False
    
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
        session = self.db_session or DatabaseFactory.create_session()
        should_close = self.db_session is None
        try:
            if not self._table_ready:
                _metadata.create_all(bind=session.get_bind(), tables=[_tool_call_audit_table], checkfirst=True)
                self._table_ready = True
            timestamp_value = audit_record.get("timestamp")
            if isinstance(timestamp_value, str):
                timestamp_value = datetime.fromisoformat(timestamp_value)
            if not isinstance(timestamp_value, datetime):
                timestamp_value = datetime.now()
            execution_time_value = audit_record.get("execution_time_ms")
            if isinstance(execution_time_value, (int, float, str)):
                execution_time_ms = float(execution_time_value)
            else:
                execution_time_ms = None
            insert_stmt = _tool_call_audit_table.insert().values(
                timestamp=timestamp_value,
                conversation_id=str(audit_record.get("conversation_id") or ""),
                user_id=str(audit_record.get("user_id") or ""),
                tool_name=str(audit_record.get("tool_name") or ""),
                tool_args=json.dumps(audit_record.get("tool_args") or {}, ensure_ascii=False),
                result=json.dumps(audit_record.get("result"), ensure_ascii=False) if audit_record.get("result") is not None else None,
                error=json.dumps(audit_record.get("error"), ensure_ascii=False) if audit_record.get("error") is not None else None,
                execution_time_ms=execution_time_ms,
                status=str(audit_record.get("status") or "success"),
                metadata=json.dumps(audit_record.get("metadata") or {}, ensure_ascii=False),
            )
            session.execute(insert_stmt)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if should_close:
                session.close()
    
    def get_audit_records(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list:
        session = self.db_session or DatabaseFactory.create_session()
        should_close = self.db_session is None
        try:
            if not self._table_ready:
                _metadata.create_all(bind=session.get_bind(), tables=[_tool_call_audit_table], checkfirst=True)
                self._table_ready = True
            conditions = []
            if conversation_id:
                conditions.append(_tool_call_audit_table.c.conversation_id == str(conversation_id))
            if user_id:
                conditions.append(_tool_call_audit_table.c.user_id == str(user_id))
            if tool_name:
                conditions.append(_tool_call_audit_table.c.tool_name == str(tool_name))
            if start_time:
                conditions.append(_tool_call_audit_table.c.timestamp >= start_time)
            if end_time:
                conditions.append(_tool_call_audit_table.c.timestamp <= end_time)
            stmt = select(_tool_call_audit_table)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.order_by(_tool_call_audit_table.c.timestamp.desc()).limit(max(1, int(limit)))
            rows = session.execute(stmt).mappings().all()
            records = []
            for row in rows:
                records.append(
                    {
                        "id": row.get("id"),
                        "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
                        "conversation_id": row.get("conversation_id"),
                        "user_id": row.get("user_id"),
                        "tool_name": row.get("tool_name"),
                        "tool_args": json.loads(row.get("tool_args")) if row.get("tool_args") else {},
                        "result": json.loads(row.get("result")) if row.get("result") else None,
                        "error": json.loads(row.get("error")) if row.get("error") else None,
                        "execution_time_ms": row.get("execution_time_ms"),
                        "status": row.get("status"),
                        "metadata": json.loads(row.get("metadata")) if row.get("metadata") else {},
                    }
                )
            return records
        finally:
            if should_close:
                session.close()


# 全局审计日志器实例
_audit_logger = ToolCallAuditLogger()


def get_audit_logger() -> ToolCallAuditLogger:
    """获取全局审计日志器实例"""
    return _audit_logger
