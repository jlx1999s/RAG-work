"""工具调用相关的异常和错误码定义

企业级最佳实践：
1. 结构化的错误码，便于前端处理
2. 详细的错误信息，便于调试
3. 可扩展的错误分类
"""

from typing import Optional, Dict, Any
from datetime import datetime


class ToolCallErrorCode:
    """工具调用错误码"""
    
    # 超时错误
    TIMEOUT = "TOOL_TIMEOUT"
    
    # 权限错误
    PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"
    
    # 速率限制错误
    RATE_LIMIT_EXCEEDED = "TOOL_RATE_LIMIT_EXCEEDED"
    
    # 参数验证错误
    VALIDATION_ERROR = "TOOL_VALIDATION_ERROR"
    
    # 工具执行错误
    EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    
    # 工具不存在
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    
    # 未知错误
    UNKNOWN_ERROR = "TOOL_UNKNOWN_ERROR"


class ToolCallException(Exception):
    """工具调用基础异常"""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            code: 错误码
            message: 用户友好的错误消息
            details: 详细的技术错误信息（用于日志）
            metadata: 额外的元数据
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于JSON序列化"""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ToolExecutionTimeoutError(ToolCallException):
    """工具执行超时异常"""
    
    def __init__(
        self, 
        tool_name: str, 
        timeout: float,
        details: Optional[str] = None
    ):
        super().__init__(
            code=ToolCallErrorCode.TIMEOUT,
            message=f"工具 '{tool_name}' 执行超时（{timeout}秒）",
            details=details,
            metadata={
                "tool_name": tool_name,
                "timeout_seconds": timeout
            }
        )


class ToolPermissionDeniedError(ToolCallException):
    """工具权限拒绝异常"""
    
    def __init__(
        self, 
        tool_name: str, 
        user_role: str,
        required_roles: list,
        details: Optional[str] = None
    ):
        super().__init__(
            code=ToolCallErrorCode.PERMISSION_DENIED,
            message=f"用户角色 '{user_role}' 无权调用工具 '{tool_name}'",
            details=details,
            metadata={
                "tool_name": tool_name,
                "user_role": user_role,
                "required_roles": required_roles
            }
        )


class ToolRateLimitExceededError(ToolCallException):
    """工具调用速率超限异常"""
    
    def __init__(
        self, 
        tool_name: str,
        max_calls: int,
        time_window: int,
        details: Optional[str] = None
    ):
        super().__init__(
            code=ToolCallErrorCode.RATE_LIMIT_EXCEEDED,
            message=f"工具 '{tool_name}' 调用频率超限，请{time_window}秒后重试",
            details=details,
            metadata={
                "tool_name": tool_name,
                "max_calls": max_calls,
                "time_window_seconds": time_window
            }
        )


class ToolValidationError(ToolCallException):
    """工具参数验证异常"""
    
    def __init__(
        self, 
        tool_name: str,
        validation_errors: Dict[str, str],
        details: Optional[str] = None
    ):
        super().__init__(
            code=ToolCallErrorCode.VALIDATION_ERROR,
            message=f"工具 '{tool_name}' 参数验证失败",
            details=details,
            metadata={
                "tool_name": tool_name,
                "validation_errors": validation_errors
            }
        )


class ToolExecutionError(ToolCallException):
    """工具执行异常"""
    
    def __init__(
        self, 
        tool_name: str,
        error_message: str,
        details: Optional[str] = None
    ):
        super().__init__(
            code=ToolCallErrorCode.EXECUTION_ERROR,
            message=f"工具 '{tool_name}' 执行失败: {error_message}",
            details=details,
            metadata={
                "tool_name": tool_name,
                "error_message": error_message
            }
        )


class ToolNotFoundError(ToolCallException):
    """工具不存在异常"""
    
    def __init__(
        self, 
        tool_name: str,
        available_tools: list,
        details: Optional[str] = None
    ):
        super().__init__(
            code=ToolCallErrorCode.TOOL_NOT_FOUND,
            message=f"工具 '{tool_name}' 不存在",
            details=details,
            metadata={
                "tool_name": tool_name,
                "available_tools": available_tools
            }
        )
