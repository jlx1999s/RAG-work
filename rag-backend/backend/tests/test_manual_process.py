"""手动触发文档解析测试"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.service.knowledge_library import start_document_processing


async def main():
    """测试手动触发文档解析"""
    # 测试文档ID 27（血脂管理指南）
    document_id = 27
    user_id = "1"
    
    print(f"开始测试文档 {document_id} 的解析...")
    
    result = await start_document_processing(document_id, user_id)
    
    print(f"\n解析结果:")
    print(f"  状态码: {result.status}")
    print(f"  消息: {result.msg}")
    print(f"  数据: {result.data}")
    
    # 等待30秒让处理完成
    print("\n等待 30 秒让文档处理完成...")
    await asyncio.sleep(30)
    
    print("\n测试完成！请运行检查脚本查看结果:")
    print("uv run python backend/tests/test_check_milvus_data.py")


if __name__ == "__main__":
    asyncio.run(main())
