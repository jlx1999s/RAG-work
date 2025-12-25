"""测试糖尿病风险评估工具的ReAct Agent集成"""
import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.agent.graph.raggraph import RAGGraph
from backend.agent.contexts.raggraph_context import RAGContext
from backend.agent.models.raggraph_models import RetrievalMode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

async def test_diabetes_risk_assessment():
    """测试糖尿病风险评估工具调用"""
    
    print("=" * 60)
    print("测试：糖尿病风险评估工具（ReAct Agent架构）")
    print("=" * 60)
    
    # 初始化LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    # 创建RAG Graph（不需要embedding_model，因为不做检索）
    rag_graph = RAGGraph(
        llm=llm,
        embedding_model=None,
        enable_checkpointer=False  # 测试时禁用checkpoint
    )
    
    # 创建上下文
    context = RAGContext(
        user_id="test_user",
        session_id="test_session_diabetes"
    )
    
    # 测试用例1: 提供核心参数（年龄、BMI、血压、家族史）
    test_question = "请帮我进行糖尿病风险评估，我今年55岁，BMI 26.5，血压偏高，有家族史"
    
    print(f"\n📝 测试问题：{test_question}")
    print("\n" + "=" * 60)
    
    input_data = {
        "question": test_question,
        "retrieval_mode": RetrievalMode.NO_RETRIEVAL  # 不需要检索
    }
    
    try:
        # 异步流式执行
        print("\n🔄 开始执行 RAG Graph（流式输出）...\n")
        
        async for step in rag_graph.astream(
            input_data=input_data,
            context=context,
            stream_mode="updates"
        ):
            # 输出每个节点的执行情况
            for node_name, node_state in step.items():
                print(f"\n📍 节点: {node_name}")
                
                # 输出关键状态信息
                if "need_retrieval" in node_state:
                    print(f"   - need_retrieval: {node_state['need_retrieval']}")
                if "need_tool" in node_state:
                    print(f"   - need_tool: {node_state['need_tool']}")
                if "selected_tool" in node_state:
                    print(f"   - selected_tool: {node_state['selected_tool']}")
                if "final_answer" in node_state and node_state["final_answer"]:
                    print(f"\n✅ 最终回答:\n{node_state['final_answer']}")
        
        print("\n" + "=" * 60)
        print("✅ 测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_diabetes_risk_assessment())
