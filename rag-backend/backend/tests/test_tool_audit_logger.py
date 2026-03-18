from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agent.tools.audit import ToolCallAuditLogger


def _create_logger():
    engine = create_engine("sqlite:///:memory:", future=True)
    session = sessionmaker(bind=engine, future=True)()
    return ToolCallAuditLogger(db_session=session), session


def test_tool_audit_logger_should_persist_record():
    audit_logger, session = _create_logger()
    audit_logger.log_tool_call(
        conversation_id="conv-1",
        user_id="user-1",
        tool_name="hypertension_risk_assessment",
        tool_args={"age": 36, "systolic_bp": 130, "diastolic_bp": 85},
        result={"risk_level": "low"},
        execution_time_ms=42.5,
        metadata={"source": "unit_test"},
    )

    records = audit_logger.get_audit_records(conversation_id="conv-1", limit=10)

    assert len(records) == 1
    assert records[0]["conversation_id"] == "conv-1"
    assert records[0]["user_id"] == "user-1"
    assert records[0]["tool_name"] == "hypertension_risk_assessment"
    assert records[0]["tool_args"]["age"] == 36
    assert records[0]["result"]["risk_level"] == "low"
    assert records[0]["status"] == "success"
    session.close()


def test_tool_audit_logger_should_filter_by_tool_name():
    audit_logger, session = _create_logger()
    audit_logger.log_tool_call(
        conversation_id="conv-2",
        user_id="user-2",
        tool_name="hypertension_risk_assessment",
        tool_args={"age": 40, "systolic_bp": 135, "diastolic_bp": 88},
    )
    audit_logger.log_tool_call(
        conversation_id="conv-2",
        user_id="user-2",
        tool_name="diabetes_risk_assessment",
        tool_args={"age": 40, "bmi": 26.2},
        error={"code": "missing"},
    )

    filtered_records = audit_logger.get_audit_records(
        conversation_id="conv-2",
        tool_name="diabetes_risk_assessment",
        limit=10,
    )

    assert len(filtered_records) == 1
    assert filtered_records[0]["tool_name"] == "diabetes_risk_assessment"
    assert filtered_records[0]["status"] == "error"
    session.close()
