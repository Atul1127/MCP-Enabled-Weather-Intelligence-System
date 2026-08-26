import json

import observability


def test_nested_spans_have_parent_ids(tmp_path, monkeypatch):
    log_path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(observability, "LOG_PATH", log_path)
    trace_id = observability.new_trace_id()

    with observability.span("outer", trace_id=trace_id):
        with observability.span("inner", trace_id=trace_id):
            pass

    events = observability.read_trace(trace_id)
    ends = [event for event in events if event["event"] == "span.end"]
    assert len(ends) == 2
    outer = next(event for event in ends if event["span"] == "outer")
    inner = next(event for event in ends if event["span"] == "inner")
    assert outer["parent_span_id"] is None
    assert inner["parent_span_id"] == outer["span_id"]


def test_trace_summary_reports_tools_and_spans(tmp_path, monkeypatch):
    log_path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(observability, "LOG_PATH", log_path)
    trace_id = observability.new_trace_id()
    observability.emit("agent.tool", trace_id=trace_id, tool="get_weather", success=True)
    with observability.span("agent.reason", trace_id=trace_id):
        pass

    summary = observability.summarize_trace(trace_id)
    assert summary["tool_calls"] == 1
    assert summary["tools"] == ["get_weather"]
    assert summary["spans"][0]["span"] == "agent.reason"
