from evaluation.exception_utils import classify_exception, root_exception


def test_root_exception_unwraps_exception_group():
    leaf = RuntimeError("inner failure")
    group = ExceptionGroup("outer", [ExceptionGroup("inner", [leaf])])
    assert root_exception(group) is leaf


def test_quota_is_infrastructure_failure():
    assert classify_exception(RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")) == "quota_exhausted"


def test_authentication_is_infrastructure_failure():
    assert classify_exception(RuntimeError("GEMINI_API_KEY is not set")) == "authentication_failure"


def test_network_is_infrastructure_failure():
    assert classify_exception(TimeoutError("network timeout")) == "network_failure"


def test_application_errors_remain_application_errors():
    assert classify_exception(ValueError("invalid plan")) == "ValueError"
