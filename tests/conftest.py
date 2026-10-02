def pytest_report_teststatus(report, config):
    """Pytest hook: emit only 'F' for failed tests and nothing for passing/skipped tests."""
    if report.when in ("setup", "call") and report.failed:
        return "failed", "F", "FAILED"
    if report.passed:
        return "passed", "", ""
    if report.skipped:
        return "skipped", "", ""
