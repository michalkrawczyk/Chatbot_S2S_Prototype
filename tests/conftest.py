import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires real API key / external service")
    config.addinivalue_line("markers", "integration_lite: real filesystem I/O only, no network")
    config.addinivalue_line("markers", "slow: long-running (model load etc.)")
    config.addinivalue_line("markers", "nemo: requires nemo_toolkit[asr] and GPU/CPU model download")
