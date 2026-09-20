import socket

import pytest
import requests


@pytest.fixture(autouse=True)
def no_network_or_real_credentials(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Evaluation harness tests must not use the network")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
