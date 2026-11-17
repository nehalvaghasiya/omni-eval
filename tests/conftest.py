"""Tests import the current checked-out package, not another global release."""

import importlib.util
import ipaddress
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_sessionstart(session):
    if os.getenv("OMNIEVAL_REQUIRE_EXTRAS") == "1":
        missing = [
            name
            for name in ("fastapi", "uvicorn", "numpy", "sklearn", "pyarrow")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise pytest.UsageError(
                "Full verification needs make setup; missing: " + ", ".join(missing)
            )


@pytest.fixture(autouse=True)
def offline_network(monkeypatch):
    """Ordinary tests may connect only to loopback, never public services."""
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex

    def check(sock, address):
        if sock.family == socket.AF_UNIX:
            return
        host = address[0]
        if host == "localhost":
            return
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if not local:
            raise PermissionError("test network is restricted to loopback")

    def connect(sock, address):
        check(sock, address)
        return original(sock, address)

    def connect_ex(sock, address):
        check(sock, address)
        return original_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
