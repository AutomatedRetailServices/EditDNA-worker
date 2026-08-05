from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rq_worker

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeConnection:
    def __init__(self) -> None:
        self.ping_calls = 0

    def ping(self) -> bool:
        self.ping_calls += 1
        return True


class FakeRedis:
    calls = []
    connection = FakeConnection()

    @classmethod
    def from_url(cls, redis_url, **kwargs):
        cls.calls.append((redis_url, kwargs))
        return cls.connection


class FakeWorker:
    instances = []
    work_exception = None
    work_result = True

    def __init__(self, queues, connection=None, worker_ttl=None):
        self.queues = queues
        self.connection = connection
        self.worker_ttl = worker_ttl
        self.work_calls = 0
        self.work_kwargs = []
        FakeWorker.instances.append(self)

    def work(self, **kwargs):
        self.work_calls += 1
        self.work_kwargs.append(kwargs)
        if FakeWorker.work_exception is not None:
            raise FakeWorker.work_exception
        return FakeWorker.work_result


@pytest.fixture(autouse=True)
def launcher_fakes(monkeypatch):
    FakeRedis.calls = []
    FakeRedis.connection = FakeConnection()
    FakeWorker.instances = []
    FakeWorker.work_exception = None
    FakeWorker.work_result = True
    monkeypatch.setattr(rq_worker, "Redis", FakeRedis)
    monkeypatch.setattr(rq_worker, "Worker", FakeWorker)


def test_default_queue_selection(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "rediss://:secret@example.com:6380/0")
    monkeypatch.delenv("QUEUE_NAME", raising=False)

    assert rq_worker.run_worker() is True

    worker = FakeWorker.instances[0]
    assert worker.queues == ["default"]


def test_explicit_queue_selection(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "rediss://:secret@example.com:6380/0")
    monkeypatch.setenv("QUEUE_NAME", "render")

    assert rq_worker.run_worker() is True

    worker = FakeWorker.instances[0]
    assert worker.queues == ["render"]


def test_redis_from_url_receives_required_connection_options(monkeypatch):
    redis_url = "rediss://:secret@example.com:6380/0"
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.delenv("QUEUE_NAME", raising=False)

    rq_worker.run_worker()

    assert FakeRedis.calls == [(redis_url, rq_worker.REDIS_CONNECTION_OPTIONS)]
    _, options = FakeRedis.calls[0]
    assert options["socket_keepalive"] is True
    assert options["health_check_interval"] == 30
    assert options["retry_on_timeout"] is True
    assert options["socket_connect_timeout"] == 10
    assert "socket_timeout" not in options


def test_ping_is_performed_before_worker_startup(monkeypatch):
    events = []

    class OrderedConnection:
        def ping(self):
            events.append("ping")

    class OrderedRedis:
        @classmethod
        def from_url(cls, redis_url, **kwargs):
            return OrderedConnection()

    class OrderedWorker(FakeWorker):
        def __init__(self, queues, connection=None, worker_ttl=None):
            events.append("worker")
            super().__init__(queues, connection=connection, worker_ttl=worker_ttl)

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(rq_worker, "Redis", OrderedRedis)
    monkeypatch.setattr(rq_worker, "Worker", OrderedWorker)

    rq_worker.run_worker()

    assert events == ["ping", "worker"]


def test_worker_receives_queue_connection_and_worker_ttl(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QUEUE_NAME", "critical")

    rq_worker.run_worker()

    worker = FakeWorker.instances[0]
    assert worker.queues == ["critical"]
    assert worker.connection is FakeRedis.connection
    assert worker.worker_ttl == 3600
    assert worker.work_calls == 1
    assert worker.work_kwargs == [{"with_scheduler": True}]


def test_main_exits_zero_when_worker_work_returns_true(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    FakeWorker.work_result = True

    assert rq_worker.main() == 0

    assert FakeWorker.instances[0].work_calls == 1
    assert FakeWorker.instances[0].work_kwargs == [{"with_scheduler": True}]


def test_main_exits_zero_when_worker_work_returns_false(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    FakeWorker.work_result = False

    assert rq_worker.main() == 0

    assert FakeWorker.instances[0].work_calls == 1
    assert FakeWorker.instances[0].work_kwargs == [{"with_scheduler": True}]


def test_main_exits_nonzero_when_worker_work_raises_without_credentials(monkeypatch, capsys):
    redis_url = "rediss://user:super-secret@example.com:6380/0"
    monkeypatch.setenv("REDIS_URL", redis_url)
    FakeWorker.work_exception = RuntimeError(f"lost connection to {redis_url}")

    assert rq_worker.main() == 1

    output = capsys.readouterr()
    combined = output.out + output.err
    assert FakeWorker.instances[0].work_kwargs == [{"with_scheduler": True}]
    assert "RQ worker exited unexpectedly (RuntimeError)" in output.err
    assert "super-secret" not in combined
    assert "user:super-secret" not in combined
    assert redis_url not in combined


def test_no_redis_password_or_complete_url_is_printed(monkeypatch, capsys):
    redis_url = "rediss://user:super-secret@example.com:6380/0"
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("QUEUE_NAME", "default")

    rq_worker.run_worker()

    output = capsys.readouterr()
    combined = output.out + output.err
    assert "super-secret" not in combined
    assert "user:super-secret" not in combined
    assert redis_url not in combined
    assert "rediss://***@example.com:6380/0" in combined


def test_missing_redis_url_exits_clearly(monkeypatch, capsys):
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert rq_worker.main() == 1

    assert "REDIS_URL is required" in capsys.readouterr().err
    assert FakeWorker.instances == []


def test_initial_redis_connection_failure_exits_nonzero(monkeypatch, capsys):
    class FailingConnection:
        def ping(self):
            raise rq_worker.RedisError("boom")

    class FailingRedis:
        @classmethod
        def from_url(cls, redis_url, **kwargs):
            return FailingConnection()

    monkeypatch.setenv("REDIS_URL", "rediss://:secret@example.com:6380/0")
    monkeypatch.setattr(rq_worker, "Redis", FailingRedis)

    assert rq_worker.main() == 1

    output = capsys.readouterr()
    assert "Could not connect to Redis" in output.err
    assert "secret" not in output.out + output.err
    assert FakeWorker.instances == []


def test_initial_ping_timeout_exits_nonzero_without_credentials(monkeypatch, capsys):
    class TimeoutConnection:
        def ping(self):
            raise TimeoutError("timed out")

    class TimeoutRedis:
        @classmethod
        def from_url(cls, redis_url, **kwargs):
            return TimeoutConnection()

    monkeypatch.setenv("REDIS_URL", "rediss://user:secret@example.com:6380/0")
    monkeypatch.setattr(rq_worker, "Redis", TimeoutRedis)

    assert rq_worker.main() == 1

    output = capsys.readouterr()
    assert "Could not connect to Redis" in output.err
    assert "secret" not in output.out + output.err
    assert FakeWorker.instances == []


def test_start_worker_resolves_script_dir_and_uses_python3_from_external_cwd(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_file = tmp_path / "python3-calls.txt"
    real_python = sys.executable
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{calls_file}"
if [ "${{1:-}}" = "-" ]; then
  exec "{real_python}" "$@"
fi
if [ "${{1:-}}" = "{REPO_ROOT / 'rq_worker.py'}" ]; then
  echo "fake worker started: $1"
  exit 0
fi
echo "unexpected python3 invocation: $*" >&2
exit 91
"""
    )
    fake_python.chmod(0o755)

    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    redis_url = "rediss://user:super-secret@example.com:6380/0"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REDIS_URL": redis_url,
        "QUEUE_NAME": "render",
    }

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "start_worker.sh")],
        cwd=outside_cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "super-secret" not in combined
    assert "user:super-secret" not in combined
    assert redis_url not in combined
    assert "rediss://***@example.com:6380/0" in combined

    calls = calls_file.read_text().splitlines()
    assert calls[0] == "-"
    assert calls[-1] == str(REPO_ROOT / "rq_worker.py")
    assert f"PYTHONPATH ....... {REPO_ROOT}:" in result.stdout
