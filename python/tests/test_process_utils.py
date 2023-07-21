
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import pytest

from coml.process_utils import graceful_kill, run_subprocess

_script_normal = """
import time
time.sleep(360)
""".strip()

_script_handle_sigint = """
import time, sys, signal

def handler_stop_signals(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, handler_stop_signals)
time.sleep(360)
""".strip()

_script_no_sigint = """
import time

for _ in range(10):
    try:
        time.sleep(360)
    except KeyboardInterrupt:
        pass
""".strip()

_script_ended = """
print('hello world')
"""

@pytest.fixture
def script_file():
    with tempfile.NamedTemporaryFile('w', suffix='.py') as f:
        yield f

@pytest.fixture
def process_normal(script_file):
    script_file.write(_script_normal)
    script_file.flush()
    yield subprocess.Popen([sys.executable, script_file.name])

@pytest.fixture
def process_handle_sigint(script_file):
    script_file.write(_script_handle_sigint)
    script_file.flush()
    yield subprocess.Popen([sys.executable, script_file.name])

@pytest.fixture
def process_no_sigint(script_file):
    script_file.write(_script_no_sigint)
    script_file.flush()
    yield subprocess.Popen([sys.executable, script_file.name])

@pytest.fixture
def process_ended(script_file):
    script_file.write(_script_ended)
    script_file.flush()
    yield subprocess.Popen([sys.executable, script_file.name])

def test_run_subprocess():
    with tempfile.NamedTemporaryFile('w', suffix='.log') as f:
        retcode, stdout, stderr = run_subprocess([sys.executable, '-c', 'print("hello world")'], None, Path(f.name))
        assert Path(f.name).read_text().endswith('hello world\n')
        assert retcode == 0
        assert stdout == b'hello world\n'
        assert stderr == b''

def test_subprocess_three_seconds():
    with tempfile.NamedTemporaryFile('w', suffix='.log') as f:
        retcode, stdout, stderr = run_subprocess([sys.executable, '-c', 'import time; time.sleep(3); print("hello world")'], None, Path(f.name))
        assert retcode == 0
        assert stdout == b'hello world\n'
        assert stderr == b''

def test_subprocess_timeout():
    with tempfile.NamedTemporaryFile('w', suffix='.log') as f:
        retcode, stdout, stderr = run_subprocess([sys.executable, '-c', 'import time, sys; print("hello world", file=sys.stderr, flush=True); time.sleep(3600)'], None, Path(f.name), timeout=1)
        assert retcode != 0
        assert stdout == b''
        assert stderr == b'hello world\n'
        print(Path(f.name).read_text())

def test_kill(process_normal):
    time.sleep(1.)
    assert process_normal.poll() is None
    assert graceful_kill(process_normal) != 0
    assert isinstance(process_normal.returncode, int)

def test_kill_handled(process_handle_sigint):
    time.sleep(1.)
    graceful_kill(process_handle_sigint)
    assert process_handle_sigint.returncode == 0

def test_kill_hang(process_no_sigint, caplog):
    time.sleep(1.)
    assert graceful_kill(process_no_sigint) != 0
    assert 'terminate' in caplog.text

def test_kill_ended(process_ended):
    time.sleep(1.)
    assert graceful_kill(process_ended) == 0
