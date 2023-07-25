import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from subprocess import Popen, PIPE

_logger = logging.getLogger(__name__)


def run_subprocess(
    command: List[str],
    working_directory: Optional[Path] = None,
    log_file: Optional[Path] = None,
    timeout: Optional[float] = None,
    streaming: bool = True
) -> Tuple[int, bytes, bytes]:
    """Run a subprocess, streaming and capturing its output.
    This is from another project.
    """
    if timeout:
        _logger.debug("Running command with timeout %f seconds: %s", timeout, command)
    else:
        _logger.debug("Running command: %s", command)
    if working_directory:
        _logger.debug("Working directory: %s", working_directory)
    if log_file:
        _logger.debug("Output saved to: %s", log_file)

    stdout, stderr = b"", b""
    file_handle = None
    proc = None
    try:
        start_time = time.time()
        if log_file is not None:
            file_handle = log_file.open("wb")
            file_handle.write(f"Command: {command}\n".encode())
        proc = Popen(
            command,
            stdout=PIPE,
            stderr=PIPE,
            cwd=working_directory,
            bufsize=0,
        )

        # TODO:
        # Piping output is sometimes not working on certain environments.
        # Further investigation is needed.

        # Make the stdout and stderr non-blocking

        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)

        while True:
            # Read stdout
            while True:
                try:
                    out = os.read(proc.stdout.fileno(), 128)
                except BlockingIOError:
                    out = b""
                    pass
                if out:
                    if streaming:
                        sys.stdout.buffer.write(out)
                        sys.stdout.flush()
                    if file_handle is not None:
                        file_handle.write(out)
                    stdout += out
                else:
                    break

            # Read stderr
            while True:
                try:
                    err = os.read(proc.stderr.fileno(), 128)
                except BlockingIOError:
                    err = b""
                    pass
                if err:
                    if streaming:
                        sys.stderr.buffer.write(err)
                        sys.stderr.flush()
                    if file_handle is not None:
                        file_handle.write(err)
                    stderr += err
                else:
                    break

            if file_handle is not None:
                file_handle.flush()

            # See if the process has terminated
            if proc.poll() is not None:
                returncode = proc.returncode
                if returncode != 0:
                    _logger.error("Command failed with return code %d: %s", returncode, command)
                else:
                    _logger.debug("Command finished with return code %d: %s", returncode, command)
                return returncode, stdout, stderr

            # See if we timed out
            if timeout is not None and time.time() - start_time > timeout:
                _logger.warning("Command timed out (%f seconds): %s", timeout, command)
                returncode = graceful_kill(proc)
                if returncode is None:
                    _logger.error("Return code is still none after attempting to kill it. The process (%d) may be stuck.", proc.pid)
                    returncode = 1
                return returncode, stdout, stderr

            time.sleep(1)

    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            _logger.warning("Command still running as of keyboard interrupt. Kill it: %s", command)
            graceful_kill(proc)

    finally:
        if proc is not None and proc.poll() is None:
            _logger.warning("Command still running as of cleanup. Try to kill it (again): %s", command)
            graceful_kill(proc)

        if file_handle is not None:
            file_handle.close()

    return -1, b"", b""


def graceful_kill(popen: Popen) -> Optional[int]:
    """Gracefully kill a process."""
    try:
        for retry in [1., 5., 20., 60.]:
            _logger.info("Gracefully terminating %s...", popen)

            if retry > 10:
                _logger.info("Use \"terminate\" instead of \"interrupt\".")
                popen.terminate()
            else:
                popen.send_signal(signal.SIGINT)

            time.sleep(1.)  # Wait for the kill to take effect.

            retcode = popen.poll()
            if retcode is not None:
                return retcode

            _logger.warning("%s still alive. Retry to kill in %d seconds.", popen, retry)
            time.sleep(retry)

        _logger.warning("Force kill process %s...", popen)
        time.sleep(10.)  # Wait for the kill
        retcode = popen.poll()
        if retcode is not None:
            return retcode

        _logger.error("Failed to kill process %s.", popen)
    except KeyboardInterrupt:
        _logger.error("Interrupted while killing process %s. Process pid is %s.", popen, popen.pid)
    return None
