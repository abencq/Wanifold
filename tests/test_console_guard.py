"""Tests for shared/utils/console_guard.py and shared/utils/subprocess_stdin.py.

The terminal is guarded on a real pty: the child process runs inside a terminal of its own, the
tests put that terminal in raw mode (what a helper program that reads the keyboard does), and they
check the settings the child leaves behind. Run with:

    venv/bin/python -m pytest tests/test_console_guard.py -q -s
"""
import fcntl
import importlib.util
import io
import os
import pty
import re
import select
import signal
import subprocess
import sys
import termios
import time

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD_PATH = os.path.join(ROOT, "shared", "utils", "console_guard.py")
STDIN_PATH = os.path.join(ROOT, "shared", "utils", "subprocess_stdin.py")

TYPABLE = ("ICANON", "ECHO", "ECHOCTL", "ECHOE", "ISIG", "OPOST")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def module():
    """A private copy of the guard module, with the signal handlers put back afterwards."""
    saved_env = {key: value for key, value in os.environ.items() if key.startswith("WAN2GP_CONSOLE")}
    for key in list(os.environ):
        if key.startswith("WAN2GP_CONSOLE"):
            del os.environ[key]
    saved_handlers = {name: signal.getsignal(getattr(signal, name))
                      for name in ("SIGINT", "SIGTERM") if hasattr(signal, name)}
    module = load_module(f"wgp_console_guard_test_{id(object())}", GUARD_PATH)
    try:
        yield module
    finally:
        for name, handler in saved_handlers.items():
            try:
                signal.signal(getattr(signal, name), handler)
            except (ValueError, OSError, RuntimeError, TypeError):
                pass
        os.environ.update(saved_env)
        for key in list(os.environ):
            if key.startswith("WAN2GP_CONSOLE") and key not in saved_env:
                del os.environ[key]
        sys.modules.pop(module.__name__, None)


@pytest.fixture
def stdin_module():
    module = load_module(f"wgp_subprocess_stdin_test_{id(object())}", STDIN_PATH)
    yield module
    sys.modules.pop(module.__name__, None)


def show(text):
    text = str(text)
    print(text if text.endswith("\n") else f"{text}\n")


def seed_typable(fd):
    """Give the pty the settings a real terminal has: Ctrl+C, line editing and output processing."""
    attrs = termios.tcgetattr(fd)
    attrs[1] |= termios.OPOST
    attrs[3] |= (termios.ISIG | termios.ICANON | termios.ECHO |
                 termios.ECHOCTL | termios.ECHOE | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def raw_the_terminal(fd):
    """Take the terminal over the way a native tool does: no echo, no line editing, no Ctrl+C."""
    attrs = termios.tcgetattr(fd)
    attrs[1] &= ~termios.OPOST
    attrs[3] &= ~(termios.ISIG | termios.ICANON | termios.ECHO | termios.ECHOE | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)



def flag_names(fd):
    """The typable-terminal flags a descriptor currently has, as a set."""
    attrs = termios.tcgetattr(fd)
    oflags, lflags = attrs[1], attrs[3]
    bits = {"ISIG": (lflags, termios.ISIG), "OPOST": (oflags, termios.OPOST)}
    on = set()
    for name in TYPABLE:
        flags, bit = bits.get(name, (lflags, getattr(termios, name)))
        if flags & bit:
            on.add(name)
    return on


def run_in_pty(source, setup=None, timeout=15.0):
    """Run ``source`` (Python) inside a pty of its own; return what it printed and the final flags.

    The parent keeps a slave descriptor open, so the terminal settings can be read after the child
    is gone -- exactly the state the shell would be left in.
    """
    master, slave = pty.openpty()
    if setup is not None:
        setup(slave)
    pid = os.fork()
    if pid == 0:
        os.close(master)
        os.setsid()
        fcntl.ioctl(slave, 0x5409, 0)                   # TIOCSCTTY: this pty is the child's terminal
        for target in (0, 1, 2):
            os.dup2(slave, target)
        if slave > 2:
            os.close(slave)
        os.execv(sys.executable, [sys.executable, "-c", source])
    output = b""
    deadline = time.time() + timeout
    alive = True
    while time.time() < deadline:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                chunk = b""
            if chunk:
                output += chunk
        if alive:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    alive = False
            except ChildProcessError:
                alive = False
        if not alive and not select.select([master], [], [], 0.2)[0]:
            break
    flags = flag_names(slave)
    os.close(slave)
    os.close(master)
    if alive:                                                   # the child hung: do not leak it
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    return output.decode("utf-8", "replace"), flags


def child_source(body, module_path=GUARD_PATH):
    """A child program with the module under test imported from its real file."""
    return (
        "import importlib.util, os, sys, termios, tty, signal\n"
        f"spec = importlib.util.spec_from_file_location('cg', {module_path!r})\n"
        "cg = importlib.util.module_from_spec(spec)\n"
        "sys.modules['cg'] = cg\n"
        "spec.loader.exec_module(cg)\n"
        f"{body}\n"
    )


def test_a_terminal_is_guarded_and_a_pipe_is_left_alone(module):
    """A run on a terminal is guarded; a run writing to a pipe has no terminal to guard."""
    out, flags = run_in_pty(child_source(
        "print('installed:', cg.install_console_guard([], report=False), flush=True)\n"
        "print('state:', cg.console_guard_state(), flush=True)\n"
    ))
    show(out)
    assert "installed: True" in out
    assert set(TYPABLE) <= flags, "a healthy terminal is handed back healthy"

    probe = subprocess.run(
        [sys.executable, "-c", child_source("print('installed:', cg.install_console_guard([]))")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True,
    )
    show(probe.stdout.decode())
    assert b"installed: False" in probe.stdout, "no terminal: the guard installs nothing"


def test_the_guard_changes_nothing_on_a_healthy_terminal(module):
    out, flags = run_in_pty(child_source(
        "before = cg.console_guard_state()\n"
        "cg.install_console_guard([], report=False)\n"
        "print('before:', before, flush=True)\n"
        "print('after:', cg.console_guard_state(), flush=True)\n"
    ), setup=seed_typable)
    show(out)
    assert "repaired" not in out.lower(), "a terminal that is fine is left alone"
    before = re.search(r"before: (.*)", out).group(1).strip()
    after = re.search(r"after: (.*)", out).group(1).strip()
    assert before == after, "a healthy terminal is not touched"
    assert set(TYPABLE) <= flags


def test_a_terminal_left_raw_by_a_child_is_put_back_on_the_way_out(module):
    """The reported defect: a helper program takes the terminal over, is killed hard, echo stays off."""
    out, flags = run_in_pty(child_source(
        "cg.install_console_guard([], report=False)\n"
        "tty.setraw(sys.stdin.fileno())\n"                     # a native tool taking the keyboard over
        "print('echo off:', not (termios.tcgetattr(0)[3] & termios.ECHO), flush=True)\n"
        "print('icanon off:', not (termios.tcgetattr(0)[3] & termios.ICANON), flush=True)\n"
        "print('isig off:', not (termios.tcgetattr(0)[3] & termios.ISIG), flush=True)\n"
        "print('restored:', cg.restore_console(), flush=True)\n"
    ), setup=seed_typable)
    show(out)
    assert "echo off: True" in out and "icanon off: True" in out and "isig off: True" in out
    assert "restored: True" in out
    assert set(TYPABLE) <= flags, "the terminal is typable again after the run ends"



def test_the_exit_handler_restores_the_terminal_without_being_called(module):
    """Nothing calls the guard at the end: atexit must do it, as it does for a normal run."""
    out, flags = run_in_pty(child_source(
        "cg.install_console_guard([], report=False)\n"
        "tty.setraw(sys.stdin.fileno())\n"
        "print('raw now', flush=True)\n"
    ))
    show(out)
    assert "raw now" in out
    assert set(TYPABLE) <= flags


def test_a_terminal_that_arrives_broken_is_repaired_at_startup(module):
    """A previous run left the terminal raw: WanGP starts on a working console and says so."""
    def leave_raw(fd):
        attrs = termios.tcgetattr(fd)
        attrs[1] &= ~termios.OPOST
        attrs[3] &= ~(termios.ISIG | termios.ICANON | termios.ECHO | termios.ECHOCTL | termios.ECHOE)
        termios.tcsetattr(fd, termios.TCSANOW, attrs)

    out, flags = run_in_pty(child_source(
        "print('installed:', cg.install_console_guard([], report=False), flush=True)\n"
        "print('repaired:', cg._STATE['repaired'], flush=True)\n"
        "print('state:', cg.console_guard_state(), flush=True)\n"
    ), setup=leave_raw)
    show(out)
    assert "installed: True" in out and "repaired: True" in out
    assert "has been repaired" in out
    assert set(TYPABLE) <= flags, "WanGP runs on a terminal a person can type in"




def test_the_interrupt_restores_the_terminal_and_then_runs_the_handler_before_it(module):
    """Chained, never replaced: restore first, then the shutdown handler that was already installed."""
    out, flags = run_in_pty(child_source(
        "import time\n"
        "def upstream_handler(signum, frame):\n"
        "    print('upstream handler ran', flush=True)\n"
        "    raise KeyboardInterrupt\n"
        "signal.signal(signal.SIGINT, upstream_handler)\n"
        "cg.install_console_guard([], report=False)\n"
        "tty.setraw(sys.stdin.fileno())\n"
        "print('guard chained:', bool(cg._STATE['handlers']), flush=True)\n"
        "os.kill(os.getpid(), signal.SIGINT)\n"
        "time.sleep(3)\n"
    ))
    show(out)
    assert "guard chained: True" in out
    assert "upstream handler ran" in out, "the handler installed before the guard still runs"
    assert set(TYPABLE) <= flags


def test_the_guard_stands_down_when_the_option_is_off(module):
    for argv in (["--console-guard=off"], ["--console-guard", "off"]):
        out, flags = run_in_pty(child_source(
            f"print('installed:', cg.install_console_guard({argv!r}, report=False), flush=True)\n"
            "print('enabled:', cg.console_guard_enabled(), flush=True)\n"
            "tty.setraw(sys.stdin.fileno())\n"
        ))
        show(out)
        assert "installed: False" in out and "enabled: False" in out
        assert not {"ECHO", "ICANON"} <= flags, "off means the terminal is left as the child left it"

    out, flags = run_in_pty(child_source(
        "os.environ['WAN2GP_CONSOLE_GUARD'] = '0'\n"
        "print('installed:', cg.install_console_guard([], report=False), flush=True)\n"
        "tty.setraw(sys.stdin.fileno())\n"
    ))
    show(out)
    assert "installed: False" in out
    assert not {"ECHO", "ICANON"} <= flags


def test_the_mcp_stdio_server_is_left_alone(module):
    out, flags = run_in_pty(child_source(
        "print('installed:', cg.install_console_guard(['--mcp'], report=False), flush=True)\n"
        "print('stdio:', cg._uses_stdio_protocol(['--mcp']), flush=True)\n"
    ))
    show(out)
    assert "installed: False" in out and "stdio: True" in out


def test_restore_is_safe_on_every_exit_path_and_as_often_as_it_is_called(module):
    out, flags = run_in_pty(child_source(
        "cg.install_console_guard([], report=False)\n"
        "tty.setraw(sys.stdin.fileno())\n"
        "print('first:', cg.restore_console(), flush=True)\n"
        "print('again:', cg.restore_console(), flush=True)\n"
        "cg._STATE['fd'] = 9998\n"                             # a descriptor that is not a terminal
        "print('closed fd:', cg.restore_console(), flush=True)\n"
    ))
    show(out)
    assert "first: True" in out and "again: True" in out and "closed fd: False" in out
    assert set(TYPABLE) <= flags



# ── The child processes: no keyboard, their own process group ─────────────────────────────────
def test_a_child_gets_no_keyboard_and_its_own_process_group(stdin_module):
    kwargs = stdin_module.child_kwargs()
    show(kwargs)
    assert kwargs["stdin"] is subprocess.DEVNULL, "the child never reads the keyboard"
    if os.name == "posix":
        assert kwargs["start_new_session"] is True, "Ctrl+C goes to WanGP, not to the child"
        assert "creationflags" not in kwargs
    else:
        flags = kwargs["creationflags"]
        assert flags & stdin_module._CREATE_NO_WINDOW
        assert flags & stdin_module._CREATE_NEW_PROCESS_GROUP


def test_a_launch_site_can_adjust_the_options_it_gets(stdin_module):
    kwargs = stdin_module.child_kwargs(stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    show(kwargs)
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE and kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] if os.name == "posix" else True


def test_a_site_that_already_set_its_own_options_is_left_alone(stdin_module):
    assert stdin_module.keeps_terminal({"stdin": subprocess.PIPE})
    assert stdin_module.keeps_terminal({"preexec_fn": os.setpgrp})
    assert stdin_module.keeps_terminal({"start_new_session": True})
    assert not stdin_module.keeps_terminal({"capture_output": True})


def test_only_ffmpeg_knows_the_nostdin_switch(stdin_module):
    assert stdin_module.no_stdin_args("/usr/local/bin/ffmpeg") == ["-nostdin"]
    assert stdin_module.no_stdin_args("ffmpeg.exe") == ["-nostdin"]
    assert stdin_module.no_stdin_args("/usr/bin/ffprobe") == []
    assert stdin_module.no_stdin_args("ffmpeg", "ffprobe") == ["-nostdin"]


def test_a_command_reading_a_pipe_keeps_its_input(stdin_module):
    assert stdin_module.reads_stdin(["ffmpeg", "-i", "pipe:0", "-y", "out.mp4"])
    assert stdin_module.reads_stdin(["ffmpeg", "-f", "mpegts", "-i", "pipe:", "-c", "copy", "out.ts"])
    assert stdin_module.reads_stdin(["ffmpeg", "-i", "-", "out.wav"])
    assert not stdin_module.reads_stdin(["ffmpeg", "-i", "clip.mp4", "out.wav"])
    assert not stdin_module.reads_stdin(["ffprobe", "-v", "error", "clip.mp4"])


LOADER = (
    "def load(name, path):\n"
    "    spec = importlib.util.spec_from_file_location(name, path)\n"
    "    mod = importlib.util.module_from_spec(spec)\n"
    "    sys.modules[name] = mod\n"
    "    spec.loader.exec_module(mod)\n"
    "    return mod\n"
)


def test_a_child_started_the_wan_gp_way_gets_no_keyboard_and_no_interrupt(stdin_module):
    """A helper program started by WanGP cannot read the keyboard, and never gets the interrupt."""
    out, flags = run_in_pty(
        "import importlib.util, os, subprocess, sys\n"
        + LOADER +
        f"ss = load('ss', {STDIN_PATH!r})\n"
        "child = subprocess.Popen([sys.executable, '-c',\n"
        "    'import sys\\nprint(\"child read: \" + repr(sys.stdin.read()), flush=True)\\n'],\n"
        "    **ss.child_kwargs())\n"
        "print('own group:', os.getpgid(child.pid) != os.getpgid(0), flush=True)\n"
        "child.wait()\n"
        "print('child done', flush=True)\n"
    , setup=seed_typable)
    show(out)
    assert "child done" in out
    assert "child read: ''" in out, "the child reads no keyboard: it gets end of input"
    assert "own group: True" in out, "Ctrl+C goes to WanGP, the one that knows how to clean up"
    assert set(TYPABLE) <= flags


def test_the_guard_saves_the_terminal_from_a_child_that_is_killed_hard(stdin_module):
    """The exact defect, end to end: a child takes the terminal over, WanGP kills it hard, the user
    gets a working prompt back."""
    out, flags = run_in_pty(
        "import importlib.util, os, signal, subprocess, sys, termios, time, tty\n"
        + LOADER +
        f"cg = load('cg', {GUARD_PATH!r})\n"
        f"ss = load('ss', {STDIN_PATH!r})\n"
        "cg.install_console_guard([], report=False)\n"
        "child = subprocess.Popen([sys.executable, '-c',\n"
        "    'import sys, time, tty\\n'\n"
        "    'tty.setraw(1)\\n'                                  # the tool raws the shared terminal\n"
        "    'sys.stdout.flush()\\n'\n"
        "    'time.sleep(30)\\n'],\n"
        "    stdout=sys.stdout, **ss.child_kwargs())\n"
        "time.sleep(1.0)\n"
        "print('taken over:', not (termios.tcgetattr(1)[3] & termios.ECHO), flush=True)\n"
        "os.killpg(os.getpgid(child.pid), signal.SIGKILL)\n"    # WanGP kills its own child hard
        "child.wait()\n"
        "print('hard kill done', flush=True)\n"
    )
    show(out)
    assert "taken over: True" in out and "hard kill done" in out
    assert set(TYPABLE) <= flags, "the shell that comes after gets a terminal it can type in"




def test_every_ffmpeg_launch_in_the_app_hands_over_no_terminal():
    """No ffmpeg / ffprobe launch may inherit the keyboard: a site that must, says so."""
    skip = {"venv", "venv.rtx30", ".git", "node_modules", "tests", "tmp", "__pycache__",
            "SageAttention", "SageAttentionFP4", "Deepy", "preprocessing", "docs"}
    call = re.compile(r"subprocess\.(Popen|run|call|check_output|check_call)\s*\(")
    leave_alone = re.compile(r"\b(stdin|input|preexec_fn|start_new_session|creationflags)\s*=")
    checked, offenders = 0, []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                text = open(path, "r", encoding="utf-8").read()
            except Exception:
                continue
            if "ffmpeg" not in text and "ffprobe" not in text:
                continue
            for match in call.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                if text[line_start:match.start()].lstrip().startswith("#"):
                    continue
                index, depth = match.end(), 1
                while index < len(text) and depth:
                    if text[index] in "([{":
                        depth += 1
                    elif text[index] in ")]}":
                        depth -= 1
                        if depth == 0:
                            break
                    index += 1
                body = text[match.start():index + 1]
                checked += 1
                if leave_alone.search(body) or "child_kwargs" in body:
                    continue
                offenders.append(f"{os.path.relpath(path, ROOT)}:"
                                 f"{text[:match.start()].count(chr(10)) + 1}")
    show(f"launch sites checked: {checked}")
    assert offenders == [], f"launch sites left inheriting the terminal: {offenders}"
