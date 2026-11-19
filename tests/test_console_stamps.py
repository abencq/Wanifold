"""Tests for shared/utils/console_stamps.py: console timestamps, console only.

One module object per test: the feature installs itself into ``sys.stdout``/``sys.stderr`` and
keeps process-global flags, so every test loads its own copy of the module and restores the
streams behind itself. There is no log copy any more, so the whole suite asserts on the bytes the
console received: run the file as ``python -m pytest tests/test_console_stamps.py -q -s`` and read
the output the tests print.
"""
import importlib.util
import io
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import types

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(PROJECT_ROOT, "shared", "utils", "console_stamps.py")
STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")

_REAL_OUT, _REAL_ERR = sys.stdout, sys.stderr
STAMP_MARK_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ")


def show(text):
    """Send what the console received to the real terminal; pytest -s shows it."""
    _REAL_OUT.write(text)
    _REAL_OUT.flush()


def stamp_marks(text):
    """How many stamps the text carries: a bar's own pipes are not stamps."""
    return len(STAMP_MARK_RE.findall(text))



class FakeStream:
    """Records exactly the text the stamper wrote."""

    encoding = "utf-8"
    errors = "strict"

    def __init__(self):
        self.buffer = io.BytesIO()
        self.flushes = 0
        self._tty = False

    def write(self, data):
        self.buffer.write(data.encode("utf-8", "replace"))
        return len(data)

    def flush(self):
        self.flushes += 1

    def isatty(self):
        return self._tty

    def fileno(self):
        return 1

    def text(self):
        return self.buffer.getvalue().decode("utf-8", "replace")


def load_module(name):
    path = os.path.abspath(MODULE_PATH)
    assert os.path.isfile(path), f'"{path}" must exist'
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
    """A private copy of the module with a clean environment, unloaded at teardown."""
    saved = {key: value for key, value in os.environ.items() if key.startswith("WAN2GP_")}
    for key in list(os.environ):
        if key.startswith("WAN2GP_"):
            del os.environ[key]
    module = load_module(f"wgp_console_stamps_test_{id(object())}")
    try:
        yield module
    finally:
        module.disable_console_stamps()
        sys.stdout, sys.stderr = (module._STATE["stdout"] or sys.stdout,
                                  module._STATE["stderr"] or sys.stderr)
        os.environ.update(saved)
        for key in list(os.environ):
            if key.startswith("WAN2GP_") and key not in saved:
                del os.environ[key]
        sys.modules.pop(module.__name__, None)


# The feature is off by default in every mode, so the stamping tests ask for it.
ON = ["--log-timestamp=on"]


def installed(module, console, argv, **kwargs):
    """Install with ``sys.stdout`` / ``sys.stderr`` pointing at one fake console."""
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = console
    sys.stderr = console
    try:
        return module.install_console_stamps(argv, **kwargs)
    finally:
        sys.stdout, sys.stderr = (module._STATE["stdout"] or saved_stdout,
                                  module._STATE["stderr"] or saved_stderr)


def bar_write(step, total, suffix=""):
    """A tqdm line as a terminal receives it: padded to the console width, ending on '\\r'."""
    bar = f"  5%|          | {step}/{total} [00:04<1:17:00,  4.06s/it]"
    return (bar + suffix).ljust(80) + "\r"


def test_one_stamp_at_the_start_of_every_line_the_console_gets(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("model loaded, 40 GB\n")
    stamper.feed("first\nsecond\nthird\n")
    lines = console.text().split("\n")
    show(console.text())
    assert lines[-1] == ""
    for index, word in enumerate(("model loaded, 40 GB", "first", "second", "third")):
        assert STAMP_RE.match(lines[index]), lines[index]
        assert lines[index].split(" | ")[1] == word
        assert lines[index].count(" | ") == 1


def test_a_line_split_across_writes_loses_nothing_and_gets_one_stamp(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    for piece in ("H3 denoising: 100%| ", "100/100", " [00:04", "<00:00, 22.87it/s]\n"):
        stamper.feed(piece)
    show(console.text())
    stamp, rest = console.text().split(" | ", 1)
    assert STAMP_RE.match(stamp + " | ")
    assert rest == "H3 denoising: 100%| 100/100 [00:04<00:00, 22.87it/s]\n"


def test_a_chunk_carrying_several_lines_stamps_every_one_of_them(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("a\nb\nc\n")
    show(console.text())
    lines = [one for one in console.text().split("\n") if one]
    assert console.text().count(" | ") == 3
    assert [one.split(" | ")[1] for one in lines] == ["a", "b", "c"]
    for one in lines:
        assert STAMP_RE.match(one), one


def test_bar_redraws_stay_bytes_for_bytes_under_one_stamp(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    redraws = "".join(bar_write(step, 20) for step in range(1, 21))
    stamper.feed("H3 denoising: ")
    stamper.feed(redraws)
    stamper.feed(" 100%| 20/20 [00:20<00:00,  1.00s/it]\n")
    show(console.text())
    payload = "H3 denoising: " + redraws + " 100%| 20/20 [00:20<00:00,  1.00s/it]\n"
    text = console.text()
    assert stamp_marks(text) == 1
    assert text.endswith(payload)
    assert len(text) == len(payload) + len(STAMP_MARK_RE.match(text).group(0))


def test_a_bar_wider_than_the_console_is_left_alone(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    line = ("H3 denoising: 100%| 100/100 [00:04<00:00, 22.87it/s] step 100/100 "
            + "x" * 80)
    stamper.feed(line)
    stamper.flush()
    show(console.text())
    assert stamp_marks(console.text()) == 1
    assert console.text().endswith(line)


def test_the_queue_status_line_keeps_overwriting_itself(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    for task in ("1/3 gen_a", "2/3 gen_b", "3/3 gen_c"):
        stamper.feed(("Processing task " + task).ljust(60) + "\r")
    show(console.text())
    assert console.text().count(" | ") == 1
    assert console.text().endswith("Processing task 3/3 gen_c".ljust(60) + "\r")


def test_colour_codes_reach_the_terminal_untouched(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("\x1b[31mERROR\x1b[0m in step 12\n\x1b[32mdone\x1b[0m\n")
    show(console.text())
    text = console.text()
    assert text.count("\x1b[31m") == 1 and text.count("\x1b[32m") == 1
    assert text.count("\x1b[0m") == 2
    assert text.count(" | ") == 2
    assert "\x1b[31mERROR\x1b[0m in step 12" in text


def test_empty_writes_are_ignored_and_flush_leaves_the_text_alone(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("")
    assert console.text() == ""
    stamper.feed("done\n")
    before = console.text()
    stamper.flush()
    assert console.text() == before
    assert console.flushes >= 1


def test_the_plain_run_keeps_the_legacy_output_in_every_mode(module):
    """No flag, no env var: nothing is stamped and the streams are not even wrapped."""
    modes = (["--process", "queue.zip"], ["--server-name", "0.0.0.0"], [], ["--ask-deepy"],
             ["--deepy-server", "--listen"], ["--mcp"], ["--test"], ["--dry-run"], ["-h"],
             ["--refresh-catalog"])
    saved_out, saved_err = sys.stdout, sys.stderr
    for argv in modes:
        try:
            assert module.install_console_stamps(argv) is False, argv
            assert module.console_stamps_enabled() is False, argv
            sys.stdout.write("plain output\n")
            sys.stdout.flush()
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
        assert sys.stdout is saved_out, argv
        assert sys.stderr is saved_err, argv


def test_the_flag_turns_the_stamps_on_whatever_the_mode_is():
    """A stamped mode needs a fresh module per mode: installing is a once-per-process act."""
    for index, argv in enumerate((["--process", "queue.zip"], ["--server-name", "0.0.0.0"], [],
                                  ["--ask-deepy"])):
        stamp_module = load_module(f"wgp_console_stamps_on_probe_{index}")
        console = FakeStream()
        sys.stdout, sys.stderr = console, console
        try:
            assert stamp_module.install_console_stamps(ON + list(argv)) is True, argv
            sys.stdout.write("stamped line\n")
            sys.stdout.flush()
        finally:
            stamp_module.disable_console_stamps()
            sys.stdout, sys.stderr = (stamp_module._STATE["stdout"] or sys.__stdout__,
                                      stamp_module._STATE["stderr"] or sys.__stderr__)
            sys.modules.pop(stamp_module.__name__, None)
        show(console.text())
        assert STAMP_RE.match(console.text()), argv
        assert "stamped line" in console.text(), argv


def test_mcp_stdio_is_never_installed(module):
    assert module._uses_stdio_protocol(["--mcp"]) is True
    assert module._uses_stdio_protocol(["--mcp", "--mcp-transport=stdio"]) is True
    assert module._uses_stdio_protocol(["--mcp", "--mcp-transport=sse"]) is False
    assert module._uses_stdio_protocol(["--process", "queue.zip"]) is False
    console = FakeStream()
    assert installed(module, console, ON + ["--mcp"]) is False
    assert module.console_stamps_enabled() is False
    assert console.text() == ""
    assert sys.stdout is not console


def test_install_stamps_from_the_very_first_line(module):
    console = FakeStream()
    assert installed(module, console, ON + ["--process", "queue.zip"]) is True
    wrapped = module._STATE["streams"][0]
    wrapped.write("Model loaded: Wan 2.2 14B\n")
    wrapped.write("H3 denoising: 100%| 100/100 [00:04<00:00, 22.87it/s]\n")
    show(console.text())
    lines = [one for one in console.text().split("\n") if one]
    assert "WanGP console log start:" in lines[0]
    for line in lines:
        assert STAMP_RE.match(line), line
    assert " | Model loaded: Wan 2.2 14B" in console.text()
    assert " | H3 denoising: 100%| 100/100 [00:04<00:00, 22.87it/s]" in console.text()


def test_install_is_idempotent(module):
    console = FakeStream()
    assert installed(module, console, ON + ["--process", "queue.zip"]) is True
    header = console.text()
    assert installed(module, console, ON + ["--process", "queue.zip"]) is True
    assert installed(module, console, ON + ["--log-timestamp=on"]) is True
    assert console.text() == header


def test_nothing_is_written_and_no_folder_is_created(module):
    with tempfile.TemporaryDirectory() as tmp_path:
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            console = FakeStream()
            installed(module, console, ON + ["--process", "queue.zip"])
            module._STATE["streams"][0].write("line\n")
            assert os.listdir(tmp_path) == []
            assert module.console_stamp_state() == {"enabled": True,
                                                    "format": module.resolve_format("full"),
                                                    "separator": " | "}
            assert stamp_marks(console.text()) == 2
        finally:
            os.chdir(cwd)


def test_the_flags_are_read_from_the_raw_command_line(module):
    console = FakeStream()
    installed(module, console, ON + ["--process", "queue.zip", "--log-timestamp-format=clock",
                               "--log-timestamp-sep=dash"])
    assert module._STATE["format"] == "%H:%M:%S"
    assert module._STATE["separator_resolved"] == " - "
    module._STATE["streams"][0].write("model loaded\n")
    show(console.text())
    assert re.search(r"^\d\d:\d\d:\d\d - model loaded$", console.text(), re.MULTILINE)


def test_timestamp_off_restores_the_streams_untouched(module):
    stdout, stderr = sys.stdout, sys.stderr
    console = FakeStream()
    sys.stdout = console
    sys.stderr = console
    try:
        assert module.install_console_stamps(["--process", "--log-timestamp=off"]) is False
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    assert console.text() == ""
    assert module.console_stamps_enabled() is False
    assert sys.stdout is stdout and sys.stderr is stderr


def test_the_env_var_is_the_default_for_every_mode(module):
    os.environ["WAN2GP_LOG_TIMESTAMP"] = "1"
    console = FakeStream()
    installed(module, console, ["--ask-deepy"])
    assert module.console_stamps_enabled() is True
    module._STATE["streams"][0].write("hello from the chat\n")
    show(console.text())
    assert re.search(r"^\d{4}-\d{2}-\d{2} \d\d:\d\d:\d\d \| hello from the chat$",
                     console.text(), re.MULTILINE)


def test_the_parsed_options_are_applied_by_configure(module):
    args = types.SimpleNamespace(log_timestamp_format="ms", log_timestamp_sep="none")
    console = FakeStream()
    installed(module, console, ON + ["--process", "queue.zip"])
    module._STATE["streams"][0].write("before\n")
    assert module.configure_console_stamps(args, ["--process", "queue.zip"]) == ""
    module._STATE["streams"][0].write("configured\n")
    module._STATE["streams"][0].write("after\n")
    show(console.text())
    assert " | before" in console.text()
    assert re.search(r"^\d\d:\d\d:\d\d\.\d{3} configured$", console.text(), re.MULTILINE)
    assert re.search(r"^\d\d:\d\d:\d\d\.\d{3} after$", console.text(), re.MULTILINE)


def test_the_stamp_flag_takes_the_on_off_forms(module):
    switch = lambda argv: module._argv_on_off(argv, "--log-timestamp")
    assert switch(["--process", "--log-timestamp"]) is True            # the flag on its own
    assert switch(["--process", "--log-timestamp=on"]) is True
    assert switch(["--process", "--log-timestamp", "on"]) is True
    assert switch(["--process", "--log-timestamp", "off"]) is False
    assert switch(["--process", "--log-timestamp=off"]) is False
    assert switch(["--process", "queue.zip"]) is None                  # left out: the default (off)
    console = FakeStream()
    installed(module, console, ["--process", "--log-timestamp"])
    assert module.console_stamp_state()["enabled"] is True
    module._STATE["streams"][0].write("bare flag line\n")
    show(console.text())
    assert re.search(r"^\d{4}-\d{2}-\d{2} \d\d:\d\d:\d\d \| bare flag line$", console.text(), re.MULTILINE)


def test_the_stream_interface_the_app_relies_on_survives(module):
    console = FakeStream()
    installed(module, console, ON + ["--process", "queue.zip"])
    wrapped = module._STATE["streams"][0]
    assert isinstance(wrapped, module.ConsoleStampStream)
    assert wrapped.encoding == "utf-8"
    assert wrapped.errors == "strict"
    assert wrapped.writable() is True
    assert wrapped.isatty() is False
    assert wrapped.fileno() == 1
    assert wrapped.buffer is console.buffer
    assert wrapped.write(b"bytes in\n") == len(b"bytes in\n")
    assert "bytes in" in console.text()


def test_wgp_py_installs_the_stamps_from_raw_argv(tmp_path):
    """wgp.py must install the feature before the heavy imports, with the console-only defaults."""
    script = tmp_path / "probe_console_stamps_bootstrap.py"
    script.write_text(textwrap.dedent(f"""
        import io, os, sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        os.chdir({str(PROJECT_ROOT)!r})
        source = open({os.path.join(PROJECT_ROOT, 'wgp.py')!r}, encoding="utf-8").read()
        head = source.split("import warnings")[0]
        assert "install_console_stamps(sys.argv)" in head, "the feature must install from raw argv"
        assert "install_console_stamps(__file__" not in head, "no anchor path any more"
        import importlib.util
        spec = importlib.util.spec_from_file_location("probe_console_stamps", {MODULE_PATH!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        console = io.StringIO()
        sys.stdout = console
        assert module.install_console_stamps(["--process", "x.zip", "--log-timestamp=on"]) is True
        sys.stdout = sys.__stdout__
        print(console.getvalue(), end="")
    """), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    output = (result.stdout or b"").decode("utf-8", "replace")
    assert result.returncode == 0, output[-3000:]
    assert "WanGP console log start:" in output
    assert not os.path.exists(os.path.join(str(tmp_path), "crash"))


def test_an_empty_line_is_echoed_without_a_stamp(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("first\n\nsecond\n")
    show(console.text())
    lines = console.text().split("\n")
    assert lines[1] == "", "an empty line must reach the console empty"
    assert lines[0].split(" | ")[1] == "first"
    assert lines[2].split(" | ")[1] == "second"
    assert stamp_marks(console.text()) == 2


def test_the_bare_print_of_the_shutdown_leaves_no_stamp_behind(module):
    """The bare print() WanGP prints on the way out is what used to leave a lone stamp."""
    console = FakeStream()
    stream = module.ConsoleStampStream(console)
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    module._STATE["stdout"], module._STATE["stderr"] = saved_stdout, saved_stderr
    sys.stdout = sys.stderr = stream
    try:
        print()
        print("Autosave: Queue is empty, nothing to save.")
        print()
        stream.flush()
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
    show(console.text())
    lines = console.text().split("\n")
    assert lines[0] == "" and lines[3] == ""
    assert STAMP_RE.match(lines[1]) and lines[1].count(" | ") == 1
    assert lines[1].split(" | ")[1] == "Autosave: Queue is empty, nothing to save."
    assert len(lines) == 4 and stamp_marks(console.text()) == 1


def test_a_line_carrying_only_line_controls_carries_no_stamp(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("\n")
    stamper.feed("text\n")
    stamper.feed("\n")
    stamper.feed("\r")
    stamper.feed("\x08")
    stamper.flush()
    show(console.text())
    assert STAMP_MARK_RE.sub("", console.text()) == "\ntext\n\n\r\x08"
    assert console.text().startswith("\n"), "the empty opening line is echoed as it is"
    assert stamp_marks(console.text()) == 1


def test_a_split_line_is_stamped_at_its_start_and_only_once(module):
    """A bar opening with padding and a line split across writes: one stamp, every byte kept."""
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("  ")                                  # padding first: it echoes, the stamp waits
    stamper.feed("H3 denoising: 100%| ")
    stamper.feed("100/100 [00:04")
    stamper.feed("<00:00, 22.87it/s]")
    stamper.feed("\r" + " " * 60)                       # the padded tail of the same line
    stamper.feed("\n")
    stamper.feed("\n")                                  # the empty line that follows
    show(console.text())
    payload = ("  " + "H3 denoising: 100%| 100/100 [00:04<00:00, 22.87it/s]"
               + "\r" + " " * 60 + "\n" + "\n")
    text = console.text()
    assert text.startswith("  "), "padding written before any text echoes, the stamp follows"
    assert STAMP_MARK_RE.sub("", text) == payload
    assert stamp_marks(text) == 1


def test_nothing_but_the_stamps_is_added_to_every_line_the_console_gets(module):
    """Byte-identity over the shapes WanGP really prints: bars, colour codes, empty lines."""
    console = FakeStream()
    stamper = module._LineStamper(console)
    payload = (
        "model loaded, 40 GB\n"
        + "".join(bar_write(step, 20) for step in (5, 10, 15, 20))
        + " 100%| 20/20 [00:20<00:00,  1.00s/it]\n"
        + "\n"
        + "\x1b[31mERROR\x1b[0m in step 12\n"
        + "\x1b[32mdone\x1b[0m\n"
        + "Autosave: Queue is empty, nothing to save.\n"
        + "\x08\n"
        + "Session duration 2h 25m 28s\n"
        + "Queue completed: 2/2 tasks in 12m 22s."
    )
    stamper.feed(payload)
    show(console.text())
    text = console.text()
    assert STAMP_MARK_RE.sub("", text) == payload       # only the stamps were inserted
    assert stamp_marks(text) == 7


def test_the_interrupt_writes_one_newline_and_runs_the_handler_it_found(module):
    console = FakeStream()
    console._tty = True
    previous = signal.getsignal(signal.SIGINT)
    ran = []

    def prior_handler(signum, frame):
        ran.append((signum, "prior"))

    writes = []
    saved = (module.os.getpid, module.os.getppid, module.os.isatty, module.os.write)
    signal.signal(signal.SIGINT, prior_handler)
    try:
        assert installed(module, console, ON, quiet=True) is True
        handler = signal.getsignal(signal.SIGINT)
        assert getattr(handler, "__wgp_interrupt_chain__", False), "chain onto it, never replace it"
        module.os.getpid = lambda: 4321
        module.os.getppid = lambda: 4320
        module.os.isatty = lambda fd: True
        module.os.write = lambda fd, data: writes.append((fd, data))
        handler(signal.SIGINT, None)
        assert writes == [(1, b"\n")], "one newline on the first console descriptor that is a tty"
        assert ran == [(signal.SIGINT, "prior")], "the handler in force must still run"
        assert console.text() == "", "the newline goes to the terminal, not to the stamped stream"
    finally:
        module.os.getpid, module.os.getppid, module.os.isatty, module.os.write = saved
        signal.signal(signal.SIGINT, previous)


def test_the_default_interrupt_still_raises_after_the_newline(module):
    console = FakeStream()
    console._tty = True
    previous = signal.getsignal(signal.SIGINT)
    writes = []
    saved = (module.os.getpid, module.os.getppid, module.os.isatty, module.os.write)
    try:
        # The stamps chain onto whatever handler is in force, and another test file
        # may already have replaced it in this process, so the handler this check is
        # about - Python's own - is put in force here rather than assumed.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        assert installed(module, console, ON, quiet=True) is True
        handler = signal.getsignal(signal.SIGINT)
        module.os.getpid = lambda: 4321
        module.os.getppid = lambda: 4320
        module.os.isatty = lambda fd: True
        module.os.write = lambda fd, data: writes.append((fd, data))
        try:
            handler(signal.SIGINT, None)
            raise AssertionError("the default handler must keep raising KeyboardInterrupt")
        except KeyboardInterrupt:
            pass
        assert writes[0] == (1, b"\n"), "the newline comes before the interruption runs on"
        assert all(data == b"\n" for _fd, data in writes)
    finally:
        module.os.getpid, module.os.getppid, module.os.isatty, module.os.write = saved
        signal.signal(signal.SIGINT, previous)


def test_the_interrupt_newline_is_never_armed_for_a_pipe(module):
    console = FakeStream()                                # FakeStream.isatty() is False
    previous = signal.getsignal(signal.SIGINT)
    try:
        assert installed(module, console, ON) is True
        assert module._STATE["interrupt_handler"] is None
        assert signal.getsignal(signal.SIGINT) is previous
    finally:
        signal.signal(signal.SIGINT, previous)


def test_the_interrupt_is_armed_once_only(module):
    console = FakeStream()
    console._tty = True
    previous = signal.getsignal(signal.SIGINT)
    try:
        assert installed(module, console, ON, quiet=True) is True
        armed = module._STATE["interrupt_handler"]
        assert armed is signal.getsignal(signal.SIGINT)
        assert module._arm_interrupt_newline() is None, "a second install must not chain twice"
        assert module._STATE["interrupt_handler"] is armed
        assert module.console_stamp_state()["enabled"] is True
    finally:
        signal.signal(signal.SIGINT, previous)
def test_a_traceback_line_is_the_last_line_its_stream_stamps(module):
    """A crash dump is for reading: its marker line carries the last stamp, all under it clean."""
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("Loading model handlers\n")
    stamper.feed("Traceback (most recent call last):\n")
    stamper.feed('  File "/home/calimet/software/graphics/Wan2GP/models/ltx2/ltx2_handler.py", '
                 "line 424, in _migrate_loras\n")
    stamper.feed("    lora_dir = Path(resolve_lora_dir(spec['lora_dir'], lora_root, wgp.args.lora_config))\n")
    stamper.feed("    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    stamper.feed("FileExistsError: [Errno 17] File exists: 'loras'\n")
    show(console.text())
    text = console.text()
    lines = text.split("\n")
    assert stamp_marks(text) == 2
    assert STAMP_RE.match(lines[0]) and lines[0].endswith("Loading model handlers")
    assert STAMP_RE.match(lines[1]) and lines[1].endswith("Traceback (most recent call last):")
    assert lines[2].startswith('  File "/home/calimet')
    assert lines[3].startswith("    lora_dir") and lines[4].startswith("    ^^^^")
    assert lines[5].startswith("FileExistsError: [Errno 17]")
    assert text.endswith("FileExistsError: [Errno 17] File exists: 'loras'\n")


def test_a_whole_traceback_printed_as_one_write_keeps_its_dump_clean(module):
    """Python's chained dump arrives in one write: the opener is stamped, the dump is not."""
    console = FakeStream()
    stamper = module._LineStamper(console)
    dump = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.12/pathlib.py", line 1313, in mkdir\n'
        "    os.mkdir(self, mode)\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'loras/ltx2'\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "/home/calimet/software/graphics/Wan2GP/wgp.py", line 2956, in <module>\n'
        "FileExistsError: [Errno 17] File exists: 'loras'\n"
    )
    stamper.feed(dump)
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 1
    assert text == STAMP_MARK_RE.match(text).group(0) + dump


def test_the_prompt_after_a_crash_reaches_the_console_unstamped(module):
    console = FakeStream()
    module._STATE["stdout"], module._STATE["stderr"] = sys.stdout, sys.stderr
    stamp_stream = module.ConsoleStampStream(console)
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stamp_stream
    try:
        print("Generation finished in 12m 22s (queue task 1)")
        print("Traceback (most recent call last):")
        print('  File "/home/calimet/software/graphics/Wan2GP/wgp.py", line 2956, in <module>')
        print("FileExistsError: [Errno 17] File exists: 'loras'")
        print("calimet@roy:~/software/graphics/Wan2GP$", end="")
        stamp_stream.flush()
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
    show(console.text())
    text = console.text()
    lines = text.split("\n")
    assert stamp_marks(text) == 2
    assert STAMP_RE.match(lines[0]) and lines[0].endswith("Generation finished in 12m 22s (queue task 1)")
    assert STAMP_RE.match(lines[1]) and lines[1].endswith("Traceback (most recent call last):")
    assert not STAMP_RE.match(lines[2]) and not STAMP_RE.match(lines[3])
    assert lines[4] == "calimet@roy:~/software/graphics/Wan2GP$"


def test_the_marker_takes_the_stamps_off_only_from_its_own_line_start(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed('the word Traceback (most recent call last) quoted inside a line\n')
    stamper.feed("Traceback (most recent call last):\n")
    stamper.feed('  File "wgp.py", line 1, in <module>\n')
    stamper.feed('ValueError: a quoted \"Traceback (most recent call last):\" comes later\n')
    stamper.feed("the line after the dump\n")
    show(console.text())
    assert stamp_marks(console.text()) == 2


def test_a_bar_close_write_of_cursor_codes_is_echoed_unstamped(module):
    """What a progress bar writes when it closes at exit: cursor moves and a carriage return, nothing to draw.

    The stamp written on that write landed on a line the terminal showed as empty, and it stayed open:
    the shell prompt arrived after it, as `2026-10-22 18:51:17 | user@host:~$`.
    """
    console = FakeStream()
    stamper = module._LineStamper(console)
    closing = "\n\n" + "\x1b[A" * 2 + "\r" + "\x1b[A" * 2
    stamper.feed(closing)
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 0, "codes the terminal consumes carry no text: nothing to stamp"
    assert text == closing, "the bar's own bytes reach the console untouched"
    assert stamper.close_open_line() is False, "nothing was stamped, so there is no open line to close"


def test_an_erase_a_cursor_show_and_a_link_url_line_stays_empty(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    invisible = "\x1b[K\x1b[?25h\x1b(B\x1b[m\x1b]8;;https://wangp.ai\x07\x1b]8;;\x07"
    stamper.feed(invisible + "\n")
    show(console.text())
    assert console.text() == invisible + "\n"
    assert stamp_marks(console.text()) == 0


def test_the_control_bytes_an_interrupt_leaves_behind_are_not_stamped(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("\x03\x04\x7f\x9bK\x1b")                      # Ctrl+C, Ctrl+D, DEL, an 8-bit CSI, a lone ESC
    show(console.text())
    assert stamp_marks(console.text()) == 0


def test_a_bar_line_that_draws_glyphs_is_stamped_once(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("\x1b[?25lEncoding Text Prompt: 0%|          | 0/48 [00:00<?, ?it/s]\r")
    stamper.feed("\x1b[?25h")
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 1, "the bar draws, so its line is stamped once"
    assert STAMP_RE.match(text)
    assert text.endswith("\x1b[?25h")
    assert stamper.close_open_line() is True, "the line is open under a stamp: it gets its newline"
    assert console.text().endswith("\n")


def test_a_hyperlink_label_still_carries_its_text(module):
    """The URL of an OSC 8 link is invisible, the label is drawn (tqdm issue 1856)."""
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("\x1b]8;;https://github.com/tqdm/tqdm/issues/1856\x07docs\x1b]8;;\x07\n")
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 1
    assert text == STAMP_MARK_RE.match(text).group(0) + "\x1b]8;;https://github.com/tqdm/tqdm/issues/1856\x07docs\x1b]8;;\x07\n"


def test_the_open_line_is_closed_once_and_only_under_a_stamp(module):
    console = FakeStream()
    stream = module.ConsoleStampStream(console)
    stream.write("Autosave: Queue is empty, nothing to save.\n")
    assert stream.close_open_line() is False, "a closed line needs nothing"
    stream.write("Denoising 50%\r")
    assert stream.close_open_line() is True
    assert stream.close_open_line() is False, "the line is closed: the newline is written once"
    show(console.text())
    lines = console.text().split("\n")
    assert STAMP_RE.match(lines[0]) and lines[0].endswith("Autosave: Queue is empty, nothing to save.")
    assert STAMP_RE.match(lines[1]) and lines[1].endswith("Denoising 50%\r")
    assert lines[2] == "", "the shell prompt that follows starts on a line of its own"


def test_the_console_of_a_stamped_run_ends_with_no_stamp_on_its_last_line(module):
    """The shutdown as the app writes it: the last prints, then a bar's closing write."""
    console = FakeStream()
    console._tty = True
    installed(module, console, ON, quiet=True)
    stream = module._STATE["streams"][0]
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stream
    try:
        print("Generation finished in 12m 22s (queue task 1)")
        print("Autosave: Queue is empty, nothing to save.")
        stream.write("Denoising 50%|████████     | 25/50 [00:13<00:26]\r\x1b[K")
        stream.write("\n\n" + "\x1b[A" * 2 + "\r" + "\x1b[A" * 2)     # the bar's closing write
        stream.flush()
        closed = module.close_open_lines()
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 3
    assert closed is False, "the bar closed its own line: there is nothing left to close"
    last_line = text.rsplit("\n", 1)[-1]
    assert stamp_marks(last_line) == 0, "no stamp rides on the line the shell prompt starts on"
    assert ("\n\n" + "\x1b[A" * 2 + "\r" + "\x1b[A" * 2) in text, "the bar's own bytes are untouched"


def test_a_run_that_ends_mid_bar_closes_its_stamped_line(module):
    console = FakeStream()
    console._tty = True
    installed(module, console, ON, quiet=True)
    stream = module._STATE["streams"][0]
    stream.write("Loading LTX2.3 model\r")
    stream.flush()
    assert module.close_open_lines() is True
    assert module.close_open_lines() is False, "the newline is written once"
    show(console.text())
    text = console.text()
    assert text == STAMP_MARK_RE.match(text).group(0) + "Loading LTX2.3 model\r\n"


def test_a_pipe_run_gets_no_extra_newline_at_exit(module):
    console = FakeStream()                      # not a terminal
    installed(module, console, ON, quiet=True)
    stream = module._STATE["streams"][0]
    stream.write("Denoising 50%\r")
    stream.flush()
    assert module.close_open_lines() is False
    text = console.text()
    assert text.endswith("Denoising 50%\r")
    assert text == STAMP_MARK_RE.match(text).group(0) + "Denoising 50%\r"


WELCOME_YELLOW = "\x1b[1m\x1b[93m"
BANNER_MAGENTA = "\x1b[1m\x1b[95m"
COLOUR_OFF = "\x1b[0m"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
WANGP_WELCOME = " WanGP v18.888 - Copyright (c) 2024-2026 DeepBeepMeep "
MMGP_WELCOME = " Memory Management for the GPU Poor (mmgp 4.4.1) - Copyright (c) 2024-2026 DeepBeepMeep "
MMGP_PROFILE = " Applying MMGP Memory Profile 4 - LowRAM_LowVRAM (v4.4.1) "


def banner(message, colour=BANNER_MAGENTA):
    """A header as the app writes it: colours, the message centred in a field of 100 stars."""
    return f"{colour}{message:*^100}{COLOUR_OFF}"


def test_the_starred_header_line_is_echoed_without_a_stamp(module):
    """The mmgp welcome line is a rule of stars: its bytes stay, the log lines keep their stamps."""
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("[CUDA] Compute Cache: /home/calimet/software/graphics/Wan2GP/ComputeCache\n")
    stamper.feed(banner(MMGP_WELCOME) + "\n")
    stamper.feed("[RAM] MMGP RAM Allocator: the RAM of freed CPU tensors goes back to the system\n")
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 2, "the header carries no stamp: the two log lines do"
    lines = text.split("\n")
    assert STAMP_RE.match(lines[0]) and lines[0].endswith("ComputeCache")
    assert not STAMP_RE.match(lines[1]), lines[1]
    assert lines[1] == banner(MMGP_WELCOME), "the header's own bytes reach the console untouched"
    assert len(ANSI_RE.sub("", lines[1])) == 100, "the rule keeps the 100 columns it was built with"
    assert STAMP_RE.match(lines[2]) and lines[2].endswith("goes back to the system")


def test_every_starred_rule_stays_unstamped(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    rules = [banner(WANGP_WELCOME, WELCOME_YELLOW), banner(MMGP_WELCOME), banner(MMGP_PROFILE),
             "*" * 100, "   ****** Memory Management ******", "***** section *****"]
    for one in rules:
        stamper.feed(one + "\n")
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 0, "a rule of stars is decoration: nothing is stamped"
    assert text == "".join(one + "\n" for one in rules), "the rules reach the console byte for byte"


def test_a_header_printed_with_print_keeps_its_line_unstamped(module):
    """print writes the text and then the newline: the line is a header at its very start."""
    console = FakeStream()
    stream = module.ConsoleStampStream(console)
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stream
    try:
        print(banner(MMGP_WELCOME))
        print("[VRAM] MMGP Optimized VRAM Allocator, spilling into RAM when VRAM is full")
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
    stream.flush()
    show(console.text())
    lines = console.text().split("\n")
    assert stamp_marks(console.text()) == 1
    assert lines[0] == banner(MMGP_WELCOME)
    assert STAMP_RE.match(lines[1]) and lines[1].endswith("when VRAM is full")


def test_the_single_star_bullets_of_the_server_keep_their_stamp(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("* Running on local URL:  http://172.17.0.6:7860\n")
    stamper.feed("* To create a public link, set `share=True` in `launch()`.\n")
    show(console.text())
    lines = console.text().split("\n")
    for index, text in enumerate(("http://172.17.0.6:7860", "share=True")):
        assert STAMP_RE.match(lines[index]), lines[index]
        assert lines[index].split(" | ", 1)[1].startswith("* ")
        assert text in lines[index]


def test_the_star_run_that_opens_a_line_decides_it(module):
    """Three stars open a rule: a one-star bullet, a two-star bold and stars inside a line read on."""
    console = FakeStream()
    stamper = module._LineStamper(console)
    cases = (("**Mothersuperior** for the YuE2 adapter\n", True),
             ("**\n", True),
             ("***\n", False),
             ("MMGP profile ****** applied\n", True),
             ("* To create a public link, set `share=True` in `launch()`.\n", True),
             ("***** section *****\n", False))
    for text, _ in cases:
        stamper.feed(text)
    show(console.text())
    lines = console.text().split("\n")
    for (text, stamped), line in zip(cases, lines):
        assert bool(STAMP_RE.match(line)) is stamped, f"{text!r} -> {line!r}"


def test_a_header_under_a_traceback_stays_unstamped(module):
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("Traceback (most recent call last):\n")
    stamper.feed(banner(MMGP_WELCOME) + "\n")
    stamper.feed("ValueError: the memory profile is not a number\n")
    show(console.text())
    text = console.text()
    lines = text.split("\n")
    assert stamp_marks(text) == 1, "the marker line carries the last stamp the stream writes"
    assert STAMP_RE.match(lines[0]) and lines[0].endswith("Traceback (most recent call last):")
    assert lines[1] == banner(MMGP_WELCOME)
    assert lines[2] == "ValueError: the memory profile is not a number"


def test_a_run_that_ends_on_a_header_adds_no_newline(module):
    console = FakeStream()
    console._tty = True
    installed(module, console, ON, quiet=True)
    stream = module._STATE["streams"][0]
    stream.write(banner(MMGP_WELCOME) + "\r")
    stream.flush()
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 0, "no stamp on a header, so nothing is left open on it"
    assert module.close_open_lines() is False, "there is no open stamp to close: the prompt needs no newline"
    assert text == banner(MMGP_WELCOME) + "\r"


def test_a_header_whose_first_write_is_two_stars_is_stamped(module):
    """The stamp decision is taken at the line start: a write of two stars is text at that point.

    No app header is written that way: the banners arrive in one write (``tqdm.write``) and ``print``
    writes the whole text before the newline, so a header never splits in front of its stars.
    """
    console = FakeStream()
    stamper = module._LineStamper(console)
    stamper.feed("**")
    stamper.feed("**** Memory Management for the GPU Poor ****\n")
    show(console.text())
    text = console.text()
    assert stamp_marks(text) == 1
    assert text == STAMP_MARK_RE.match(text).group(0) + "****** Memory Management for the GPU Poor ****\n"

