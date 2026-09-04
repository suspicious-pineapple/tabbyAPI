"""
Internal logging utility.
"""

import logging
import os
import requests
import json
from datetime import datetime, timezone
import re
from collections.abc import Mapping, Sequence, Set

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

_w = os.getenv("TABBY_LOG_CONSOLE_WIDTH")
_default_console_width = int(_w) if _w is not None and _w.isnumeric() else None
RICH_CONSOLE = Console(width=_default_console_width)
LOG_LEVEL = os.getenv("TABBY_LOG_LEVEL", "INFO")


def get_progress_bar():
    return Progress(console=RICH_CONSOLE)


def get_loading_progress_bar():
    """Gets a pre-made progress bar for loading tasks."""

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
        # Bars disappear once loading is done on a terminal; a plain log keeps
        # the final state as a single line instead
        transient=RICH_CONSOLE.is_terminal,
    )


_LEVEL_STYLES = {
    "TRACE": "dim blue",
    "DEBUG": "cyan",
    "INFO": "green",
    "SUCCESS": "bold green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold white on red",
}

# Width of the "LEVEL:" column, so messages line up across levels
_LEVEL_WIDTH = 9


def render_log_record(record: dict, message: str, console: Console) -> Text:
    """
    Lay out one log record with the timestamp and level on the left and the
    message wrapped to the console width on the right. Continuation lines are
    indented to the message column, so a long or multi-line message stays
    aligned instead of running back under the timestamp.
    """

    # The file log keeps the full date; the console only needs the time of day
    time = record["time"]
    level = record["level"].name

    out = Text(no_wrap=True)
    out.append(f"{time:%H:%M:%S}.{time.microsecond // 1000:03d} ", style="grey37")
    out.append(f"{level}:", style=_LEVEL_STYLES.get(level, "cyan"))
    out.append(" " * (_LEVEL_WIDTH - len(level)))

    indent = out.cell_len
    width = max(console.width - indent, 20)

    # Printing a plain string would run the console's highlighter (numbers,
    # paths, URLs); do the same for the Text we build here
    body = console.highlighter(Text(message.rstrip("\n")))
    lines = body.wrap(console, width)

    for index, line in enumerate(lines):
        if index:
            out.append("\n" + " " * indent)
        line.rstrip()
        out.append_text(line)

    return out


def _console_sink(message):
    """Loguru sink that prints records through the rich console."""

    RICH_CONSOLE.print(render_log_record(message.record, str(message), RICH_CONSOLE))


# Uvicorn log handler
# Uvicorn log portions inspired from https://github.com/encode/uvicorn/discussions/2027#discussioncomment-6432362
class UvicornLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(exception=record.exc_info).log(record.levelname, self.format(record).rstrip())


# Uvicorn config for logging. Passed into run when creating all loggers in server
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "uvicorn": {
            "class": f"{UvicornLoggingHandler.__module__}.{UvicornLoggingHandler.__qualname__}",  # noqa
        },
    },
    "root": {"handlers": ["uvicorn"], "propagate": False, "level": LOG_LEVEL},
    # Uvicorn's startup chatter duplicates what TabbyAPI already logs, so only
    # its warnings and errors get through. Access lines are gated separately by
    # the network.access_log option
    "loggers": {
        "uvicorn": {"level": "WARNING"},
        "uvicorn.error": {"level": "WARNING"},
        "uvicorn.access": {"level": LOG_LEVEL},
    },
}


def setup_logger():
    """Bootstrap the logger."""

    logger.remove()

    logger.add(
        _console_sink,
        level=LOG_LEVEL,
        format="{message}",
    )
    # Add file logging
    logger.add(
        "logs/{time}.log",
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="20 MB",  # Rotate file when it reaches 20MB
        retention="1 week",  # Keep logs for 1 week
        compression="zip",  # Compress rotated log
    )


"""
Extended logging via Seq.
"""

_DATA_URL_RE = re.compile(r"^(data:)([^;,]+)?(?:;[^,]*)?(;base64),(.*)$", re.DOTALL)


def _sanitize_for_logging(obj, head=1024, tail=1024):
    def truncate_string(s: str) -> str:
        if head + tail >= len(s):
            return s

        omitted = len(s) - head - tail
        return f"{s[:head]} [<- {omitted:,} chars truncated ->] {s[-tail:]}"

    def sanitize_string(s: str) -> str:
        m = _DATA_URL_RE.match(s)
        if m:
            prefix1, mime_type, prefix3, payload = m.groups()
            mime_type = mime_type or "application/octet-stream"
            prefix = f"{prefix1}{mime_type}{prefix3}"
            return f"{prefix} [<- {len(payload):,} chars truncated ->]"

        return truncate_string(s)

    def walk(value):
        if isinstance(value, str):
            return sanitize_string(value)

        if isinstance(value, Mapping):
            return {k: walk(v) for k, v in value.items()}

        if isinstance(value, tuple):
            return tuple(walk(v) for v in value)

        if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
            return {walk(v) for v in value}

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [walk(v) for v in value]

        return value

    return walk(obj)


class XLogger:
    def __init__(self):
        self.seqlog_url = None
        self.headers = {}
        self.enabled = False

    def _get_timestamp_now(self):
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def setup(self, seqlog_url: str = "http://localhost:5341", api_key: str | None = None):
        self.seqlog_url = seqlog_url.rstrip("/")
        self.headers = {"Content-Type": "application/vnd.serilog.clef"}
        if api_key:
            self.headers["X-Seq-ApiKey"] = api_key

        # Check if seqlog is reachable
        try:
            r = requests.post(
                self.seqlog_url + "/ingest/clef",
                data=(f'{{"@t":"{self._get_timestamp_now()}","@m":"TabbyAPI startup probe"}}\n'),
                headers=self.headers,
                timeout=2,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            reason = e.__class__.__name__
            logger.warning(f"Seq logging disabled: could not reach {self.seqlog_url} ({reason})")
            logger.debug(f"Seq probe error: {e}")
            return

        self.enabled = True
        logger.info(f"Enabled logging to seqlog instance at {self.seqlog_url}")

    def _commit(self, log_level: str, log_message: str, log_extra: dict):
        if not self.enabled:
            return

        try:
            if log_extra is None:
                log_extra = {}
            elif not isinstance(log_extra, dict):
                log_extra = {"extra": str(log_extra)}
            log_extra = _sanitize_for_logging(log_extra)
            event = {
                "@t": self._get_timestamp_now(),
                "@m": log_message,
                "@l": log_level,
                **log_extra,
            }
            try:
                data = json.dumps(event, default=str) + "\n"
            except Exception as e:
                data = "## Failed to serialize log data: " + str(e)
            r = requests.post(
                self.seqlog_url + "/ingest/clef",
                data=data,
                headers=self.headers,
                timeout=2,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to write log event to Seq, logging disabled: {e}")
            self.enabled = False

    def _compose(self, log_message, details):
        return (log_message + " " + details) if details else log_message

    def verbose(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        self._commit("Verbose", log_message, log_extra)

    def debug(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.debug(self._compose(log_message, details))
        self._commit("Debug", log_message, log_extra)

    def info(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.info(self._compose(log_message, details))
        self._commit("Information", log_message, log_extra)

    def warning(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.warning(self._compose(log_message, details))
        self._commit("Warning", log_message, log_extra)

    def error(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.error(self._compose(log_message, details))
        self._commit("Error", log_message, log_extra)


xlogger = XLogger()
