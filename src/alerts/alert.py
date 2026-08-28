"""Alerts: telling somebody that something happened.

An Alert is a small, plain description of one notable event. Handlers decide
what to do with it, and an AlertDispatcher fans one alert out to several
handlers without letting a broken handler take down the caller::

    dispatcher = AlertDispatcher([ConsoleAlertHandler()])
    dispatcher.send(Alert(
        event_id="7",
        event_type="person_entered_zone",
        timestamp=time.time(),
        zone="front_door",
        track_id=1,
        message="Person entered front_door",
    ))

Independent of OpenCV, YOLO, the camera and storage: this module only knows
about its own Alert type, so it can be unit tested with plain data and wired
into the pipeline later. Nothing here talks to any external service.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional, Protocol, TextIO, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """One notification about a single event."""

    event_id: str
    event_type: str
    timestamp: float
    zone: str
    track_id: int
    message: str


@runtime_checkable
class AlertHandler(Protocol):
    """Anything that can deliver an Alert somewhere.

    A handler is just an object with a send() method, so tests and future
    handlers (a file logger, a chat bot) need no base class.
    """

    def send(self, alert: Alert) -> None:
        """Deliver the alert. May raise: the dispatcher isolates failures."""


class ConsoleAlertHandler:
    """Prints alerts to a text stream. The development/testing handler.

    Writes to stdout by default; tests pass their own stream to capture the
    output instead of printing it.
    """

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def send(self, alert: Alert) -> None:
        self.stream.write(self.format(alert) + "\n")
        self.stream.flush()

    @staticmethod
    def format(alert: Alert) -> str:
        """One readable line describing the alert."""
        return (
            f"[ALERT] {alert.event_type} "
            f"zone={alert.zone} track={alert.track_id} "
            f"event={alert.event_id}: {alert.message}"
        )


class AlertDispatcher:
    """Sends one alert to every registered handler, isolating failures.

    A handler that raises is logged and skipped; the remaining handlers still
    receive the alert, and send() never raises. A dropped notification must
    never stop the surveillance loop.

    It has the same send() shape as a handler, so a dispatcher can itself be
    used wherever a handler is expected.
    """

    def __init__(self, handlers: Optional[list[AlertHandler]] = None) -> None:
        self.handlers: list[AlertHandler] = list(handlers) if handlers else []

    def add_handler(self, handler: AlertHandler) -> None:
        """Register one more handler to receive future alerts."""
        self.handlers.append(handler)

    def send(self, alert: Alert) -> int:
        """Deliver the alert to every handler and return how many succeeded."""
        delivered = 0
        for handler in self.handlers:
            try:
                handler.send(alert)
            except Exception:  # a broken handler must not affect the others
                logger.exception(
                    "Alert handler %s failed for event %s",
                    type(handler).__name__,
                    alert.event_id,
                )
            else:
                delivered += 1

        return delivered
