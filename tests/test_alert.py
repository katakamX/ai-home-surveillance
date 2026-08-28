"""Alert tests. No external service is involved: handlers write to an
in-memory stream, and failures are simulated with mocks.
"""

import io
import unittest
from unittest.mock import MagicMock

from src.alerts.alert import Alert, AlertDispatcher, AlertHandler, ConsoleAlertHandler

ALERT_LOGGER = "src.alerts.alert"


def make_alert(event_id="7", **overrides):
    """An Alert with sensible defaults, for tests that only care about one field."""
    values = dict(
        event_id=event_id,
        event_type="person_entered_zone",
        timestamp=100.0,
        zone="front_door",
        track_id=1,
        message="Person entered front_door",
    )
    values.update(overrides)
    return Alert(**values)


def failing_handler(error=RuntimeError("handler down")):
    """A handler mock whose send() always raises."""
    handler = MagicMock()
    handler.send.side_effect = error
    return handler


class TestAlert(unittest.TestCase):
    def test_carries_every_field(self):
        alert = make_alert()

        self.assertEqual(alert.event_id, "7")
        self.assertEqual(alert.event_type, "person_entered_zone")
        self.assertEqual(alert.timestamp, 100.0)
        self.assertEqual(alert.zone, "front_door")
        self.assertEqual(alert.track_id, 1)
        self.assertEqual(alert.message, "Person entered front_door")

    def test_alerts_with_the_same_values_are_equal(self):
        self.assertEqual(make_alert(), make_alert())


class TestAlertHandlerProtocol(unittest.TestCase):
    def test_console_handler_satisfies_the_protocol(self):
        self.assertIsInstance(ConsoleAlertHandler(io.StringIO()), AlertHandler)

    def test_dispatcher_satisfies_the_protocol_too(self):
        self.assertIsInstance(AlertDispatcher(), AlertHandler)

    def test_an_object_without_send_does_not_satisfy_the_protocol(self):
        self.assertNotIsInstance(object(), AlertHandler)


class TestConsoleAlertHandler(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        self.handler = ConsoleAlertHandler(self.stream)

    def test_writes_one_line_per_alert(self):
        self.handler.send(make_alert())
        self.handler.send(make_alert(event_id="8"))

        self.assertEqual(len(self.stream.getvalue().strip().splitlines()), 2)

    def test_line_mentions_the_important_fields(self):
        self.handler.send(make_alert())

        line = self.stream.getvalue()
        self.assertIn("person_entered_zone", line)
        self.assertIn("front_door", line)
        self.assertIn("Person entered front_door", line)
        self.assertIn("7", line)

    def test_defaults_to_stdout_when_no_stream_is_given(self):
        import sys

        self.assertIs(ConsoleAlertHandler().stream, sys.stdout)


class TestAlertDispatcher(unittest.TestCase):
    def test_sends_the_alert_to_every_handler(self):
        first, second = MagicMock(), MagicMock()
        alert = make_alert()

        AlertDispatcher([first, second]).send(alert)

        first.send.assert_called_once_with(alert)
        second.send.assert_called_once_with(alert)

    def test_returns_the_number_of_successful_deliveries(self):
        dispatcher = AlertDispatcher([MagicMock(), MagicMock()])

        self.assertEqual(dispatcher.send(make_alert()), 2)

    def test_no_handlers_is_safe(self):
        self.assertEqual(AlertDispatcher().send(make_alert()), 0)

    def test_accepts_handlers_added_later(self):
        dispatcher = AlertDispatcher()
        handler = MagicMock()

        dispatcher.add_handler(handler)
        dispatcher.send(make_alert())

        handler.send.assert_called_once()

    def test_does_not_share_the_caller_s_handler_list(self):
        handlers = []
        dispatcher = AlertDispatcher(handlers)

        dispatcher.add_handler(MagicMock())

        self.assertEqual(handlers, [])

    def test_works_end_to_end_with_a_console_handler(self):
        stream = io.StringIO()
        dispatcher = AlertDispatcher([ConsoleAlertHandler(stream)])

        delivered = dispatcher.send(make_alert())

        self.assertEqual(delivered, 1)
        self.assertIn("front_door", stream.getvalue())


class TestHandlerFailuresAreIsolated(unittest.TestCase):
    """A broken handler is logged and skipped; nothing propagates to the caller."""

    def test_a_failing_handler_does_not_raise(self):
        dispatcher = AlertDispatcher([failing_handler()])

        with self.assertLogs(ALERT_LOGGER, level="ERROR"):
            dispatcher.send(make_alert())  # should not raise

    def test_a_failing_handler_does_not_stop_the_others(self):
        good_first, good_last = MagicMock(), MagicMock()
        dispatcher = AlertDispatcher([good_first, failing_handler(), good_last])

        with self.assertLogs(ALERT_LOGGER, level="ERROR"):
            delivered = dispatcher.send(make_alert())

        good_first.send.assert_called_once()
        good_last.send.assert_called_once()
        self.assertEqual(delivered, 2)

    def test_the_failure_is_logged_with_the_event_id(self):
        dispatcher = AlertDispatcher([failing_handler()])

        with self.assertLogs(ALERT_LOGGER, level="ERROR") as logs:
            dispatcher.send(make_alert(event_id="42"))

        self.assertIn("42", logs.output[0])

    def test_every_handler_failing_reports_zero_deliveries(self):
        dispatcher = AlertDispatcher([failing_handler(), failing_handler()])

        with self.assertLogs(ALERT_LOGGER, level="ERROR"):
            self.assertEqual(dispatcher.send(make_alert()), 0)

    def test_a_later_alert_still_reaches_a_previously_failing_handler(self):
        handler = MagicMock()
        handler.send.side_effect = [RuntimeError("blip"), None]
        dispatcher = AlertDispatcher([handler])

        with self.assertLogs(ALERT_LOGGER, level="ERROR"):
            dispatcher.send(make_alert())
        self.assertEqual(dispatcher.send(make_alert(event_id="8")), 1)


if __name__ == "__main__":
    unittest.main()
