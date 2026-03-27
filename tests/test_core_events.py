"""Tests for edgefinder/core/events.py."""

from edgefinder.core.events import EventBus


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda data: received.append(data))
        bus.publish("test.event", {"key": "value"})
        assert received == [{"key": "value"}]

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []
        bus.subscribe("test.event", lambda d: results.append("a"))
        bus.subscribe("test.event", lambda d: results.append("b"))
        bus.publish("test.event", None)
        assert results == ["a", "b"]

    def test_unsubscribe(self):
        bus = EventBus()
        results = []
        handler = lambda d: results.append("called")
        bus.subscribe("test.event", handler)
        bus.unsubscribe("test.event", handler)
        bus.publish("test.event", None)
        assert results == []

    def test_handler_exception_does_not_crash_others(self):
        bus = EventBus()
        results = []

        def bad_handler(d):
            raise ValueError("boom")

        bus.subscribe("test.event", bad_handler)
        bus.subscribe("test.event", lambda d: results.append("ok"))
        bus.publish("test.event", None)
        assert results == ["ok"]

    def test_publish_no_subscribers_is_noop(self):
        bus = EventBus()
        bus.publish("nonexistent.event", {"data": 1})  # should not raise

    def test_clear(self):
        bus = EventBus()
        results = []
        bus.subscribe("test.event", lambda d: results.append("called"))
        bus.clear()
        bus.publish("test.event", None)
        assert results == []

    def test_different_event_types_isolated(self):
        bus = EventBus()
        results_a = []
        results_b = []
        bus.subscribe("event.a", lambda d: results_a.append(d))
        bus.subscribe("event.b", lambda d: results_b.append(d))
        bus.publish("event.a", "hello")
        assert results_a == ["hello"]
        assert results_b == []
