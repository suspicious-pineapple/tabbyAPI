"""Tests for generator recovery after a generation error (PR #461)."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("exllamav3")

from backends.exllamav3.model import ExllamaV3Container  # noqa: E402
from common.health import HealthManager  # noqa: E402


class FakeJob:
    def __init__(self, cancelled=False):
        self.cancelled = cancelled
        self.cancel_calls = 0

    async def cancel(self):
        self.cancel_calls += 1
        self.cancelled = True


def make_container(generator):
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.generator = generator
    container.recreations = 0

    async def create_generator():
        container.recreations += 1

    container.create_generator = create_generator
    return container


async def recover(container, job, ex=RuntimeError("boom")):
    await container._recover_from_generation_error(ex, job)
    # Let a scheduled create_generator() run
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def clean_health():
    HealthManager.issues.clear()
    yield
    HealthManager.issues.clear()


def test_latched_detection():
    assert make_container(None)._generator_latched()
    assert make_container(SimpleNamespace())._generator_latched()
    assert make_container(SimpleNamespace(error=RuntimeError("x")))._generator_latched()
    assert not make_container(SimpleNamespace(error=None))._generator_latched()


def test_contained_error_keeps_generator_and_cancels_only_this_job():
    container = make_container(SimpleNamespace(error=None))
    job = FakeJob()

    asyncio.run(recover(container, job))

    assert container.recreations == 0
    assert job.cancel_calls == 1
    assert HealthManager.issues.maxlen == 100 and len(HealthManager.issues) == 0


def test_contained_error_does_not_cancel_twice():
    container = make_container(SimpleNamespace(error=None))
    job = FakeJob(cancelled=True)

    asyncio.run(recover(container, job))

    assert job.cancel_calls == 0
    assert container.recreations == 0


def test_latched_error_recreates_and_records_health_event():
    container = make_container(SimpleNamespace(error=RuntimeError("engine died")))
    job = FakeJob()

    asyncio.run(recover(container, job, ex=RuntimeError("engine died")))

    assert container.recreations == 1
    # Recreation cancels every job itself; the consumer must not touch the dead generator
    assert job.cancel_calls == 0
    assert [issue.description for issue in HealthManager.issues] == ["RuntimeError: engine died"]


def test_wrapper_without_latch_attribute_recreates():
    container = make_container(SimpleNamespace())
    job = FakeJob()

    asyncio.run(recover(container, job))

    assert container.recreations == 1
    assert len(HealthManager.issues) == 1
