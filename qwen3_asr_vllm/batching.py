"""Bounded, time-windowed batching for a single synchronous ASR engine."""

import asyncio
from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool


class BatchQueueFull(Exception):
    pass


@dataclass
class _Request:
    waveform: object
    options: dict
    future: asyncio.Future
    arrived_at: float


class DynamicBatcher:
    def __init__(self, engine, *, max_batch_size=8, max_wait_ms=20, max_queue_size=128):
        if max_batch_size < 1 or max_wait_ms < 0 or max_queue_size < 1:
            raise ValueError("Batch size and queue size must be positive; wait must be nonnegative.")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.task = None
        self.accepting = False
        self.batches = 0
        self.requests = 0
        self.last_batch_size = 0
        self.max_observed_batch_size = 0
        self.retry_splits = 0

    def start(self):
        if self.task is not None:
            raise RuntimeError("Batcher has already started.")
        self.accepting = True
        self.task = asyncio.create_task(self._run())

    async def close(self):
        if self.task is None:
            return
        self.accepting = False
        # The sentinel follows queued requests, so a normal shutdown drains them.
        await self.queue.put(None)
        await self.task
        self.task = None

    async def submit(self, waveform, options):
        if not self.accepting:
            raise RuntimeError("Batcher is not accepting requests.")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _Request(waveform, options, future, loop.time())
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise BatchQueueFull from exc
        return await future

    def info(self):
        return {"queued": self.queue.qsize(), "max_queue_size": self.queue.maxsize,
                "max_batch_size": self.max_batch_size, "max_wait_ms": self.max_wait * 1000,
                "batches": self.batches, "requests": self.requests,
                "last_batch_size": self.last_batch_size,
                "max_observed_batch_size": self.max_observed_batch_size,
                "retry_splits": self.retry_splits}

    async def _infer(self, active):
        active = [item for item in active if not item.future.done()]
        if not active:
            return
        try:
            results = await run_in_threadpool(
                self.engine.transcribe_batch, [item.waveform for item in active],
                request_options=[item.options for item in active],
            )
            if len(results) != len(active):
                raise RuntimeError("Engine returned an unexpected number of results.")
        except ValueError as exc:
            # Isolate an invalid request without retrying resource failures such
            # as GPU OOM. A bad prompt must not fail unrelated requests.
            if len(active) > 1:
                self.retry_splits += 1
                middle = len(active) // 2
                await self._infer(active[:middle])
                await self._infer(active[middle:])
            elif not active[0].future.done():
                active[0].future.set_exception(exc)
        except Exception as exc:
            for item in active:
                if not item.future.done():
                    item.future.set_exception(exc)
        else:
            for item, result in zip(active, results):
                if not item.future.done():
                    item.future.set_result(result)

    async def _run(self):
        loop = asyncio.get_running_loop()
        stopping = False
        while not stopping:
            first = await self.queue.get()
            if first is None:
                break
            batch = [first]
            deadline = first.arrived_at + self.max_wait
            while len(batch) < self.max_batch_size:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            active = [item for item in batch if not item.future.cancelled()]
            if not active:
                continue
            size = len(active)
            self.batches += 1
            self.requests += size
            self.last_batch_size = size
            self.max_observed_batch_size = max(self.max_observed_batch_size, size)
            await self._infer(active)
