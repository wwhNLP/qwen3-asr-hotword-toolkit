import asyncio
import threading
import unittest

from qwen3_asr_vllm.batching import BatchQueueFull, DynamicBatcher


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def transcribe_batch(self, waveforms, *, request_options):
        self.calls.append((waveforms, request_options))
        self.started.set()
        if not self.release.wait(5):
            raise RuntimeError("test timed out")
        if self.fail:
            raise ValueError("inference failed")
        return [{"audio": waveform, "hotwords": option["hotwords"]}
                for waveform, option in zip(waveforms, request_options)]


class DynamicBatchingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.batcher = DynamicBatcher(self.engine, max_batch_size=4, max_wait_ms=30, max_queue_size=8)
        self.batcher.start()

    async def asyncTearDown(self):
        self.engine.release.set()
        await self.batcher.close()

    async def test_batch_preserves_order_and_request_vocabulary(self):
        tasks = [self.batcher.submit(i, {"hotwords": words, "max_new_tokens": 8 + i})
                 for i, words in enumerate([None, ["甲"], [], ["乙"]])]
        result = await asyncio.gather(*tasks)
        self.assertEqual(result, [{"audio": 0, "hotwords": None}, {"audio": 1, "hotwords": ["甲"]},
                                  {"audio": 2, "hotwords": []}, {"audio": 3, "hotwords": ["乙"]}])
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual([option["max_new_tokens"] for option in self.engine.calls[0][1]], [8, 9, 10, 11])
        self.assertEqual(self.batcher.info()["max_observed_batch_size"], 4)

    async def test_timeout_flushes_partial_batch(self):
        result = await asyncio.wait_for(self.batcher.submit(1, {"hotwords": None}), timeout=1)
        self.assertEqual(result["audio"], 1)
        self.assertEqual(len(self.engine.calls[0][0]), 1)

    async def test_failure_reaches_all_requests_and_worker_survives(self):
        self.engine.fail = True
        results = await asyncio.gather(*(self.batcher.submit(i, {"hotwords": None}) for i in range(2)),
                                       return_exceptions=True)
        self.assertTrue(all(isinstance(result, ValueError) for result in results))
        self.engine.fail = False
        self.assertEqual((await self.batcher.submit(3, {"hotwords": None}))["audio"], 3)

    async def test_invalid_request_does_not_fail_other_requests(self):
        original = self.engine.transcribe_batch

        def infer(waveforms, *, request_options):
            if 2 in waveforms:
                raise ValueError("invalid request")
            return original(waveforms, request_options=request_options)

        self.engine.transcribe_batch = infer
        results = await asyncio.gather(*(self.batcher.submit(i, {"hotwords": None}) for i in range(4)),
                                       return_exceptions=True)
        self.assertEqual([results[i]["audio"] for i in (0, 1, 3)], [0, 1, 3])
        self.assertIsInstance(results[2], ValueError)
        self.assertGreater(self.batcher.info()["retry_splits"], 0)

    async def test_full_queue_and_cancelled_requests(self):
        await self.batcher.close()
        self.engine.release.clear()
        self.batcher = DynamicBatcher(self.engine, max_batch_size=1, max_wait_ms=0, max_queue_size=1)
        self.batcher.start()
        first = asyncio.create_task(self.batcher.submit(1, {"hotwords": None}))
        self.assertTrue(await asyncio.to_thread(self.engine.started.wait, 2))
        second = asyncio.create_task(self.batcher.submit(2, {"hotwords": None}))
        await asyncio.sleep(0)
        with self.assertRaises(BatchQueueFull):
            await self.batcher.submit(3, {"hotwords": None})
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        self.engine.release.set()
        self.assertEqual((await first)["audio"], 1)
        await self.batcher.close()
        self.assertEqual(len(self.engine.calls), 1)

    async def test_close_drains_queued_requests(self):
        tasks = [asyncio.create_task(self.batcher.submit(i, {"hotwords": None})) for i in range(3)]
        await asyncio.sleep(0)
        await self.batcher.close()
        self.assertEqual([result["audio"] for result in await asyncio.gather(*tasks)], [0, 1, 2])
