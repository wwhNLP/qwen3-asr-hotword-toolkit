"""Regression test for prefix handling in the standalone legacy RAG."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("pypinyin"), "pypinyin is required for hotword retrieval")
class LegacyRagTest(unittest.TestCase):
    def test_finds_match_after_window_prefix(self):
        from hotword.rag_fast import FastRAG

        rag = FastRAG(threshold=0.85)
        self.assertEqual(rag._python_distance_simple([9, 1, 2], [1, 2]), (0.0, 3))
        self.assertEqual(rag._score_candidates([9, 1, 2], [("term", [1, 2], [1])]),
                         [("term", 1.0, 3)])
        self.assertEqual(rag._score_candidates([9, 1, 3], [("term", [1, 2], [1])]), [])


if __name__ == "__main__":
    unittest.main()
