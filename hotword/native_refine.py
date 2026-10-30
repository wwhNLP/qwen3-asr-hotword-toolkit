"""Native CPU refinement over precomputed phoneme substitution costs."""

import numpy as np
from numba import njit

from .algo_calc import _is_similar_phoneme, lcs_length


_MAX_COST_BYTES = 16 * 1024 * 1024


@njit(cache=True, nogil=True)
def match_costs(costs, word_starts, word_ends, threshold):
    """Preserve the Python matcher's pruning, boundary and tie semantics."""
    n, m = costs.shape
    results = []
    if n == 0 or m == 0:
        return results
    previous = np.full(m + 1, np.inf)
    current = np.full(m + 1, np.inf)
    previous_starts = np.zeros(m + 1, dtype=np.int64)
    current_starts = np.zeros(m + 1, dtype=np.int64)
    for j in range(m + 1):
        if j < m and word_starts[j]:
            previous[j] = 0.0
            previous_starts[j] = j

    for i in range(n):
        current[0] = np.inf
        current_starts[0] = 0
        row_min = np.inf
        for j in range(1, m + 1):
            match = previous[j - 1] + costs[i, j - 1]
            delete = previous[j] + 1.0
            insert = current[j - 1] + 1.0
            if match <= delete:
                if match <= insert:
                    distance, start = match, previous_starts[j - 1]
                else:
                    distance, start = insert, current_starts[j - 1]
            elif delete <= insert:
                distance, start = delete, previous_starts[j]
            else:
                distance, start = insert, current_starts[j - 1]
            current[j] = distance
            current_starts[j] = start
            if distance < row_min:
                row_min = distance
        if row_min > n * (1.0 - threshold) + 2:
            return results
        previous, current = current, previous
        previous_starts, current_starts = current_starts, previous_starts

    for j in range(1, m + 1):
        if not word_ends[j - 1] or previous[j] >= n * 0.8:
            continue
        score = 1.0 - previous[j] / n
        if score >= threshold:
            results.append((score, previous_starts[j], j))
    # Stable insertion sort: equal scores retain increasing end positions.
    for i in range(1, len(results)):
        item = results[i]
        j = i
        while j > 0 and results[j - 1][0] < item[0]:
            results[j] = results[j - 1]
            j -= 1
        results[j] = item
    return results


def substitution_cost(hotword, source):
    value, language, tone = hotword
    source_value, source_language = source
    if language != source_language:
        return 1.0
    if value == source_value:
        return 0.0
    if language == "zh":
        return 0.5 if tone or _is_similar_phoneme(value, source_value) else 1.0
    if language == "en":
        return 1.0 - lcs_length(value, source_value) / max(len(value), len(source_value))
    return 1.0


class NativeRefiner:
    """Vocabulary-scoped encoding; query costs never survive a request."""

    def __init__(self, hotwords):
        symbols = {}
        self.aliases = {}
        for target, aliases in hotwords.items():
            encoded = []
            for phonemes in aliases:
                ids = []
                for phoneme in phonemes:
                    key = (phoneme.value, phoneme.lang, phoneme.is_tone)
                    ids.append(symbols.setdefault(key, len(symbols)))
                encoded.append(np.asarray(ids, dtype=np.int64))
            self.aliases[target] = encoded
        self.symbols = list(symbols)
        # Compile/load the sole array signature while loading the vocabulary.
        match_costs(np.zeros((1, 1), dtype=np.float64),
                    np.ones(1, dtype=np.bool_), np.ones(1, dtype=np.bool_), 0.6)
        if not getattr(match_costs, "nopython_signatures", ()):
            raise ImportError("Numba JIT is disabled.")

    def prepare(self, input_info, targets):
        active = sorted({int(value) for target in targets
                         for alias in self.aliases[target] for value in alias})
        if len(active) * len(input_info) * 8 > _MAX_COST_BYTES:
            return None
        sources = list(dict.fromkeys((item[0], item[1]) for item in input_info))
        source_indices = {source: index for index, source in enumerate(sources)}
        columns = [source_indices[(item[0], item[1])] for item in input_info]
        unique_costs = np.empty((len(active), len(sources)), dtype=np.float64)
        rows = np.full(len(self.symbols), -1, dtype=np.int64)
        for row, symbol in enumerate(active):
            rows[symbol] = row
            for column, source in enumerate(sources):
                unique_costs[row, column] = substitution_cost(self.symbols[symbol], source)
        costs = unique_costs[:, columns]
        starts = np.asarray([item[2] for item in input_info], dtype=np.bool_)
        ends = np.asarray([item[3] for item in input_info], dtype=np.bool_)
        return PreparedRefinement(self.aliases, rows, costs, starts, ends)


class PreparedRefinement:
    def __init__(self, aliases, rows, costs, starts, ends):
        self.aliases, self.rows, self.costs = aliases, rows, costs
        self.starts, self.ends = starts, ends

    def search(self, target, alias, start, end, threshold):
        encoded = self.aliases[target][alias]
        if len(encoded) * (end - start) * 8 > _MAX_COST_BYTES:
            return None
        rows = self.rows[encoded]
        costs = np.ascontiguousarray(self.costs[rows, start:end])
        return match_costs(costs, self.starts[start:end], self.ends[start:end], float(threshold))
