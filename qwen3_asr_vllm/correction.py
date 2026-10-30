"""Conservative literal correction using aliases explicitly supplied by users."""

import re
from collections import defaultdict

_DIGIT_VARIANTS = {"0": "0零", "1": "1一", "2": "2二", "3": "3三", "4": "4四",
                   "5": "5五", "6": "6六", "7": "7七", "8": "8八", "9": "9九"}


def normalize_mixed_identifiers(text, selected):
    """Normalize exact alphanumeric identifiers containing spoken digits."""
    count = 0
    for target in sorted(set(selected), key=len, reverse=True):
        if not re.fullmatch(r"[A-Za-z]{2,}[0-9]+", target):
            continue
        parts = [f"[{_DIGIT_VARIANTS[c]}]" if c in _DIGIT_VARIANTS else re.escape(c) for c in target]
        pattern = re.compile(r"(?<![A-Za-z0-9])" + r"\s*".join(parts) + r"(?![A-Za-z0-9])", re.IGNORECASE)

        def replace(match):
            nonlocal count
            if match.group() == target:
                return match.group()
            count += 1
            return target

        text = pattern.sub(replace, text)
    return text, count


def correct_explicit_aliases(text, corrector, selected):
    """Replace unambiguous explicit aliases in the final transcript.

    Phonetic scores are deliberately not used as replacement confidence. A
    blacklist anywhere in the transcript suppresses that target conservatively.
    Automatically generated pronunciation aliases are retrieval-only.
    """
    aliases = defaultdict(set)
    for target, target_aliases in getattr(corrector, "explicit_aliases", {}).items():
        for alias in target_aliases:
            if len(alias) >= 2 and alias != target:
                aliases[alias].add(target)
    selected = set(selected)
    unambiguous = {}
    for alias, targets in aliases.items():
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        if alias != target and alias in getattr(corrector, "hotwords", {}):
            continue
        if target in selected and not any(word in text for word in getattr(corrector, "blacklists", {}).get(target, ())):
            unambiguous[alias] = target
    if not unambiguous:
        return text, 0

    alternatives = []
    for alias in sorted(unambiguous, key=lambda value: (-len(value), value)):
        pattern = re.escape(alias)
        if alias[0].isascii() and alias[0].isalnum():
            pattern = r"(?<![A-Za-z0-9])" + pattern
        if alias[-1].isascii() and alias[-1].isalnum():
            pattern += r"(?![A-Za-z0-9])"
        alternatives.append(pattern)
    pattern = re.compile("|".join(alternatives))
    # Do not rewrite an alias occurring inside an already-correct target.
    protected = [(match.start(), match.end()) for target in selected if target
                 for match in re.finditer(re.escape(target), text)]
    replacements = 0

    def replace(match):
        nonlocal replacements
        if any(start <= match.start() and match.end() <= end for start, end in protected):
            return match.group()
        replacements += 1
        return unambiguous[match.group()]

    return pattern.sub(replace, text), replacements
