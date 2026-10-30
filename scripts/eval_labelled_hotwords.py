#!/usr/bin/env python3
"""Compare the same labelled utterances with and without hotword retrieval."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3_asr_vllm.runtime import VLLMContextASR, read_audio
from qwen3_asr_vllm.runtime import select_hotwords
from hotword import PhonemeCorrector
from utils.asr_metrics import _PUNCT_RE, cer_counts, transcript_text


def normalized(text):
    return _PUNCT_RE.sub('', transcript_text(text).casefold())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/labels.csv')
    parser.add_argument('--audio-dir', type=Path, default=ROOT / 'data/音频文件')
    parser.add_argument('--encoding', default='gb18030')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-hotwords', type=int, default=100)
    parser.add_argument('--rag-refiner', choices=('python', 'numba', 'auto'), default='auto')
    parser.add_argument('--kv-cache-mib', type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_hotwords < 0:
        parser.error('batch size must be positive and hotword limit nonnegative')
    with args.labels.open(encoding=args.encoding, newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'file_name', 'transcript', 'hotwords'}
    if not rows or not required.issubset(rows[0]):
        parser.error(f'CSV must contain {sorted(required)} and at least one record')
    words = [[word.strip() for word in row['hotwords'].split(';') if word.strip()] for row in rows]
    shared = list(dict.fromkeys(word for sample in words for word in sample))
    # Add two genuine domain terms that do not occur in this sample's labels.
    # These deliberately irrelevant words probe small-request fallback risk.
    distracted = []
    for i, gold in enumerate(words):
        candidates = shared[(i + 3) % len(shared):] + shared[:(i + 3) % len(shared)]
        extras = [term for term in candidates if term not in gold
                  and normalized(term) not in normalized(rows[i]['transcript'])][:2]
        distracted.append(gold + extras)
    audio = [read_audio(args.audio_dir / row['file_name']) for row in rows]
    engine = VLLMContextASR(args.model, mode='ctc_rag', max_ctx_hotwords=args.max_hotwords,
                            rag_refiner=args.rag_refiner, max_new_tokens=128, max_model_len=1024,
                            max_num_seqs=args.batch_size, enforce_eager=True,
                            gpu_memory_utilization=0.08, kv_cache_memory_bytes=args.kv_cache_mib * 1024**2)
    cases = [('asr_off', 'asr', 'off'), ('asr_prompt', 'asr', 'request'),
             ('ctc_rag_off', 'ctc_rag', 'off'), ('ctc_rag_request', 'ctc_rag', 'request'),
             ('ctc_rag_request_distractors', 'ctc_rag', 'distractors'),
             ('ctc_rag_shared', 'ctc_rag', 'shared'), ('asr_second_pass', 'asr', 'second_pass')]
    report = {'normalization': 'project cer_counts: punctuation/space removed, Unicode casefold; numbers unchanged',
              'count': len(rows), 'shared_vocabulary': shared, 'cases': {}}
    for name, mode, vocabulary in cases:
        engine.mode = mode
        results = []
        start = perf_counter()
        second_pass_words = None
        if vocabulary == 'second_pass':
            # Experiment only: retrieval from the first decoder pass adds a
            # second decoder pass, and the first pass time is not included.
            corrector = PhonemeCorrector(refiner=args.rag_refiner)
            corrector.update_hotword_entries(shared)
            first_texts = [item['text'] for item in report['cases']['asr_off']['details']]
            second_pass_words = [select_hotwords(item, args.max_hotwords) for item in
                                 corrector.correct_batch(first_texts, k=args.max_hotwords)]
        for offset in range(0, len(audio), args.batch_size):
            samples = audio[offset:offset + args.batch_size]
            options = [{'hotwords': [] if vocabulary == 'off' else (
                words[i] if vocabulary == 'request' else distracted[i] if vocabulary == 'distractors'
                else second_pass_words[i] if vocabulary == 'second_pass' else shared)}
                       for i in range(offset, offset + len(samples))]
            results.extend(engine.transcribe_batch(samples, request_options=options))
        elapsed = perf_counter() - start
        details = []
        for row, gold, result in zip(rows, words, results):
            err, total = cer_counts(row['transcript'].casefold(), result['text'].casefold())
            selected = {word.casefold() for word in result.get('hotwords', [])}
            retrieved = {word.casefold() for word in (result.get('retrieved_hotwords', [])
                                                       if vocabulary != 'second_pass' else result.get('hotwords', []))}
            hyp = normalized(result['text'])
            details.append({'file': row['file_name'], 'reference': row['transcript'], 'gold_hotwords': gold,
                            'text': result['text'], 'rough': result.get('rough'),
                            'selected': result.get('hotwords', []),
                            'retrieved': result.get('retrieved_hotwords', []) if vocabulary != 'second_pass' else result.get('hotwords', []),
                            'hotword_stats': result.get('hotword_stats'),
                            'selected_gold': [w for w in gold if w.casefold() in selected],
                            'retrieved_gold': [w for w in gold if w.casefold() in retrieved],
                            'output_gold': [w for w in gold if normalized(w) in hyp],
                            'errors': err, 'reference_chars': total, 'truncated': result.get('truncated', False)})
        errors = sum(d['errors'] for d in details)
        ref_chars = sum(d['reference_chars'] for d in details)
        gold_total = sum(len(g) for g in words)
        case = {'errors': errors, 'reference_chars': ref_chars, 'cer': errors / ref_chars if ref_chars else None,
                'gold_hotwords': gold_total, 'gold_selected': sum(len(d['selected_gold']) for d in details),
                'gold_retrieved': sum(len(d['retrieved_gold']) for d in details),
                'gold_output': sum(len(d['output_gold']) for d in details),
                'elapsed_seconds': round(elapsed, 3), 'details': details}
        report['cases'][name] = case
        print(name, json.dumps({k: v for k, v in case.items() if k != 'details'}, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
