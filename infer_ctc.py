"""Qwen3-ASR 推理 (可切换推理模式)。

在 infer.py 单遍推理的基础上, 增加对 CTC 分支的利用, 通过 MODE 自由切换:

  MODE = "asr"      纯单遍 ASR (等价 infer.py)。可选把 HOTWORDS 手动拼进 system。
  MODE = "ctc"      仅跑 CTC 分支贪婪解码, 输出"粗转写"(快, 不过 LLM), 用于调试/测速。
  MODE = "ctc_rag"  两遍: CTC 贪婪出粗转写 -> PhonemeCorrector 音素 RAG 检索相关热词
                    -> 注入 system -> 主模型正式转写。

  "ctc"/"ctc_rag" 需要模型带 CTC 分支 (config.add_ctc_branch=True, 见 setup_ctc_ckpt.py)。
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', "6")
import warnings

import torch
from qwen_omni_utils import process_mm_info

# ============================ 配置 ============================
MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODE = os.environ.get("MODE", "ctc_rag")  # "asr" | "ctc" | "ctc_rag"

# 注入到 system 的热词前缀, 必须与训练时 qwen3_asr.py 的写法一致
HOTWORD_SYSTEM_PREFIX = "可能出现的热词："

# asr 模式手动注入的热词 (留空 = 纯转写, 不注入)
HOTWORDS = []

# ctc_rag 模式: 大热词库 (每行一个热词) + 检索参数 (默认用包内自带的 hotword/hot.txt)
HOT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hotword", "hot.txt")
RAG_THRESHOLD = 0.85        # 音素相似度阈值
MAX_CTX_HOTWORDS = 10       # 最多注入多少个检索到的热词

USE_AUDIO_IN_VIDEO = True

# 文件推理模式: 非空 -> 批量推理该 jsonl (每行 messages+audios[+uttid]) 并算 CER;
#                留空 -> 单条样本模式 (跑文件末尾的 wav_path)。也可用环境变量 TEST_JSON 覆盖。
TEST_JSON = os.environ.get("TEST_JSON", "")
# 结果输出 jsonl, 留空则自动放到 TEST_JSON 同目录 (加 _{MODE}_infer 后缀)。
OUT_JSON = os.environ.get("OUT_JSON", "")

# hotword 包与本文件同目录 (qwen3_asr_context/hotword)
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from qwen3_asr_model.register import register_qwen3_asr

register_qwen3_asr()
from qwen3_asr_model import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor  # noqa: E402


def _load_model_processor():
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype='auto',
        attn_implementation='flash_attention_2',
        device_map="auto",
    )
    processor = Qwen3ASRProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def _prepare_inputs(processor, wav_path, system="", use_audio_in_video=True):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "audio", "audio": wav_path}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True)
    return inputs


def generate_asr(model, processor, wav_path, hotwords=None, use_audio_in_video=True):
    """单遍主模型转写。hotwords 非空时拼进 system 做「模型内偏向」。"""
    system = HOTWORD_SYSTEM_PREFIX + "，".join(hotwords) if hotwords else ""
    inputs = _prepare_inputs(processor, wav_path, system=system,
                             use_audio_in_video=use_audio_in_video)
    inputs = inputs.to(model.device).to(model.dtype)
    thinker_result = model.generate(**inputs)
    response_text = processor.batch_decode(
        thinker_result.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response_text


@torch.no_grad()
def ctc_transcribe(model, processor, wav_path, vocab, use_audio_in_video=True):
    """CTC 分支贪婪解码, 返回粗转写 (不过 LLM)。仅支持单条音频。"""
    inputs = _prepare_inputs(processor, wav_path, system="",
                             use_audio_in_video=use_audio_in_video)
    inputs = inputs.to(model.device).to(model.dtype)
    thinker = model.thinker
    _, feats_list = thinker.get_audio_features(
        inputs["input_features"],
        feature_attention_mask=inputs.get("feature_attention_mask"),
        return_per_audio=True,
    )
    packed_logits, input_lengths = thinker._ctc_branch_packed(feats_list)
    # 单条音频: 取第一段
    length = int(input_lengths[0])
    frame_ids = packed_logits[:length].argmax(dim=-1).tolist()
    return vocab.ctc_greedy_decode(frame_ids)


@torch.no_grad()
def encode_audio(model, processor, wav_path, use_audio_in_video=True):
    """过一次音频编码器, 返回 (feats_list, audio_cat, base_inputs)。

    - feats_list : list[(T_i, D)]，供 ctc 分支复用；
    - audio_cat  : (sum_T, D)，供 LLM 走 inputs_embeds 复用 (跳过 forward 内部再过 encoder)；
    - base_inputs: 空 system 下的 processor 输出 (含 input_ids/attention_mask/feature_*)。

    一次 get_audio_features(return_per_audio=True) 同时拿到 cat 与 per-audio list,
    ctc/llm 两条路都基于同一份编码器输出, 不再重复过 encoder。
    """
    inputs = _prepare_inputs(processor, wav_path, system="",
                             use_audio_in_video=use_audio_in_video)
    inputs = inputs.to(model.device).to(model.dtype)
    audio_cat, feats_list = model.thinker.get_audio_features(
        inputs["input_features"],
        feature_attention_mask=inputs.get("feature_attention_mask"),
        return_per_audio=True,
    )
    return feats_list, audio_cat, inputs


def ctc_from_feats(model, feats_list, vocab):
    """基于已算好的 feats_list 做 CTC 贪婪解码 (复用 encoder)。返回粗转写。"""
    packed_logits, input_lengths = model.thinker._ctc_branch_packed(feats_list)
    length = int(input_lengths[0])  # 单条音频取第一段
    frame_ids = packed_logits[:length].argmax(dim=-1).tolist()
    return vocab.ctc_greedy_decode(frame_ids)


@torch.no_grad()
def asr_from_audio(model, processor, wav_path, audio_cat, hotwords=None,
                   base_inputs=None, use_audio_in_video=True):
    """复用已算好的 audio_cat 走 LLM 转写 (不再过 encoder)。

    做法: 把 audio_cat scatter 进音频占位得到 inputs_embeds, 直接喂 thinker.generate,
    从而跳过 forward 内部的 get_audio_features。hotwords 改变 system 时会重切 token
    (仅重算 mel 特征, 不过 encoder); 空 system 且给了 base_inputs 时直接复用。
    """
    thinker = model.thinker
    system = HOTWORD_SYSTEM_PREFIX + "，".join(hotwords) if hotwords else ""
    if not system and base_inputs is not None:
        inputs = base_inputs
    else:
        inputs = _prepare_inputs(processor, wav_path, system=system,
                                 use_audio_in_video=use_audio_in_video)
        inputs = inputs.to(model.device).to(model.dtype)
    input_ids = inputs["input_ids"]
    inputs_embeds = thinker.get_input_embeddings()(input_ids)
    audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        audio_mask, audio_cat.to(inputs_embeds.device, inputs_embeds.dtype))
    result = thinker.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        eos_token_id=[151643, 151645],
        num_beams=1,
        repetition_penalty=1.0,
        return_dict_in_generate=True,
    )
    # 传 inputs_embeds 时, generate 返回的 sequences 仅含新生成 token
    return processor.batch_decode(
        result.sequences, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def retrieve_hotwords(pc, rough_text, max_n=MAX_CTX_HOTWORDS):
    """用粗转写做音素 RAG 检索, 返回去重后的相关热词列表。"""
    if not rough_text:
        return []
    result = pc.correct(rough_text)
    ordered = []
    # matches: 命中并被替换的热词 (高置信); similars: top-k 相似候选
    for _origin, hotword, _score in list(result.matches) + list(result.similars):
        if hotword not in ordered:
            ordered.append(hotword)
    return ordered[:max_n]


def _load_corrector():
    from hotword import PhonemeCorrector
    from pathlib import Path

    pc = PhonemeCorrector(threshold=RAG_THRESHOLD)
    hot_content = Path(HOT_TXT).read_text(encoding="utf-8")
    n = pc.update_hotwords(hot_content)
    print(f"[hotword] loaded {n} hotwords from {HOT_TXT}")
    return pc


def _get_vocab():
    from utils.paraformer_vocab import get_vocab
    return get_vocab()


def transcribe(model, processor, wav_path, *, mode=MODE, vocab=None, pc=None,
               use_audio_in_video=True):
    """按 mode 分发推理, 全流程主模型 encoder 只过一次。

    返回 dict: {mode, rough?, hotwords?, text}。

    encoder 复用: encode_audio() 过一次 encoder 拿到 feats_list(给 ctc) 与
    audio_cat(给 llm), ctc/llm 两条路共享同一份特征, 不重复过 encoder。
    """
    # 主模型 encoder 只过这一次
    feats_list, audio_cat, base_inputs = encode_audio(
        model, processor, wav_path, use_audio_in_video=use_audio_in_video)

    if mode == "asr":
        text = asr_from_audio(model, processor, wav_path, audio_cat, hotwords=HOTWORDS,
                              base_inputs=base_inputs, use_audio_in_video=use_audio_in_video)
        return {"mode": mode, "text": text}

    if mode == "ctc":
        rough = ctc_from_feats(model, feats_list, vocab)
        return {"mode": mode, "text": rough}

    if mode == "ctc_rag":
        rough = ctc_from_feats(model, feats_list, vocab)
        hotwords = retrieve_hotwords(pc, rough)
        text = asr_from_audio(model, processor, wav_path, audio_cat, hotwords=hotwords,
                              base_inputs=base_inputs, use_audio_in_video=use_audio_in_video)
        return {"mode": mode, "rough": rough, "hotwords": hotwords, "text": text}

    raise ValueError(f"未知 MODE: {mode!r} (可选: asr / ctc / ctc_rag)")


# ============================ 文件推理 + CER ============================
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

# CER 归一化: 去空格与标点 (参考多带标点, 模型输出/参考统一清理后再比)
_PUNCT_RE = re.compile(
    r"[\s，。！？、；：,\.!\?;:\"'“”‘’（）()《》〈〉【】\[\]{}…—\-·・~`@#\$%\^&\*_\+=\|/\\<>]+"
)


def _norm_cer(s):
    return _PUNCT_RE.sub("", s or "")


def _edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _iter_jsonl(path):
    """逐行读取 jsonl, 产出 (wav, ref, uttid)。"""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = ""
            for msg in obj.get("messages", []):
                if msg.get("role") == "assistant":
                    ref = msg.get("content", "")
            wav = obj["audios"][0]
            uttid = obj.get("uttid", os.path.splitext(os.path.basename(wav))[0])
            yield wav, ref, uttid


def _default_out_json(test_json, mode):
    base, _ = os.path.splitext(test_json)
    return f"{base}_{mode}_infer.jsonl"


def run_file(model, processor, test_json, *, mode=MODE, vocab=None, pc=None,
             out_json="", use_audio_in_video=True):
    """批量推理 jsonl 文件, 逐条写结果并计算整体 CER。"""
    out_json = out_json or _default_out_json(test_json, mode)
    total_err = total_ref = n = 0
    t0 = time.time()
    print(f"[file] 推理 {test_json}\n[file] mode={mode}  "
          f"输出 -> {out_json}", flush=True)
    with open(out_json, "w", encoding="utf-8") as fout:
        for wav, ref, uttid in _iter_jsonl(test_json):
            try:
                out = transcribe(model, processor, wav, mode=mode, vocab=vocab, pc=pc,
                                 use_audio_in_video=use_audio_in_video)
                hyp = out.get("text", "")
            except Exception as e:  # noqa: BLE001
                out, hyp = {"mode": mode, "text": ""}, ""
                print(f"[ERR] {wav}: {e}", flush=True)
            r, h = _norm_cer(ref), _norm_cer(hyp)
            dist = _edit_distance(r, h)
            total_err += dist
            total_ref += len(r)
            n += 1
            rec = {"uttid": uttid, "wav": wav, "ref": ref, "hyp": hyp,
                   "err": dist, "ref_len": len(r)}
            if mode == "ctc_rag":
                rec["rough"] = out.get("rough", "")
                rec["hotwords"] = out.get("hotwords", [])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                fout.flush()
                print(f"[{n}] CER={total_err / max(total_ref, 1):.4f}  "
                      f"({n / (time.time() - t0):.2f} utt/s)", flush=True)
    cer = total_err / max(total_ref, 1)
    dt = time.time() - t0
    print("=" * 50)
    print(f"样本数        : {n}")
    print(f"总参考字符    : {total_ref}")
    print(f"总编辑距离    : {total_err}")
    print(f"CER           : {cer:.4f} ({cer * 100:.2f}%)")
    print(f"耗时          : {dt:.1f}s ({n / max(dt, 1e-9):.2f} utt/s)")
    print(f"逐条结果      : {out_json}", flush=True)
    return cer


if __name__ == "__main__":
    # ============================ 使用方法 ============================
# 1. 配置 (文件顶部):
#    - MODEL_PATH : 权重路径。ctc/ctc_rag 模式需指向"带 CTC 分支"的 checkpoint
#                   (config.add_ctc_branch=True, 由 setup_ctc_ckpt.py 生成/训练得到)。
#    - MODE       : "asr" | "ctc" | "ctc_rag" 三选一。
#    - HOT_TXT / RAG_THRESHOLD / MAX_CTX_HOTWORDS : ctc_rag 的热词库与检索参数。
#    - HOTWORDS   : 仅 asr 模式手动注入的热词 (留空 = 纯转写)。
#    - TEST_JSON  : 文件推理模式。设为 jsonl 路径则批量推理并算 CER;
#                   留空 = 单条样本模式 (跑下方 wav_path)。可用环境变量覆盖。
#    - OUT_JSON   : 结果输出路径 (留空自动放 TEST_JSON 同目录, 加 _{MODE}_infer 后缀)。
#
# 2. 运行:
#    cd qwen3_asr_context
#    python infer_ctc.py                       # 按顶部配置
#    TEST_JSON=/path/xxx.jsonl python infer_ctc.py   # 文件推理模式
#
# 3. 三种模式:
#    - asr     : 纯单遍主模型转写 (等价 infer.py)。
#    - ctc     : 只跑 CTC 贪婪解码出粗转写 (快, 不过 LLM), 常用于调试/测速。
#    - ctc_rag : CTC 粗转写 -> 音素 RAG 检索热词 -> 注入 system -> 主模型正式转写。
#
# 4. 作为库调用 (自定义 wav / 批量):
#    from infer_ctc import (_load_model_processor, _get_vocab,
#                           _load_corrector, transcribe)
#    model, processor = _load_model_processor()
#    vocab = _get_vocab()            # ctc / ctc_rag 需要
#    pc = _load_corrector()          # 仅 ctc_rag 需要
#    out = transcribe(model, processor, "/path/to/x.wav",
#                     mode="ctc_rag", vocab=vocab, pc=pc)
#    # 返回: {"mode", "text", 以及 ctc_rag 下的 "rough"/"hotwords"}
#    print(out["text"])
#
# 5. encoder 复用: transcribe 内部只过一次主模型 encoder,
#    ctc / llm 两条路共享同一份编码器输出 (encode_audio ->
#    feats_list 给 ctc, audio_cat 给 llm 走 inputs_embeds), 不重复过 encoder。
# ================================================================
    print(f"Run Qwen3 ASR (mode={MODE})...")
    model, processor = _load_model_processor()

    vocab = pc = None
    if MODE in ("ctc", "ctc_rag"):
        assert model.thinker.ctc_branch is not None, (
            "当前 checkpoint 未启用 CTC 分支 (config.add_ctc_branch=False)。"
            "请用 setup_ctc_ckpt.py 生成带 CTC 分支的模型, 或改 MODE='asr'。"
        )
        vocab = _get_vocab()
    if MODE == "ctc_rag":
        pc = _load_corrector()

    if TEST_JSON:
        # 文件推理模式: 批量推理 jsonl 并算 CER
        run_file(model, processor, TEST_JSON, mode=MODE, vocab=vocab, pc=pc,
                 out_json=OUT_JSON, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    else:
        # 单条样本模式 (默认)
        wav_path = ''
        out = transcribe(model, processor, wav_path, mode=MODE, vocab=vocab, pc=pc,
                         use_audio_in_video=USE_AUDIO_IN_VIDEO)

        if out.get("mode") == "ctc_rag":
            print(f"CTC 粗转写: {out['rough']}")
            print(f"检索热词  : {out['hotwords']}")
        print(f"转写结果  : {out['text']!r}")
