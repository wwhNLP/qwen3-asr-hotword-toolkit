# coding=utf-8
"""SeACo-Paraformer 字表 (vocab8404) 工具, 供 CTC 分支使用。

tokens.json 来自 modelscope 模型
``iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch``,
共 8404 个 token:
  - idx 0   : ``<blank>``  (CTC blank)
  - idx 1   : ``<s>``
  - idx 2   : ``</s>``
  - idx 8403: ``<unk>``
  - 其余为中文字符 / 英文 BPE 片段 (形如 ``and@@``)。

CTC 直接复用该字表: ``odim = 8404``, ``blank_id = 0``。

文本 → token 序列 **完全对齐 FunASR ``CharTokenizer.text2tokens`` + ``seg_tokenize``**
(见 https://github.com/modelscope/FunASR/blob/main/funasr/tokenizer/char_tokenizer.py):
  1. 按空格切词 ``line.strip().split(" ")``;
  2. 每个词 ``word.lower()``;
  3. 整词命中 ``seg_dict`` → 用其 BPE 片段 (如 ``hello`` -> ``hel@@ lo``);
  4. 否则若以中日韩汉字 / 字母 / 数字开头 → 逐字符查 ``seg_dict`` (汉字自映射), 未登录字符 → ``<unk>``;
  5. 否则整词 → ``<unk>``。
seg_dict 来自同一模型仓库的 ``seg_dict`` 文件 (``paraformer_seg_dict.txt``), 为必需资源:
缺失时构造 ``ParaformerVocab`` 会直接断言失败。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List

_TOKENS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paraformer_tokens.json")
_SEG_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paraformer_seg_dict.txt")

BLANK = "<blank>"
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"

# FunASR seg_tokenize 的字符判定: 以 汉字 / 英文字母 / 数字 开头才逐字符切分。
_SEG_PATTERN = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")


class ParaformerVocab:
    """字符级词表: char <-> id 双向映射 + CTC 贪婪解码。"""

    def __init__(self, tokens_path: str = _TOKENS_PATH, seg_dict_path: str = _SEG_DICT_PATH):
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens: List[str] = json.load(f)
        self.id2tok = {i: t for i, t in enumerate(self.tokens)}
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.blank_id = self.tok2id.get(BLANK, 0)
        self.unk_id = self.tok2id.get(UNK, len(self.tokens) - 1)
        # 解码时需跳过的非文本特殊符
        self._special_ids = {
            self.tok2id[t] for t in (BLANK, SOS, EOS, UNK) if t in self.tok2id
        }
        self.seg_dict = self._load_seg_dict(seg_dict_path)

    @staticmethod
    def _load_seg_dict(seg_dict_path: str) -> Dict[str, List[str]]:
        """加载 FunASR seg_dict: ``key \\t tok1 tok2 ...``, value 拆成 token 列表。"""
        assert os.path.isfile(seg_dict_path), (
            f"seg_dict 文件不存在: {seg_dict_path} (需从 SeACo-Paraformer 模型仓库下载 seg_dict, "
            "供 FunASR CharTokenizer 分词使用)"
        )
        seg: Dict[str, List[str]] = {}
        with open(seg_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if not parts or not parts[0]:
                    continue
                key = parts[0]
                value = parts[1] if len(parts) >= 2 else parts[0]
                toks = value.split()
                seg[key] = toks if toks else [key]
        return seg

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @staticmethod
    def _is_punct(ch: str) -> bool:
        """是否为标点 (Unicode 类别 P*/S*, 覆盖中英文标点及部分符号)。"""
        cat = unicodedata.category(ch)
        return cat[0] in ("P", "S")

    def text2tokens(self, text: str) -> List[str]:
        """文本 → token 串, 对齐 FunASR ``CharTokenizer.text2tokens`` + ``seg_tokenize``。

        - 按空格切词, 逐词 ``lower()``;
        - 整词命中 seg_dict → BPE 片段; 否则若以汉字/字母/数字开头 → 逐字符查 seg_dict
          (汉字自映射, 未登录 → ``<unk>``); 否则整词 → ``<unk>``。
        """
        out: List[str] = []
        for word in text.strip().split(" "):
            if not word:
                continue
            word = word.lower()
            if word in self.seg_dict:
                out.extend(self.seg_dict[word])
            elif _SEG_PATTERN.match(word):
                for ch in word:
                    if ch in self.seg_dict:
                        out.extend(self.seg_dict[ch])
                    else:
                        out.append(UNK)
            else:
                out.append(UNK)
        return out

    def encode_text(self, text: str, strip_punct: bool = False) -> List[int]:
        """文本 → id 序列, 按 FunASR seg_tokenize 切 token 后查表 (未登录 token → <unk>)。

        strip_punct=True 时先去掉标点符号 (CTC 训练目标不需要标点), 保留空白分词。
        """
        if strip_punct:
            text = "".join(
                " " if ch.isspace() else ("" if self._is_punct(ch) else ch)
                for ch in text
            )
        return [self.tok2id.get(t, self.unk_id) for t in self.text2tokens(text)]

    def _ids_to_tokens(self, ids: List[int], skip_special: bool = True) -> List[str]:
        """id 序列 → token 串 (保留 ``@@`` 续接符, 交由 postprocess 处理)。"""
        toks: List[str] = []
        for i in ids:
            i = int(i)
            if skip_special and i in self._special_ids:
                continue
            tok = self.id2tok.get(i)
            if tok is not None:
                toks.append(tok)
        return toks

    def decode_ids(self, ids: List[int], skip_special: bool = True) -> str:
        """id 序列 → 字符串, 对齐 FunASR ``sentence_postprocess``。

        - 纯中文: 逐字拼接, 无空格;
        - 纯英文: BPE ``@@`` 片段拼词, 词间补空格;
        - 中英混合: 中文无空格, 英文成词后补空格;
        - 连续单字母 (如 s h e) 由 ``abbr_dispose`` 合并为大写缩写 (SHE)。
        """
        return self._sentence_postprocess(self._ids_to_tokens(ids, skip_special))

    def ctc_greedy_decode(self, frame_ids: List[int]) -> str:
        """CTC 贪婪解码: 合并连续重复 + 去 blank, 再走 postprocess。"""
        collapsed: List[int] = []
        prev = None
        for i in frame_ids:
            i = int(i)
            if i == prev:
                continue
            prev = i
            if i == self.blank_id:
                continue
            collapsed.append(i)
        return self.decode_ids(collapsed, skip_special=True)

    # ------------------------------------------------------------------
    # 以下解码后处理完全移植自 FunASR funasr/utils/postprocess_utils.py
    # (sentence_postprocess / abbr_dispose / isChinese / isAllChinese / isAllAlpha)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_chinese(ch: str) -> bool:
        return ("\u4e00" <= ch <= "\u9fff") or ("\u0030" <= ch <= "\u0039") or ch == "@"

    @classmethod
    def _is_all_chinese(cls, word) -> bool:
        word_lists = []
        for i in word:
            cur = i.replace(" ", "").replace("</s>", "").replace("<s>", "")
            cur = cur.replace("<unk>", "").replace("<OOV>", "")
            word_lists.append(cur)
        if len(word_lists) == 0:
            return False
        for ch in word_lists:
            if cls._is_chinese(ch) is False:
                return False
        return True

    @classmethod
    def _is_all_alpha(cls, word) -> bool:
        word_lists = []
        for i in word:
            cur = i.replace(" ", "").replace("</s>", "").replace("<s>", "")
            cur = cur.replace("<unk>", "").replace("<OOV>", "")
            word_lists.append(cur)
        if len(word_lists) == 0:
            return False
        for ch in word_lists:
            if ch.isalpha() is False and ch != "'":
                return False
            elif ch.isalpha() is True and cls._is_chinese(ch) is True:
                return False
        return True

    @staticmethod
    def _abbr_dispose(words: List[str]) -> List[str]:
        words_size = len(words)
        word_lists: List[str] = []
        abbr_begin: List[int] = []
        abbr_end: List[int] = []
        last_num = -1
        for num in range(words_size):
            if num <= last_num:
                continue
            if len(words[num]) == 1 and words[num].encode("utf-8").isalpha():
                if (num + 1 < words_size and words[num + 1] == " " and num + 2 < words_size
                        and len(words[num + 2]) == 1 and words[num + 2].encode("utf-8").isalpha()):
                    abbr_begin.append(num)
                    num += 2
                    abbr_end.append(num)
                    while True:
                        num += 1
                        if num < words_size and words[num] == " ":
                            num += 1
                            if (num < words_size and len(words[num]) == 1
                                    and words[num].encode("utf-8").isalpha()):
                                abbr_end.pop()
                                abbr_end.append(num)
                                last_num = num
                            else:
                                break
                        else:
                            break
        last_num = -1
        for num in range(words_size):
            if num <= last_num:
                continue
            if num in abbr_begin:
                abbr_word = words[num].upper()
                num += 1
                while num < words_size:
                    if num in abbr_end:
                        abbr_word += words[num].upper()
                        last_num = num
                        break
                    else:
                        if words[num].encode("utf-8").isalpha():
                            abbr_word += words[num].upper()
                    num += 1
                word_lists.append(abbr_word)
            else:
                word_lists.append(words[num])
        return word_lists

    @classmethod
    def _sentence_postprocess(cls, words: List[str]) -> str:
        middle_lists: List[str] = []
        word_lists: List[str] = []
        word_item = ""

        for word in words:
            if word in ["<s>", "</s>", "<unk>", "<OOV>", BLANK]:
                continue
            middle_lists.append(word)

        if cls._is_all_chinese(middle_lists):
            for ch in middle_lists:
                word_lists.append(ch.replace(" ", ""))

        elif cls._is_all_alpha(middle_lists):
            for ch in middle_lists:
                if "@@" in ch:
                    word_item += ch.replace("@@", "")
                else:
                    word_item += ch
                    word_lists.append(word_item)
                    word_lists.append(" ")
                    word_item = ""

        else:
            alpha_blank = False
            for ch in middle_lists:
                if cls._is_all_chinese(ch):
                    if alpha_blank is True:
                        word_lists.pop()
                    word_lists.append(ch)
                    alpha_blank = False
                elif "@@" in ch:
                    word_item += ch.replace("@@", "")
                    alpha_blank = False
                elif cls._is_all_alpha(ch):
                    word_item += ch
                    word_lists.append(word_item)
                    word_lists.append(" ")
                    word_item = ""
                    alpha_blank = True
                else:
                    word_lists.append(ch)

        word_lists = cls._abbr_dispose(word_lists)
        return "".join(word_lists).strip()


@lru_cache(maxsize=1)
def get_vocab(tokens_path: str = _TOKENS_PATH) -> ParaformerVocab:
    """进程内单例, 避免重复加载。"""
    return ParaformerVocab(tokens_path)


if __name__ == "__main__":
    v = get_vocab()
    print("vocab_size:", v.vocab_size, "blank_id:", v.blank_id, "unk_id:", v.unk_id)
    for s in ["吉莉帮我存一瓶药", "打雷了快跑", "hello世界"]:
        ids = v.encode_text(s)
        print(f"{s!r} -> {ids} -> {v.decode_ids(ids)!r}")
    # CTC 贪婪解码演示 (含 blank=0 与重复)
    demo = [0, 1912, 1912, 0, 0, 2923, 2923, 2923, 0]
    print("greedy:", v.ctc_greedy_decode(demo))
