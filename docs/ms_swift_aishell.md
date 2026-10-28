# 使用 ms-swift 微调 Qwen3-ASR-0.6B 并交给 vLLM 推理

本流程从官方预训练模型 `Qwen/Qwen3-ASR-0.6B` 开始微调，并非随机初始化训练。
阶段一训练 LLM 的热词上下文能力，冻结音频编码器和投影；阶段二冻结整个主模型，只训练新增 CTC 分支。
这些是本项目提供的训练方案和起始超参数，作者未公开完整训练配方，不承诺复现其指标。

已使用用户指定的 ms-swift 环境、官方 0.6B 权重和公开短音频，在 A800 上完成两阶段各 2 步训练、
验证、保存、导出及 vLLM 0.22.1 的默认 torch.compile/CUDA Graph 推理检查。
CTC 训练前后主模型的 611 个权重张量完全一致，vLLM 加载的 84 个 CTC 张量已逐项校验。
这项短流程验证不代表完成了 AISHELL-1 全量训练，也不代表热词识别效果已经收敛。

SeACo 的 `data/dev` 和 `data/test` 是评测子集，没有提供训练集。
训练使用 AISHELL-1 官方 `wav/train`；验证使用 `wav/dev`；最终测试使用 SeACo 的 test ID 子集。
不要将 SeACo 的 dev/test 音频、转写或热词表混入训练集。

## 1. 环境与文件

本流程针对 ms-swift 4.4.2 / Transformers 4.57.6；本项目的自定义 CTC 类通过 external plugin 加载，
不需要安装 `qwen-asr`。训练使用 PyTorch/ms-swift，部署阶段全部神经网络推理由 vLLM 执行。

```bash
conda activate ms-swift
cd /path/to/qwen3-asr-hotword-toolkit
python -m pip install -r training/requirements.txt
python -m pip install -e . --no-deps
```

新增文件：

- `training/prepare_aishell.py`：按官方划分转换数据并构造训练热词。
- `training/swift_plugin.py`：注册本项目模型、音频模板和 CTC 标签/loss。
- `scripts/train_swift.sh`：两个阶段的训练入口，可通过额外参数覆盖默认值。
- `training/export_checkpoint.py`：完整权重检查和推理导出。

还需准备官方 0.6B 模型、AISHELL-1 原始音频与转写、SeACo 的评测 ID 和热词表。

## 2. 下载并解包数据

AISHELL-1 音频压缩包约 15 GB，解包后另需空间；训练 checkpoint 也需预留空间。
原始下载地址见 [OpenSLR 33](https://www.openslr.org/33/)。例如在项目中执行：

```bash
mkdir -p data
cd data
curl -fL -C - -o data_aishell.tgz https://openslr.magicdatatech.com/resources/33/data_aishell.tgz
tar -xzf data_aishell.tgz
# 外层包的 wav/ 中仍有每个说话人的压缩包，需再解一次。
find data_aishell/wav -maxdepth 1 -type f -name '*.tar.gz' \
  -exec tar -xzf '{}' -C data_aishell/wav \;
# 若 SeACo 的 data/ 已准备好，跳过克隆。
if [ ! -d SeACo-Paraformer/data ]; then
  git clone https://github.com/R1ckShi/SeACo-Paraformer.git
fi
cd ..
```

预期目录：

```text
data/data_aishell/
  transcript/aishell_transcript_v0.8.txt
  wav/train/<speaker>/*.wav
  wav/dev/<speaker>/*.wav
  wav/test/<speaker>/*.wav
data/SeACo-Paraformer/data/{dev,test}/{uttid,hotword.txt,r1-hotword.txt}
```

数据准备脚本拒绝跨划分重复 ID，检查音频为 16 kHz 单声道 PCM16，并过滤超长音频。

```bash
python -m training.prepare_aishell \
  --aishell-root data/data_aishell \
  --seaco-root data/SeACo-Paraformer \
  --output-dir artifacts/aishell
```

生成 `train_sft.jsonl` / `dev_sft.jsonl`、`train_ctc.jsonl` / `dev_ctc.jsonl`，
以及 `dev_ne.jsonl` / `test_ne.jsonl` 和各自的评测热词表。`summary.json` 记录数量与过滤情况。

AISHELL-1 没有配套的命名实体训练标注。本脚本默认仅从 train 转写中用 jieba 的人名/地名/机构名/
专名词性启发式提取候选；它不等价于人工实体标注。可用 `--train-hotwords /path/to/train-only.txt`
提供质量更高的训练词表；只保留在 train 转写中出现的词。
默认样本混合为 20% 空上下文、10% 仅干扰词、其余含候选正热词及干扰词；
每条最多 4 个正热词、8 个干扰词。剩余样本如果没有命中正热词，也只含干扰词。
训练和验证构造使用同一份 train 词表，SeACo 的评测词表不参与训练词表构造。

主模型训练样式如下；assistant 是正确转写，不能把热词列表作为答案：

```json
{"messages":[{"role":"system","content":"可能出现的热词：北京大学，清华大学"},{"role":"user","content":"<audio>"},{"role":"assistant","content":"北京大学欢迎你"}],"audios":["/absolute/path/to/train.wav"]}
```

CTC 阶段使用空 system，插件将参考文本转换成 Paraformer 8404 词表标签。数据准备时，若原文
含词表外字符，会先尝试 Unicode NFKC 归一化（例如全角英文字母转为半角）；若仍无法编码，则仅从
CTC 清单中跳过该样本。归一化及跳过的样本 ID 和数量记录在 `summary.json` 的 `ctc_preparation` 中；
SFT、ASR 和实体评测清单仍保留原始转写。
训练脚本默认启用 ms-swift 的严格模式：遇到未知 CTC 标签、无法对齐的样本或其他数据错误时
直接报错，不用其他样本替代。应修正或显式过滤出错的样本后再继续训练和验证。

## 3. 下载 0.6B 基座并训练热词 SFT

脚本默认加载项目内的 `model/Qwen3-ASR-0.6B`，并从 `artifacts/aishell` 读取数据。
当前机器的该模型目录已链接到本地基座。其他环境需将权重放入此目录，或建立符号链接：

```bash
mkdir -p model
ln -s /path/to/Qwen3-ASR-0.6B model/Qwen3-ASR-0.6B
```

也可以通过 `MODEL_PATH` 和 `DATA_DIR` 指定其他模型或数据路径。

```bash
bash scripts/train_swift.sh
```

也可以运行 `hf download Qwen/Qwen3-ASR-0.6B --local-dir model/Qwen3-ASR-0.6B` 下载基座。
不要使用作者的 1.7B 权重代替 0.6B 基座。

这里采用完整 LLM 参数微调，冻结 `thinker.audio_tower`（含投影），不使用 LoRA。
默认 FP32 参数 + BF16 自动混合精度计算，避免小学习率更新直接舍入到 BF16 参数中；
默认使用物理 GPU 3、6 做单机两卡数据并行，每卡 batch=8、累积 8 步，有效 batch=128；
同时使用 SDPA、梯度检查点、学习率 `2e-6`、2 个 epoch。可以通过
`CUDA_VISIBLE_DEVICES=1,3 bash scripts/train_swift.sh` 换卡；`NPROC_PER_NODE` 默认等于所选卡数。
有效 batch = 每卡 batch × GPU 数 × 梯度累积步数；可在脚本末尾通过
`--per_device_train_batch_size` 和 `--gradient_accumulation_steps` 覆盖默认值。
更换 GPU 数量时，有效 batch 也会随之变化。多任务端口冲突时可设置 `MASTER_PORT`。
每卡 batch=8 尚未经真实多样本 GPU 训练验证；不同长度音频会影响显存占用，建议先跑短流程。
先跑少量步数检查数据/显存，然后根据 dev loss 和实际热词识别指标调参；不要仅依据训练 loss。
不同显卡的显存占用需实测，显存不足可先降低 `--max_length`，但必须确保音频 token 未被截断。

短流程检查示例：

```bash
OUTPUT_DIR="$PWD/artifacts/sft-smoke" \
  bash scripts/train_swift.sh sft \
  --max_steps 2 --gradient_accumulation_steps 1 \
  --save_steps 2 --eval_steps 2 --logging_steps 1
```

正式训练输出在 `artifacts/sft-0.6b/checkpoint-<step>`。阶段二默认选择包含配置与权重文件的
最高步数 checkpoint，并打印所选路径；最高步数不一定是验证集上最优的结果，且复用输出目录时可能包含旧运行的检查点。
如需选择其他 checkpoint，应通过 `MODEL_PATH` 明确指定。阶段二不能重新指向官方基座。

## 4. 冻结主模型，训练 CTC

```bash
bash scripts/train_swift.sh ctc

# 可选：指定根据验证结果选出的 SFT checkpoint
MODEL_PATH="$PWD/artifacts/sft-0.6b/checkpoint-<step>" \
  bash scripts/train_swift.sh ctc
```

CTC loader 在阶段一模型上创建 CTC 分支（首次加载出现该分支未初始化的提示属于预期），
然后由 ms-swift 冻结 `thinker` 并只解冻 `thinker.ctc_branch`。
默认 5 层、8 个注意力头、FFN 4096、8404 输出字表，0.6B 的分支输入宽度由编码器输出自动确定。
默认学习率 `1e-4`、5 个 epoch；随机初始化的分支应单独调学习率，而不直接沿用 LLM 的 `2e-6`。
CTC 阶段不计算 LM loss，编码器在 no_grad 下执行，也不执行 LLM 前向。
插件检查 CTC 标签和最少对齐帧数，避免无效样本被 `zero_infinity` 静默跳过。

阶段二保存的是包含编码器、LLM 和已训练 CTC 分支的完整 checkpoint。
如果之后重新训练/修改音频编码器，需要重新适配 CTC，不能随意拼接不同模型的分支。

## 5. 导出并用 vLLM 验证

```bash
python -m training.export_checkpoint \
  --checkpoint artifacts/ctc-0.6b/checkpoint-<step> \
  --output ckpt/qwen3-asr-hotword-0.6b-ctc
```

导出检查所有权重是否完整加载，并一并保存 tokenizer、processor 和生成配置。
默认要求模型带 CTC。只部署 SFT 模型的纯 ASR 模式时，可显式加 `--asr-only`。
使用已有的 vLLM 环境（版本和安装方式见主 README），不要在训练环境盲目升级推理依赖。

```bash
CUDA_VISIBLE_DEVICES=0 python infer_vllm.py \
  --model ckpt/qwen3-asr-hotword-0.6b-ctc \
  --audio /path/to/example.wav \
  --mode ctc_rag --hot-txt /path/to/deployment-hotwords.txt \
  --verify-encoder-reuse

# 开发集用于调检索阈值和热词上限。
CUDA_VISIBLE_DEVICES=0 python infer_vllm.py \
  --model ckpt/qwen3-asr-hotword-0.6b-ctc \
  --test-json artifacts/aishell/dev_ne.jsonl \
  --hot-txt artifacts/aishell/dev_hotword.txt --mode ctc_rag \
  --rag-threshold 0.8 --max-ctx-hotwords 8 \
  --out-json artifacts/dev_results.jsonl

# 参数确定后再进行最终 test 评测。
CUDA_VISIBLE_DEVICES=0 python infer_vllm.py \
  --model ckpt/qwen3-asr-hotword-0.6b-ctc \
  --test-json artifacts/aishell/test_ne.jsonl \
  --hot-txt artifacts/aishell/test_hotword.txt --mode ctc_rag \
  --rag-threshold 0.8 --max-ctx-hotwords 8 \
  --out-json artifacts/test_results.jsonl
```

上述 CLI 输出 CER；热词 Recall/Precision/F1 还需要按 SeACo 的非重叠实体计数口径单独打分。
CTC+RAG 正常识别不能证明训练有效；应与官方基座、SFT 后纯 ASR、最终 CTC+RAG 在同一评测集上对比。

参考：[ms-swift 自定义模型](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-model.html)、
[SeACo 评测集](https://github.com/R1ckShi/SeACo-Paraformer#open-hotword-customization-test-sets)、
[AISHELL-1](https://www.openslr.org/33/)。
