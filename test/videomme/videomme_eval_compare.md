# Video-MME 评测参数对比：原生 Torch vs llama.cpp-omni

> 对比对象
> - **Torch 版**：`/cache/hanqingzhe/Video-MME/py-eval`（`run_videomme.py` + `videomme.py`），以 HuggingFace Transformers + `decord` 直接加载 `MiniCPM-o-4_5` bf16 权重调用 `model.chat(...)`。
> - **CPP 版**：`/cache/hanqingzhe/Video-MME/cpp-eval/videomme`，基于打过补丁的 `llama.cpp-omni` `llama-server`，通过 HTTP omni streaming API (`omni_init` / `reset` / `prefill` / `decode`) 推理。
>
> 数据集统一为 `/cache/hanqingzhe/Video-MME/Video-MME/videomme/test-00000-of-00001.parquet` + `data/<videoID>.mp4`。
>
> **⚠️ 实际运行配置说明**：Torch 版 `py-eval/videomme.py` 代码里虽然**写了** CoT / subtitle / stack 拼帧 / timestamp 采样等逻辑，但 `py-eval/commend` 启动脚本
>
> ```bash
> CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_videomme.py 2>&1 | tee videomme_run.log
> ```
>
> **没有设置任何相关环境变量**，全部走默认值，`videomme_run.log` 也显示 `use subtitles=False`：
>
> | 环境变量 | 默认值 | 实际生效行为 |
> |---|---|---|
> | `USE_COT` | `0` | 不走 CoT，用普通多选模版，`max_new_tokens=100` |
> | `USE_SUBTITLE` | `0` | 不注入字幕 |
> | `STACK_NUMS` | `1` | 不做网格拼帧 |
> | `KEY_FRAME_NUMS` | `0` | — |
> | `SAMPLE_STRATEGY` | `default` | 1 FPS + 超过 64 帧时 uniform_sample |
> | `MAX_NUM_FRAMES` | `64` | — |
>
> 因此本文档的"差异"讨论分两层看：
> 1. **代码层面可配置差异**（Torch 支持但 CPP 没实现，如 CoT/subtitle/stack） — CPP 版目前没对齐；
> 2. **当前实际评测中真正生效的差异** — 只剩下"模型精度 / 图像通路 / `max_slice_nums` 配置疑点 / `min_new_tokens`"这几项。
>
> 所以只要后续 Torch 侧维持 `commend` 的默认跑法，CPP 侧只需对齐第 2 层就够了；不需要急着把 CoT 和 subtitle 在 CPP 上也实现出来。

---

## 1. 运行入口与并行方式

| 项目 | Torch 版 | CPP 版 |
|---|---|---|
| 启动命令 | `CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_videomme.py`（见 `py-eval/commend`） | `python eval_cpp_pipeline.py --num-gpus N --base-port P`（见 `cpp-eval/videomme/README.md`） |
| 模型加载 | `AutoModel.from_pretrained(..., attn_implementation="sdpa", torch_dtype=bfloat16, init_vision=True, init_audio=False, init_tts=False)` | `llama-server` 加载 `MiniCPM-o-4_5-llm-Q4_K_M.gguf`（`LLM_MODEL_PATH`），视觉/音频 encoder 由 `omni_init` 在 `GGUF_MODEL_DIR` 里按需加载；`media_type=2`（audio+vision），`use_tts=false` |
| 并行粒度 | `torchrun` 多进程，每 rank 按 `itertools.islice(range(len(df)), rank, None, world_size)` 做数据切片，单卡串行推理 | 按 `video_id` 分组成 900×3 题；`split_into_chunks` 按 GPU 均匀切分；每张 GPU 一个 `llama-server` + 一条 worker 线程，内部**串行**处理视频 |
| 单题 KV 管理 | `model.chat` 每次 stateless，自然独立 | 每题前 `POST /v1/stream/reset` 清空 KV cache + `system_prompt_initialized=false`（补丁新增） |

---

## 2. 视频抽帧（Frame Sampling）

### 2.1 Torch 版（`py-eval/videomme.py`）

核心环境变量：

```text
MAX_NUM_FRAMES   = 64   （默认）
STACK_NUMS       = 1    默认不做网格拼接
KEY_FRAME_NUMS   = 0
SAMPLE_STRATEGY  = default    可选: default / uniform / timestamp
USE_COT          = 0
USE_SUBTITLE     = 0
```

解码用 `decord.VideoReader`，取 `avg_fps = vr.get_avg_fps()`、`total_frames = len(vr)`、`sample_fps = round(avg_fps / 1)`，按策略生成 `frame_idx`：

| `SAMPLE_STRATEGY` | 采样逻辑 |
|---|---|
| `default`（默认） | `frame_idx = range(0, len(vr), sample_fps)` ≈ 1 FPS |
| `uniform` | `frame_idx = range(0, len(vr))` 全量，后面统一下采样 |
| `timestamp` | 长视频（>MAX_NUM_FRAMES 秒）先按 0.1 s 间隔生成候选，再 `uniform_sample` 到 64；短视频按 1 s 间隔取帧并附时间戳 |

上限控制（非 `timestamp` 策略）：

```python
if stack_nums > 1:
    if key_frame_nums > 0:
        max_num_frame = (MAX_NUM_FRAMES // 2) * (stack_nums + key_frame_nums)
    else:
        max_num_frame = MAX_NUM_FRAMES * stack_nums
else:
    max_num_frame = MAX_NUM_FRAMES  # 64
if len(frame_idx) > max_num_frame:
    frame_idx = uniform_sample(frame_idx, max_num_frame)
```

其中 `uniform_sample` 的下标为每段中点 `int(i*gap + gap/2)`（`videomme.py:129-132`）。

取帧后：

```python
video = vr.get_batch(frame_idx).asnumpy()
video = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in video]
```

随后根据 `stack_nums/key_frame_nums` 可能调用 `concat_images` 把若干帧拼成 2×2 / 3×1 / 2×1 等网格（letterbox 白底、黑色分隔线、`line_width=6`），并附带 system prompt 解释 "混合帧 + 合成帧" 的格式（默认关闭，`stack_nums=1` 时直接用原始帧）。

### 2.2 CPP 版（`eval_cpp_video_prep.py`）

参数（`eval_cpp_config.py`）：

```text
MAX_NUM_FRAMES = 64
MAX_FPS        = 1.0
MAX_SLICE_NUMS = 0            ← 见下方 3.2 注意事项
```

解码优先级 `decord → ffmpeg CLI → torchvision`，只对齐 **`default` 策略**：

```python
sample_fps = max(round(avg_fps), 1)
frame_idx  = list(range(0, total_frames, sample_fps))
if len(frame_idx) > MAX_NUM_FRAMES:
    frame_idx = _uniform_sample(frame_idx, MAX_NUM_FRAMES)
frames_np  = vr.get_batch(frame_idx).asnumpy()
frames     = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in frames_np]
```

`_uniform_sample` 公式 (`gap = len/n, idx = int(i*gap + gap/2)`) 与 Torch 版完全一致。

ffmpeg fallback 用 `-vf fps=1.0 -vframes 64` 直接抽帧；torchvision fallback 用 `torch.linspace(0, total_frames-1, nframes).round()` 均匀取帧。**这两条路径的采样位置跟 decord 不同**，只是应急路径。

### 2.3 差异小结

| 差异点 | Torch | CPP |
|---|---|---|
| 抽帧策略 | 支持 `default / uniform / timestamp`；`timestamp` 有时间戳注入 | **只有 default 一种** |
| 网格拼帧 `stack_nums / key_frame_nums` | ✅ 支持 `concat_images` | ❌ 未实现 |
| 候选帧上限 | `stack_nums * MAX_NUM_FRAMES`（最多 64 × stack） | 固定 64 |
| fallback 解码器 | 仅 decord（失败直接抛异常） | decord → ffmpeg → torchvision |
| 抽帧输出 | `PIL.Image` 列表直接进 `model.chat` | `PIL.Image` 用 `JPEG quality=95` 写入 `tmp_frames/<video_id>/frame_%03d.jpg`，路径传给 server |
| JPEG 损失 | 无 | **有**：JPEG 编码 → server `stbi_load` 再解码，会引入肉眼几乎看不见但存在的量化噪声 |

---

## 3. 图片预处理 / 视觉编码

### 3.1 Torch 版

`model.chat(msgs=[{'role':'user', 'content': final_video + [question]}], max_slice_nums=1, use_image_id=False, ...)`

- `max_slice_nums=1`：MiniCPM-V 的 HD 切片关闭，每帧**只编码 1 个 overview chunk**（≈96 tokens）
- `use_image_id=False`：不在每张图片前加 `<img_i>` 标识
- 图片 tensor 由内部 `image_processor` 做 resize / normalize，bf16 送入 ViT

### 3.2 CPP 版

每帧单独调一次 `/v1/stream/prefill`：

```json
{
  "audio_path_prefix": "",
  "img_path_prefix": "/abs/path/tmp_frames/<video_id>/frame_000.jpg",
  "cnt": 0,
  "max_slice_nums": 0,
  "prompt": "\n",
  "skip_system_prompt": true   // 仅第一帧
}
```

关键设计：

1. **`skip_system_prompt=true`（仅 `cnt=0`）**：补丁 (`llama-cpp-omni.patch`) 新增的参数，直接跳过 llama.cpp-omni 原本硬编码的 voice-clone system prompt，只 eval `<|im_start|>user\n`，对齐 Torch 版 `system_prompt=""`。
2. **帧后换行分隔**：每帧 prefill 请求里附带 `"prompt": "\n"`（`eval_cpp_http_client.py:109, 119-126`），用于在 `max_slice_nums=1` 时显式产生 "每帧后换行分隔"，更贴近 Torch 侧把帧列表和文本拼接的输入形态。
3. **`max_slice_nums` 注意事项**：配置文件里写的是 `MAX_SLICE_NUMS = 0`，而补丁文档约定 `-1=用全局设置, >=1=本次切片数`，**`0` 不在规范取值内**。对齐 Torch 版 `max_slice_nums=1` 的做法应当是把它改成 `1`。这是当前两侧**尚未完全对齐**的一个点。
4. Server 端 ViT：每帧单独编码为 vision embedding 后直接 `llama_decode` 进 KV；整套视频 64 帧 ≈ 64×96+分隔符，远低于 `CTX_SIZE=40960`。

### 3.3 差异小结

| 差异点 | Torch | CPP |
|---|---|---|
| 图像进入模型的路径 | in-memory PIL → processor → ViT | PIL → JPEG file → server → `stbi_load` → ViT |
| 每帧分隔符 | 靠 `model.chat` 内部对 list-of-images 的处理，无显式 `\n` | 显式 `prompt="\n"` 每帧追加 |
| `max_slice_nums` | 1（无切片） | **0**（与文档规范不符，需改为 1） |
| `use_image_id` | False | 未暴露，`llama.cpp-omni` 默认不加 image_id |
| 精度 | bf16 ViT | GGUF 模型（如 `Q4_K_M`）量化 ViT + LLM |

---

## 4. 提示词（Prompt）模板

### 4.1 Torch 版（`videomme.py:258-266`）

**不开 CoT（默认，`USE_COT=0`）：**

```text
Carefully read the following question and select the letter corresponding to the correct answer.Highlight the applicable choices without giving explanations.
{question}
Options:
{option_0}
{option_1}
{option_2}
{option_3}
```

> 注意：`correct answer.` 与 `Highlight` 之间**没有空格**，是 Python 代码里字面量拼接的，CPP 侧保持了同样的字面量。

**开 CoT（`USE_COT=1`）：**

```text
Carefully read the following multichoice question, solve it step by step and finally pick the option associated with the correct answer in the format of "Answer: selected option"

{question}
Options:
{option_0}
...
```

**Stack 拼帧情境（`stack_nums>1`）** 额外注入 system prompt：

```text
You are a video model receiving a mixed frame sequence: some are standard
full frames, others are composite (multi grids, left-to-right, top-to-bottom
order). Parse all frames into a unified temporal sequence, then analyze the
content.
```

**Subtitle 情境（`use_subtitle=True` 且 `.srt` 存在）**：在用户 content 最前面插入：

```text
Imagining yourself as a customer service agent overseeing an uploaded video. The video comprises a sequence of clips and subtitles.
```

随后按帧组插入 `subtitle: ...` 片段，`timestamp` 策略下还会在每张图前加 `<12.3 seconds>` 标记。

消息结构：`[{'role':'system', 'content': sys_prompt}?, {'role':'user', 'content': [img_or_grid, ..., text_parts]}]`。

### 4.2 CPP 版（`eval_cpp_config.py` + `eval_cpp_pipeline.py`）

```python
USER_PROMPT_TEMPLATE = (
    "Carefully read the following question and select the letter corresponding to the correct answer."
    "Highlight the applicable choices without giving explanations.\n"
    "{question}\n"
    "Options:\n{options}"
)
options_text = "\n".join(options)   # parquet 里 options 已是 "A. xxx" 形式
```

展开后字符串与 Torch 版**完全一致**（含那个紧贴的 `correct answer.Highlight`）。

对话骨架：

```
[frame 0]  \n        ← cnt=0, skip_system_prompt=true
[frame 1]  \n
...
[frame 63] \n
<USER_PROMPT_TEMPLATE>           ← cnt=64, 文本 prefill
decode()
```

**不支持的特性**：`use_cot` / `stack_nums` / `use_subtitle` / `timestamp 时间戳注入` 在 CPP pipeline 中**均未实现**；system prompt 永远靠 `skip_system_prompt` 跳过，不下发任何 system 消息。

### 4.3 差异小结

| 差异点 | Torch | CPP |
|---|---|---|
| 基础多选模版 | 完全一致 | 完全一致 |
| CoT prompt | 可通过 `USE_COT=1` 启用 | 未实现 |
| System prompt（默认） | 不发送（`sys_prompt=""`） | 通过 `skip_system_prompt=true` 在 server 内显式跳过硬编码的 voice-clone system prompt |
| Subtitle | 支持（`pysubs2` 合并时间窗内字幕） | 未实现 |
| 时间戳注入 | `timestamp` 策略下 `<t seconds>` 前缀 | 未实现 |
| 每帧分隔 | model.chat 内部处理 | 每帧 prefill 带 `"\n"` |

---

## 5. 采样 / 解码参数

### 5.1 Torch 版（`videomme.py:351-366`）

```python
res = model.chat(
    image=None,
    msgs=inp_msgs,
    tokenizer=tokenizer,
    do_sample=True,
    num_beams=1,
    max_new_tokens=max_new_tokens,   # CoT=8192, 否则 100
    min_new_tokens=1,
    temperature=0.2,
    top_p=0.8,
    top_k=100,
    repetition_penalty=1.02,
    max_inp_length=4096*10,          # 40960
    max_slice_nums=1,
    use_image_id=False,
)
```

### 5.2 CPP 版

分两层：

1. **llama-server 启动参数**（`eval_cpp_server_manager.py:107-118`）：

   ```text
   --ctx-size   40960   # CTX_SIZE
   --n-gpu-layers 999
   --temp          0.2  # TEMPERATURE
   --top-p         0.8
   --top-k         100
   --repeat-penalty 1.02
   ```

2. **`omni_init` 请求**（`eval_cpp_http_client.py:69-82`）：

   ```json
   {"media_type": 2, "use_tts": false, "n_predict": 100, "model_dir": "<GGUF_MODEL_DIR>"}
   ```

   `n_predict=100` 对齐 Torch `max_new_tokens=100`。

### 5.3 差异小结

| 参数 | Torch | CPP | 备注 |
|---|---|---|---|
| 解码策略 | `do_sample=True, num_beams=1` | server 默认采样（`--temp/--top-p/--top-k`） | 两侧都是单路随机采样 |
| `temperature` | 0.2 | 0.2 | ✅ |
| `top_p` | 0.8 | 0.8 | ✅ |
| `top_k` | 100 | 100 | ✅ |
| `repetition_penalty` | 1.02 | 1.02 | ✅ |
| `max_new_tokens` | 100（非 CoT） | `n_predict=100` | ✅ |
| `min_new_tokens` | 1 | 未设置 | 轻微差异，`llama.cpp` 默认 0 |
| `max_inp_length` | 40960 | `CTX_SIZE=40960` | ✅ |
| `max_slice_nums` | 1 | **0**（与文档规范不符） | ⚠️ 需修 |
| `use_image_id` | False | 不支持配置 | 一致（server 默认不加） |
| CoT 8192 tokens | 支持 | 不支持（`MAX_TOKENS` 固定 100） | ⚠️ |
| seed | 未指定（非 deterministic） | 未指定 | 两侧均为随机采样，多次运行会有轻微波动 |
| 权重精度 | bf16 | GGUF（如 `Q4_K_M`） | **最根本的精度差** |

> `eval_cpp_config.py` 里对 `TEMPERATURE` 的注释写的是 "`0.0` 为 greedy（对齐 Python `do_sample=False`）"，而实际 Torch 版是 `do_sample=True, T=0.2`，**注释与代码/Torch 实现都不一致**，以实际代码 `0.2` 为准。

---

## 6. 答案后处理与打分

### 6.1 Torch 版

- 直接取 `res[0].lower() == answer.lower()` 判断首字母命中
- 若不命中：把 `.()[],:;!*#{}` 替换空格后 `.split()`，统计 `{A,B,C,D}` 出现情况；若正确答案单独出现且只命中 1 个选项（`A` 单独出现时要求 `len(predict_all)<5` 过滤异常 token），视为命中
- CoT 模式下先 `res.split('Answer:')[-1].strip()` 再走上面流程
- 分 `short / medium / long` 三档统计 accuracy

### 6.2 CPP 版

- `extract_answer` 函数 (`eval_cpp_pipeline.py:124-145`)：优先整串只含单个 A–D，否则正则 `(?<![a-zA-Z])([A-D])(?![a-zA-Z])` 抓首个独立字母
- 主 pipeline 记录 `response = <提取后的字母>`（原始模型输出只写到 log，不落盘）
- 后续由 `eval_your_result.py`（Video-MME 官方脚本）按 `duration / domain / sub_category / task_type` 维度统计
- 配套 `rerun_failed.py`：对 `response ∉ {A,B,C,D}` 的题目单卡重跑

### 6.3 差异小结

| 差异点 | Torch | CPP |
|---|---|---|
| 原始输出保留 | 存 `ori_res` 和 `conversation`（图片替换成 `<image_i>`） | 只存清洗后的字母到 `response` 字段，原文仅日志 |
| 宽松匹配逻辑 | Python 自行实现一套 | `extract_answer` 更简洁，并追加 `rerun_failed` 兜底 |
| 分桶统计 | `short/medium/long` 三档 | 使用 `eval_your_result.py` 多维（含 domain/sub_category/task_type） |

---

## 7. 端到端关键差异一览

按 **"当前 `commend` 默认跑法实际生效值"** 列：

| 维度 | Torch 实际生效 | CPP 版 | 对齐情况 |
|---|---|---|---|
| 模型权重 | bf16 HF | GGUF 量化（默认 Q4_K_M） | 精度不同，**这是准确率差的主因** |
| 抽帧策略 | `default`（1 FPS + 超 64 帧 uniform）；uniform/timestamp 代码里有但未启用 | 仅 `default` | ✅ 实际一致 |
| stack 拼帧 | `stack_nums=1`，未启用 | 不支持 | ✅ 实际一致 |
| 图像通路 | in-mem PIL → ViT | JPEG 落盘 → server → ViT | JPEG 有轻微损失 |
| `max_slice_nums` | 1 | 0（配置疑似误填） | ⚠️ 需要改成 1 |
| `use_image_id` | False | False（默认） | ✅ |
| System prompt | 空（`stack_nums=1` 默认分支不注入） | 通过 `skip_system_prompt=true` 跳过 voice-clone SP | ✅（补丁对齐） |
| 每帧分隔 | model.chat 内部 | 每帧 `"\n"` | ✅ 近似对齐 |
| User prompt 模板 | 非 CoT 版本（`USE_COT=0`） | 与 Torch 非 CoT 版本一致 | ✅ |
| CoT（`USE_COT=1` → max_new=8192） | 代码支持，**默认关** | 未实现 | ✅ 实际一致（潜在风险项） |
| Subtitle（`USE_SUBTITLE=1`） | 代码支持，**默认关**（日志 `use subtitles=False`） | 未实现 | ✅ 实际一致（潜在风险项） |
| 采样参数 (T/top_p/top_k/rep_pen) | 0.2 / 0.8 / 100 / 1.02 | 0.2 / 0.8 / 100 / 1.02 | ✅ |
| `max_new_tokens` | 100 | 100（`n_predict`） | ✅ |
| `min_new_tokens` | 1 | 未设置 | 小差异 |
| Ctx 长度 | 40960 | 40960 (`CTX_SIZE`) | ✅ |
| 并行 | `torchrun` 多 rank 多进程 | `ThreadPoolExecutor` + 每卡一个 `llama-server` | 独立实现 |
| KV 管理 | 每题 `model.chat` 天然独立 | 每题 `reset` + `system_prompt_initialized=false` | ✅ |
| 结果记录粒度 | 保留 `ori_res / conversation / hit / index` | 只保留提取后的字母 | Torch 更便于回溯 |

换句话说，**按默认命令跑，真正会影响 acc 对比的差异只剩 3 条**：

1. 模型精度（bf16 vs GGUF 量化）
2. `max_slice_nums`（Torch=1，CPP 误填 0）
3. 图像通路的 JPEG 编解码损耗（不可避免）

其余诸如 CoT / Subtitle / Stack / Timestamp / Uniform 的差异都只是"代码能力差异"，并不会在当前评测中实际触发。

---

## 8. 复现 / 对齐建议（落地 TODO）

### 8.1 必须修（直接影响当前评测对齐）

1. **把 `MAX_SLICE_NUMS` 从 `0` 改成 `1`**，保持与 Torch `max_slice_nums=1` 一致，避免落入 `llama.cpp-omni` 未定义行为。
2. **记录原始响应**：把 `response_text` 也写到输出 JSON（新加一个 `raw_response` 字段），方便对照 Torch 的 `ori_res`。
3. **修正 `eval_cpp_config.py` 里 `TEMPERATURE` 的注释**："0.0 为 greedy" 的说法与代码（0.2 随机采样）、与 Torch (`do_sample=True, T=0.2`) 都不符。

### 8.2 仅当 Torch 侧改启用高级选项时才需要跟进

当前 `py-eval/commend` 默认跑法不会触发以下特性，CPP 侧**暂不需要实现**，但留作潜在对齐项：

4. **CoT 支持**（仅 `USE_COT=1` 时必要）：
   - 把 `USER_PROMPT_TEMPLATE` 切换成 `mc_cot_prompt + question + Options:`；
   - `omni_init` 里把 `n_predict` 改到 `8192`（同时调大 `CTX_SIZE`）；
   - 在 `extract_answer` 前先 `split('Answer:')[-1]`。
5. **Subtitle 支持**（仅 `USE_SUBTITLE=1` 时必要）：CPP 侧需在 `prefill_all_frames` 每帧前/后按 Torch 的时间窗逻辑拼 `subtitle: ...` 文本 prefill。
6. **Stack 帧支持**（仅 `STACK_NUMS>1` 时必要）：移植 `concat_images` 到 `eval_cpp_video_prep.py`，并在 prefill 前把合成帧与原始帧混合写入 JPG；同时注入那段 "mixed frame sequence" 的 system prompt。
7. **Timestamp 策略**（仅 `SAMPLE_STRATEGY=timestamp` 时必要）：移植对应的 `timestamps` 生成与 `<x seconds>` 注入逻辑。

### 8.3 诊断/调优

8. **`min_new_tokens=1`**：如果发现 CPP 经常返回空串，可在 `llama-server` 命令行或 `omni_init` 里加对应限制（当前 `llama.cpp` 侧没有直接等价参数，可改走生成后重试，即现有 `rerun_failed.py` 的兜底路径）。
9. **精度对照实验**：在同一 `MiniCPM-o-4_5` 权重上，用同一组帧分别跑 Torch bf16 和 CPP `F16 / Q8_0 / Q4_K_M`，量化差值方便后续调优。

---

## 附录：关键代码位置速查

- Torch 入口：`/cache/hanqingzhe/Video-MME/py-eval/run_videomme.py`
- Torch 评测主体：`/cache/hanqingzhe/Video-MME/py-eval/videomme.py`
  - 抽帧与策略：`videomme.py:127-217`
  - 帧拼接 `concat_images`：`videomme.py:67-120`
  - Prompt 构建：`videomme.py:258-266, 262`（CoT）、`264`（默认）
  - `model.chat` 采样参数：`videomme.py:351-366`
  - 答案判定：`videomme.py:371-403`
- CPP 配置：`/cache/hanqingzhe/Video-MME/cpp-eval/videomme/eval_cpp_config.py`
- CPP 抽帧：`/cache/hanqingzhe/Video-MME/cpp-eval/videomme/eval_cpp_video_prep.py`
- CPP HTTP 客户端：`/cache/hanqingzhe/Video-MME/cpp-eval/videomme/eval_cpp_http_client.py`
- CPP 主 pipeline：`/cache/hanqingzhe/Video-MME/cpp-eval/videomme/eval_cpp_pipeline.py`
- `llama.cpp-omni` 补丁（新增 `skip_system_prompt` 等）：`/cache/hanqingzhe/Video-MME/cpp-eval/videomme/llama-cpp-omni.patch`
