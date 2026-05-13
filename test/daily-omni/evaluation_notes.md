# Daily-Omni vLLM-Omni 测评要点

## 目标

Daily-Omni 应作为多选题形式的音视频推理 benchmark 来测评。本地索引文件是 `/cache/hanqingzhe/daily-omni/daily_omni.jsonl`，每一行都会指向数据集根目录下的一段视频和一个 wav 音频文件。

这套骨架沿用现有 `test/videomme` 的结构，但 Daily-Omni 有一个关键差异：官方 MiniCPM-o 4.5 torch 测评路径使用音频+视频输入，而不是纯视频输入。

## 数据集格式

当前每条 JSONL 记录包含：

- `dataset_type`：预期为 `mcq`。
- `dataset_name`：预期为 `daily_omni`。
- `question`：题干文本。
- `choices`：四个选项字符串，已经带有 `A.`、`B.`、`C.`、`D.` 前缀。
- `gt_answer`：标准答案字母。
- `VideoPath`：相对数据集根目录的视频路径。
- `WavPath`：相对数据集根目录的音频路径。
- 分组元数据：`qa_type`、`content_parent_category`、`content_fine_category`、`video_category`、`video_duration`、`video_id`。

TODO：确认 `Videos.tar` 完整解压后，再重新检查媒体文件完整性。

## 需要对齐的 Torch Evalkit 行为

torch 测评入口是 `/cache/hanqingzhe/Video-MME/evalkit/eval_main.py`，数据集调度在 `o_e_Kit/utils/evaluation_runner.py`，MiniCPM-o 封装在 `o_e_Kit/models/minicpm/minicpmo_ou.py`。

关键点：

- Daily-Omni 通过 `OmniEvalDataset` 加载，最后用多选题评测器 `MQAEvaluator` 计分。
- 官方 Daily-Omni 脚本使用 `o_e_Kit/configs/omni_generation_configs_nosys_interleave.json`，不是更短的默认配置。
- Daily-Omni 配置使用 `max_tokens: 128`、`max_frames: 64`、`max_fps: 1.0`、`load_av: true`、`interleave_fps: 1.0`。
- prompt 内容如下：

```text
{media}
Carefully read the following question and select the letter corresponding to the correct answer.Highlight the applicable choices without giving explanations.
{question}
Options:
{options}
Please select the correct answer from the options above. Only respond with the letter.
```

- 在 vLLM-Omni 侧还需要在 assistant 开头插入空 thinking 块，即 `<think>\n\n</think>\n\n`，用于显式保持 no-thinking 行为。这一点已在 `prompting.py` 的 `NO_THINKING_SUFFIX` 中处理。
- 对 `all` / `audio` 模式，还需要走 TTS template 起始 token。不要在手写 prompt 里单独插 `<|tts_bos|>`；vLLM-Omni 的 MiniCPM-o processor 支持 `mm_processor_kwargs["use_tts"]`，会自动追加完整前缀 `<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>`。
- `all` 模式应对齐 torch 的 `omni_mode=True`：把视频抽帧作为多张 `image`，把音频按抽帧时间戳切成同数量片段，再构造 `<image><audio><image><audio>...` 的交错占位符序列。当前 `media_prep.py` 与 `prompting.py` 已按该方式实现。
- Torch 调用形态是 `model.chat(..., sampling=False/do_sample=False 等价行为, max_new_tokens=128, use_tts_template=True, omni_mode=True for interleaved content)`。
- 在 `~/o45-pure-py/modeling_minicpmo.py` 中，图像帧会变成 `<image>./</image>`，音频数组会变成 `<audio>./</audio>`，`omni_mode=True` 会把这些占位符直接拼接，中间不额外插入换行。
- 在 `~/o45-pure-py/utils.py` 中，OpenAI 风格的 `video_url` 内容会被归一化为交错列表：视频帧、音频片段、视频帧、音频片段，依此类推。

## 视频抽帧

Evalkit 优先尝试 decord，然后回退到 ffmpeg，最后回退到 torchvision。

decord 路径大约按 1 FPS 抽帧，然后用 midpoint uniform sampling 限制到 `max_frames=64`。torch helper 在不同后端中对 `max_fps` 的处理并不完全一致，所以优先用 decord 对齐；如果后续加入 fallback，需要把差异记录清楚。

TODO：增加可选的 parity 诊断模式，记录抽帧索引，并在少量媒体样本上与 torch evalkit 对比。

## 音频处理

Daily-Omni 提供独立 wav 路径。Torch 以 16 kHz 单声道加载音频；在交错视频输入中，会把音频切成与视频帧对齐的片段。

vLLM-Omni 侧 MiniCPM-o 音频/视频输入形态：

- `all`：`multi_modal_data["image"]` 放抽样视频帧，`multi_modal_data["audio"]` 放按相邻帧时间戳切出的音频片段，prompt 中按 `<image><audio>` 交错。
- `visual`：`multi_modal_data["video"]` 放视频帧列表，prompt 中使用单个 `<video>` 占位符，由 processor 展开为逐帧图像占位符。
- `audio`：`multi_modal_data["audio"]` 放整段音频，prompt 中使用单个 `<audio>` 占位符。

TODO：用真实模型推理验证 `all` 模式下多段 audio replacement 与 torch evalkit 的 token 序列是否完全一致。

## 生成与评分

为了可比性，使用 greedy decoding：

- `temperature=0.0`
- `top_p=1.0`
- `top_k=-1`
- `max_tokens=128`
- 不使用随机采样
- assistant prompt 前缀保留空 `<think></think>`，避免模型进入 thinking 输出。
- `all` / `audio` 模式传 `use_tts=True`，由 vLLM-Omni processor 自动补完整 TTS 前缀。

输出应为单个答案字母。骨架复用现有 Video-MME 的答案抽取方式，并写出 JSONL 以及按以下字段分组的 summary JSON：

- `qa_type`
- `content_parent_category`
- `content_fine_category`
- `video_category`
- `video_duration`

TODO：验证 vLLM 输出形态后，增加一个可选转换器，把结果转成 evalkit 的 `{job_id, predictions, dataset_name, model_name, timestamp}` JSON 格式，从而让官方 `MQAEvaluator` 评分同一批记录。
