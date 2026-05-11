# vllm-omni Video-MME 评测实现计划

## 目标

在 `vllm-omni` 上跑通 `MiniCPM-o-4.5` 的 Video-MME 测评，并按 `videomme_eval_compare.md` 中 CPP 版当前默认评测逻辑对齐：

- 数据集：`~/Video-MME/Video-MME`
- 模型：`/cache/hanqingzhe/o45-pure-py`
- stage config：`test/videomme/minicpmo45_text_8x4090.yaml`（从 `vllm_omni/model_executor/stage_configs/minicpmo45_8x4090.yaml` 的 LLM stage 裁剪而来，使用 8 卡 TP 跑纯文本评测，避免初始化 TTS/T2W；`engine_output_type=text` 不返回给 TTS 用的 hidden/latent；`gpu_memory_utilization=0.35` 在 KV cache 和 64 帧 ViT 峰值显存之间折中）
- 视频：1 FPS 抽帧，最多 64 帧，超出后用 midpoint uniform sample
- prompt：基础多选模板，保留 `correct answer.Highlight` 的无空格拼接
- 图像输入：单个 video placeholder 由 vLLM MiniCPM-o 4.5 processor 展开为逐帧 image placeholder，帧间换行，`max_slice_nums=1`，`use_image_id=False`，保证 no slice
- 运行方式：单进程顺序跑完整数据，不做数据分割
- 输出：保留原始模型输出、抽取后的答案、命中情况和基础元信息，便于和 Torch/CPP 结果对照

## 文件规划

- `config.py`：集中放路径、抽帧参数、prompt 模板参数、采样参数。
- `dataset.py`：读取 parquet，标准化题目字段，支持 limit/range/断点续跑。
- `video_prep.py`：实现 decord 默认抽帧逻辑，后续可补 fallback/diagnostics。
- `prompting.py`：构造 MiniCPM-o 4.5 chat prompt 和 Video-MME 多选问题文本。
- `inference.py`：封装 `Omni` 初始化、单题调用、文本输出解析。
- `scoring.py`：答案抽取、宽松匹配、分桶统计。
- `run_vllm_omni_videomme.py`：CLI 主入口，串起数据读取、抽帧、推理、断点续跑、落盘和统计摘要。

## 第一版状态

1. 已确认 Video-MME parquet 字段：`video_id`、`duration`、`domain`、`sub_category`、`url`、`videoID`、`question_id`、`task_type`、`question`、`options`、`answer`。
2. 已实现 decord 严格抽帧路径：1 FPS 候选帧，超过 64 帧后使用 midpoint uniform sample；`--dry-run` 会打印视频帧数、fps 和采样下标。
3. 已接入 `Omni.generate`，使用 `multi_modal_data={"video": frames}`、prompt 中单个 `(<video>./</video>)`、`modalities=["text"]` 和 no-slice 参数 `max_slice_nums=1/use_image_id=False`；vLLM MiniCPM-o 4.5 的 video prompt 展开已补齐帧间换行，assistant 起始后追加 HF `enable_thinking=False` 对应的空 `<think></think>` 块，对齐 Torch `model.chat` 的非 CoT 文本评测输入。
4. 已显式传入采样参数：`max_tokens=100`、`min_tokens=1`、`temperature=0.2`、`top_p=0.8`、`top_k=100`、`repetition_penalty=1.02`。
5. 已写 JSONL 输出字段：`video_id/question_id/raw_response/raw_responses/response/answer/hit/retry_count/frame_indices`，并保留 `duration/domain/sub_category/task_type/url` 等元信息；无法抽取 `A/B/C/D` 时 `response` 为 `null`。
6. 已实现默认断点续跑：读取已有 JSONL，按 `question_id` 跳过已完成样本；`--overwrite` 可重跑覆盖。
7. 已实现无效答案一次重试，答案抽取结合 CPP 独立字母 regex 与 Python 标点切词宽松匹配。
8. 已保持 decord-only 默认路径；缺少 decord 或解码失败会显式报错，不做 ffmpeg/torchvision fallback。

## 运行命令

```bash
conda activate vllm
python test/videomme/run_vllm_omni_videomme.py --dry-run --limit 3
python test/videomme/run_vllm_omni_videomme.py
```
