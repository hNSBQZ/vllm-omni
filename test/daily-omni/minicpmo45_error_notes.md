# MiniCPM-o 4.5 Daily-Omni 问题记录

## 现象

运行 `test/daily-omni/run_vllm_omni_daily_omni.py --input-mode all` 时，日志
`test/daily-omni/outputs/vllm_omni_minicpmo45_daily_omni.log` 中报
`CUDA error: device-side assert triggered`。原始栈顶可能落在
`minicpmo_4_5_omni.py` 的 `torch.zeros(...)` 或后续 Qwen/FlashAttention
forward，但这是 CUDA 异步报错后的表象。

开启 `CUDA_LAUNCH_BLOCKING=1` 复现后，真实触发点是 vLLM 的
`_merge_multimodal_embeddings`：

```text
masked_scatter_size_check: Assertion `totalElements <= srcSize` failed
```

## 根因

Daily-Omni 的 `all` 模式把视频帧作为多张 `image` 输入，并通过
`mm_processor_kwargs` 传入：

```python
{"max_slice_nums": 1, "use_image_id": False}
```

实际图像处理会按 `max_slice_nums=1` 生成每张图 64 个视觉 embedding。
但 MiniCPM-o 4.5 的 `get_image_replacement()` 在构造 prompt replacement
时没有把这两个参数传给 `get_slice_image_placeholder()`，导致占位符仍按默认
切片/image id 规则展开。复现日志中每张 image 的 placeholder 长度约为
201/202，其中 `<unk>` 数量明显大于实际 64 个 image embedding。

因此 vLLM 在把多模态 embedding scatter 回文本 embedding 时，placeholder
数量大于 source embedding 数量，触发 CUDA device-side assert。

## 修复

在 `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_llm.py`
中让 image prompt replacement 复用同一份 `hf_processor_mm_kwargs`：

- `max_slice_nums` 传给 `get_slice_image_placeholder()`，保证 prompt 中
  `<unk>` 数量和实际视觉 embedding 数量一致。
- `use_image_id` 同步传入，避免 Daily-Omni `use_image_id=False` 时仍插入
  image id token。

## 验证

用 `CUDA_LAUNCH_BLOCKING=1` 跑 `--limit 1 --input-mode all` 可用于验证
该问题是否消失。修复前同一条样本在首轮 prefill 期间失败，且 dump 中
`num_scheduled_tokens=6538`，image placeholder 被过度展开。

修复后已使用以下命令验证单样本可以正常完成，并生成
`test/daily-omni/outputs/vllm_omni_minicpmo45_debug_limit1_fixed.jsonl`：

```bash
PYTHONUNBUFFERED=1 /cache/hanqingzhe/miniconda3/envs/vllm/bin/python -u \
  test/daily-omni/run_vllm_omni_daily_omni.py \
  --limit 1 \
  --input-mode all \
  --stage-config test/daily-omni/minicpmo45_text_8x4090.yaml \
  --output test/daily-omni/outputs/vllm_omni_minicpmo45_debug_limit1_fixed.jsonl \
  --overwrite
```
