我的conda虚拟环境是vllm
数据集MME在~/Video-MME/Video-MME
现在需要你/cache/hanqingzhe/vllm-omni/test/videomme/videomme_eval_compare.md这个文档
对齐cpp版本对videomme数据集的跑分逻辑
重点包括视频处理，提示词模版，分图策略（保证no slice）
完成vllm-omni上minicpm-o-45对videomme的测评
/cache/hanqingzhe/vllm-omni/test_45_debug.py有一个minicpmo45的可运行的小demo。