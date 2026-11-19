"""Inference-optimization study for PatchVision-X and the CNN baselines (see ../README.md).

infer.models   load any of the 10 trained runs (ViT or CNN) for inference
infer.data     the CUB-200 test set, pre-processed once and cached as uint8 tensors
infer.timing   GPU latency / throughput measurement with CUDA events
infer.envinfo  records the exact hardware / software used for every benchmark
"""
