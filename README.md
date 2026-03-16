# Story Machine Benchmarks - Model Quality Comparison

Welcome to the official **Story Machine Benchmarks** repository. This page serves as a comprehensive quality and throughput comparison for the Helios Real-Time Long Video Generation model, specifically focusing on its performance under different inference step configurations on high-end hardware.

## The Model: Helios-Distilled

All tests in this benchmark matrix were performed using the **Helios-Distilled** model (`BestWishYsh/Helios-Distilled`), developed by the PKU-YuanGroup. Helios is an advanced DiT (Diffusion Transformer) video generation architecture introduced to provide fast, high-quality video synthesis. The "Distilled" version has been optimized for faster inference, forming the basis for real-time video generation claims.

## The Benchmark Matrix

The videos provided in the `assets/` directory demonstrate the direct trade-off between visual fidelity and real-time generation speed. We ran the model under two distinct configurations using the same set of cinematic prompts:

*   **`[2, 2, 2]` Inference Steps (High Quality):** The default configuration intended for maximum visual quality. While the output is visually impressive and highly detailed, the rendering overhead introduces significant stuttering when streamed live.
*   **`[1, 1, 1]` Inference Steps (High Speed):** The speed-test configuration used to achieve higher framerates (e.g., 19.5+ FPS). As shown in the videos, dropping the inference steps this low fundamentally degrades the visual structure, resulting in a blurry, incoherent output.

### Prompts Tested:
*   **Prompt A:** "A cinematic low-angle tracking shot of a cyberpunk detective walking down a neon-lit alleyway in the rain, 8k, photorealistic."
*   **Prompt B:** "A dynamic drone shot sweeping over a futuristic sci-fi metropolis with flying cars and massive holographic advertisements, vivid colors, 4k."

*Videos are available in the `assets/` folder of this repository, representing the raw, pure hardware throughput without UI or network latency.*

## Why Real-World FPS Differs from Paper Claims

Based on our analysis and testing of the Helios architecture, there are a few major reasons why the reported numbers in the research paper (1.58s / ~20.8 FPS) are faster than our pure PyTorch readout (~2.39s / ~13.8 FPS) on the exact same H100 hardware:

1. **They benchmarked pure T2V (Text-to-Video), not I2V streaming:** Their benchmark measures just the raw transformer generating latents from text. Our pipeline is running a continuous autoregressive stream where it has to pass the previous chunk's frames back into the model as an Image-to-Video (I2V) conditioning step to maintain consistency across chunks. That extra conditioning pass adds significant overhead.
2. **They didn't include VAE decoding overhead:** Their 19.5+ FPS benchmark only timed the latent generation. In our streaming pipeline, we are physically decoding those latents back into full 720p RGB pixels on the GPU so that we can actually see and stream them. The VAE decode step for 33 frames of 720p video is mathematically heavy.
3. **We are running Stage 2 (Upscaling/Refinement):** In our `stream_helios_generator_authentic.py` script, we have `is_enable_stage2=True`. They likely disabled the second stage of the generation pipeline for their raw speed-test benchmark to hit that sub-1.6s number.
4. **Python I/O and Conversion Overhead:** Our script isn't just generating the tensor; it's converting the `bfloat16` PyTorch tensor to `float32`, pulling it from GPU VRAM to CPU RAM (`.cpu().numpy()`), and reshaping/scaling the array so we can feed it into LiveKit or `ffmpeg`. Moving a 33-frame 720p uncompressed tensor from GPU to CPU takes non-trivial millisecond time.

**Essentially, they benchmarked the absolute bare-minimum matrix multiplication inside the Transformer. We are benchmarking the actual, usable pipeline that produces pixels you can look at.**

## Replicating the Results

The tests were run natively on an **NVIDIA H100 80GB HBM3** GPU. To bypass browser WebRTC limitations and capture the authentic hardware throughput, we built a Python script that intercepts the raw PyTorch frames and dynamically encodes them directly to an MP4 using `ffmpeg`, correctly duplicating frames to perfectly match the true generation delay.

The script used to generate these videos is included in this repository: `stream_helios_generator_authentic.py`.

### Requirements
*   NVIDIA H100 (or equivalent GPU with at least 48GB+ VRAM)
*   PyTorch (compiled with `inductor` backend, `mode="max-autotune"`)
*   `ffmpeg` installed on the host machine
*   OpenCV (`cv2`) for the FPS overlay
