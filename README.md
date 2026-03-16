# Story Machine Benchmarks

Welcome to the official **Story Machine Benchmarks** repository. This project serves as a comprehensive quality and throughput comparison for the **Helios Real-Time Long Video Generation model**, specifically focusing on its performance under different inference step configurations on high-end hardware.

For the full interactive visual comparison, please visit the live webpage:
👉 **[View the Live Benchmark Matrix](https://story-machine-benchmarks.onrender.com/)**

## The Models

The Helios models were developed by the **PKU-YuanGroup** (Peking University) in early 2025. Helios is an advanced DiT (Diffusion Transformer) video generation architecture introduced to provide fast, high-quality, and temporally consistent long video synthesis. 

There are three distinct variants of the Helios model, each optimized for different use cases:

*   **Helios-Base:** The standard foundational model. It offers strong visual fidelity and temporal consistency but requires significant compute power to run effectively.
*   **Helios-Mid:** A balanced middle-ground model that slightly reduces parameter counts to improve generation speeds while attempting to maintain structural integrity.
*   **Helios-Distilled (Tested Here):** The highly optimized, fast-inference model. This version was aggressively distilled from the Base model to dramatically reduce the required inference steps for generation. It forms the basis for the PKU-YuanGroup's real-time video generation claims. 

**All tests in this benchmark matrix were performed using the Helios-Distilled model** (`BestWishYsh/Helios-Distilled`) to replicate the fastest possible configuration cited in the research paper.

## Why Real-World FPS Differs from Paper Claims

Based on our analysis and testing of the Helios-Distilled architecture, there are a few major reasons why the reported numbers in the research paper (1.58s / ~20.8 FPS) are faster than our pure PyTorch hardware readout (~2.39s / ~13.8 FPS) on the exact same NVIDIA H100 hardware:

1. **They benchmarked pure T2V (Text-to-Video), not I2V streaming:** Their benchmark measures just the raw transformer generating latents from text. Our pipeline is running a continuous autoregressive stream where it has to pass the previous chunk's frames back into the model as an Image-to-Video (I2V) conditioning step to maintain consistency across chunks. That extra conditioning pass adds significant overhead.
2. **They didn't include VAE decoding overhead:** Their 19.5+ FPS benchmark only timed the latent generation. In our streaming pipeline, we are physically decoding those latents back into full 720p RGB pixels on the GPU so that we can actually see and stream them. The VAE decode step for 33 frames of 720p video is mathematically heavy.
3. **We are running Stage 2 (Upscaling/Refinement):** In our `stream_helios_generator_authentic.py` script, we have `is_enable_stage2=True`. They likely disabled the second stage of the generation pipeline for their raw speed-test benchmark to hit that sub-1.6s number.
4. **Python I/O and Conversion Overhead:** Our script isn't just generating the tensor; it's converting the `bfloat16` PyTorch tensor to `float32`, pulling it from GPU VRAM to CPU RAM (`.cpu().numpy()`), and reshaping/scaling the array so we can feed it into LiveKit or `ffmpeg`. Moving a 33-frame 720p uncompressed tensor from GPU to CPU takes non-trivial millisecond time.

**Essentially, they benchmarked the absolute bare-minimum matrix multiplication inside the Transformer. We are benchmarking the actual, usable pipeline that produces pixels you can look at.**

## Replicating the Results

The tests were run natively on a single **NVIDIA H100 80GB HBM3** GPU on Google Cloud Platform (GCP). 

To bypass browser WebRTC limitations, UI lag, and network latency to capture the *authentic hardware throughput*, we built a custom Python script that intercepts the raw PyTorch frames and dynamically encodes them directly to an MP4 on the server using `ffmpeg`. It programmatically duplicates frames to physically bake the exact generation delay into the video file and overlays the native hardware FPS via OpenCV.

The script used to generate these authentic benchmark videos is included in this repository: `stream_helios_generator_authentic.py`.

### Requirements
*   NVIDIA H100 (or equivalent GPU with at least 48GB+ VRAM)
*   PyTorch (compiled with `inductor` backend, `mode="max-autotune"`)
*   `ffmpeg` installed on the host machine
*   OpenCV (`cv2`) for the FPS overlay

### Execution
Run the included python script to automatically generate the raw performance videos:
```bash
# Requires an active HuggingFace token for model weights
export HF_TOKEN="your_hf_token"
python3 stream_helios_generator_authentic.py
```
*(Note: Changing inference step settings requires a full 15-minute `torch.compile` optimization phase before generation begins).*

---
For visual evidence of these findings, please visit the **[Live Benchmark Page](https://story-machine-benchmarks.onrender.com/)**.