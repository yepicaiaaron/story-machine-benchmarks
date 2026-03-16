# 🎥 Story Machine Benchmarks — Real-Time Video Generation

<p align="center">
    <picture>
        <img src="https://img.shields.io/badge/Status-Live-4ade80?style=for-the-badge" alt="Status Live">
    </picture>
</p>

<p align="center">
  <strong>DIFFUSE! DIFFUSE!</strong>
</p>

<p align="center">
  <a href="https://yepicaiaaron.github.io/story-machine-benchmarks/"><img src="https://img.shields.io/badge/View_Gallery-Live-blue?style=for-the-badge" alt="View Gallery"></a>
  <a href="https://github.com/StoryMachineAI/story-machine-ai"><img src="https://img.shields.io/badge/Main_Repo-GitHub-white?style=for-the-badge&logo=github" alt="Main Repo"></a>
</p>

**Story Machine Benchmarks** is the ongoing tracking project for testing the mathematical and visual limits of real-time AI video generation models on high-end hardware (NVIDIA H100). 
It provides an interactive, side-by-side visual matrix to compare model outputs as compute constraints are scaled up and down.

**This project builds upon the foundational real-time acceleration pipeline engineering pioneered in [Helios-Unleashed](https://github.com/yepicaiaaron/Helios-Unleashed).** By translating their 3-Stage Async Pipeline, NVENC zero-copy integration, and RIFE optical flow architectures into our live WebRTC dashboard, we are able to consistently document the edge of what's possible for live video synthesis.

If you want to understand the true trade-off between framerate and visual fidelity in modern video diffusion pipelines, this is the place.

[View Interactive Gallery](https://yepicaiaaron.github.io/story-machine-benchmarks/) · [Core Story Machine Repo](https://github.com/StoryMachineAI/story-machine-ai)

Preferred testing hardware: **NVIDIA H100 80GB** (to support high-step diffusion passes without instantly crushing the real-time threshold).

## Highlights

- **[Interactive Side-by-Side Player](https://yepicaiaaron.github.io/story-machine-benchmarks/)** — Play any two benchmark clips perfectly synchronized to compare structural coherence over time.
- **[Real-Time vs Cinematic Trade-offs](#real-time-diffusion-trade-offs)** — Analyze the exact impact of modifying `num_inference_steps` on prompt adherence and frame rates.
- **[RIFE Upscaling Impact](#rife-upscaling-impact)** — Visual proof of how intermediate flow estimation can artificially boost an 11 FPS raw render into a buttery-smooth 36 FPS stream.
- **[Full Matrix Rendering](#full-matrix)** — Comprehensive tests across multiple resolutions (512x288, 848x480) and inference steps (2, 3, 4, 6, 8, 10, 12).

## The Core Challenge: Real-Time Video

### Real-Time Diffusion Trade-offs (Steps vs. Adherence)
We discovered a hard mathematical trade-off on the H100 GPU concerning the `num_inference_steps` parameter and prompt adherence when running the **MemFlow (Wan2.1-based)** pipeline:
- **Low Steps (e.g., 2-4 Steps at 512x288):** Yields a highly fluid stream (~35-50 FPS raw). However, the model rushes the global structure. It ignores complex geometric constraints (like "low-angle tracking shot") and hallucinates significantly, looking more like chaotic static.
- **High Steps (e.g., 20 Steps at 576x320):** Native pipeline bottlenecks massively, breaking the stream entirely and causing severe stutter.
- **The Sweet Spot (12 Steps at 512x288):** Our final compromise for high-quality generation. By lowering the native resolution slightly to 512x288, the H100 gains the compute leeway to run 12 full diffusion passes. This 50% increase in refinement time allows the model to faithfully adhere to complex camera angles and stylistic constraints while keeping the frame generation stable.

### RIFE Upscaling Impact
**RIFE (Real-Time Intermediate Flow Estimation)** is an AI model used for Video Frame Interpolation. Instead of forcing the heavy diffusion model (MemFlow) to render every single frame of a 30 FPS video—which takes massive compute—MemFlow generates a low-framerate base video (e.g., 6-11 FPS). RIFE then acts as a post-processor, intelligently hallucinating and inserting the "in-between" frames to artificially triple the framerate.

**The Impact:** This achieves buttery-smooth cinematic playback while freeing up the H100 GPU to spend more compute time on the base frames (running higher inference steps for better prompt adherence).
- Example: A 512x288 video running at 4 inference steps natively outputs **~11.0 FPS**. When processed through RIFE, it outputs a smooth **~36.7 FPS**.

## How the Benchmarks are Generated

```
LiveKit DataChannel
       │
       ▼
┌───────────────────────────────┐
│     H100 Daydream Proxy       │
│      (WebRTC Endpoint)        │
└──────────────┬────────────────┘
               │
               ├─ Force reset_cache: True
               ├─ Map inference step arrays
               └─ Render 20-second contiguous clip
```

To ensure the benchmarks are authentic and unbiased:
1. **Cache Nuking:** We explicitly pass `"reset_cache": True` into the `initialParameters` object for every single run. This forces the GPU to throw away its memory (latent state) and start completely over from pure static noise, preventing it from just continuing a previously rendered high-quality scene.
2. **Explicit Parameter Mapping:** The `daydream-scope` backend maps inference steps to a mathematical array of noise schedules (e.g., `denoising_step_list: [1000, 750, 500, 250]` for 4 steps). We meticulously calculated and enforced these arrays for every permutation to guarantee the GPU actually obeyed the step count rather than falling back to defaults.

## Master Performance Matrix (MemFlow on H100)

*Note: All clips generated natively on the GPU for exactly 20.0 seconds. FPS drops linearly as inference steps increase due to the added diffusion compute time per frame.*

### Prompt A: Cyberpunk Detective
*(A cinematic low-angle tracking shot of a cyberpunk detective walking down a neon-lit alleyway in the rain, 8k, photorealistic.)*

| Resolution | Steps | Render Time | Average Stream FPS |
| ---------- | ----- | ----------- | ------------------ |
| 512 x 288 | 2 | 20.0s | 49.3 FPS |
| | 3 | 20.0s | 38.9 FPS |
| | 4 | 20.0s | 36.7 FPS |
| | 6 | 20.0s | 27.5 FPS |
| | 8 | 20.0s | 22.1 FPS |
| | 10 | 20.0s | 18.8 FPS |
| | 12 | 20.0s | 16.3 FPS |
| 848 x 480 | 2 | 20.0s | 28.3 FPS |
| | 3 | 20.0s | 24.3 FPS |
| | 4 | 20.0s | 20.6 FPS |
| | 6 | 20.0s | 16.5 FPS |
| | 8 | 20.0s | 13.3 FPS |
| | 10 | 20.0s | 11.3 FPS |
| | 12 | 20.0s | 9.3 FPS |

### Prompt B: Drone Shot Metropolis
*(A dynamic drone shot sweeping over a futuristic sci-fi metropolis with flying cars and massive holographic advertisements, vivid colors, 4k.)*

| Resolution | Steps | Render Time | Average Stream FPS |
| ---------- | ----- | ----------- | ------------------ |
| 512 x 288 | 2 | 20.0s | 47.2 FPS |
| | 3 | 20.0s | 40.5 FPS |
| | 4 | 20.0s | 34.8 FPS |
| | 6 | 20.0s | 27.1 FPS |
| | 8 | 20.0s | 23.0 FPS |
| | 10 | 20.0s | 19.3 FPS |
| | 12 | 20.0s | 16.7 FPS |
| 848 x 480 | 2 | 20.0s | 28.7 FPS |
| | 3 | 20.0s | 23.9 FPS |
| | 4 | 20.0s | 20.5 FPS |
| | 6 | 20.0s | 16.4 FPS |
| | 8 | 20.0s | 13.3 FPS |
| | 10 | 20.0s | 11.1 FPS |
| | 12 | 20.0s | 9.4 FPS |

---

*This project is actively maintained to support the development of the Story Machine AI platform.*