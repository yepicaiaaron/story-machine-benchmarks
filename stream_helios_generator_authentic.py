import asyncio
import os
import cv2
import numpy as np
import torch
from livekit import rtc, api
import time
import sys
import threading
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Helios')))

LK_URL = "wss://chatgptme-sp76gr03.livekit.cloud"
LK_API_KEY = "APIRBVfLnF2B2WF"
LK_API_SECRET = "cEqJAr8wQHkRSZrM7o8oHY2HSguTn54gC5XBIxAs3pF"
ROOM_NAME = "helios-dedicated-stream"

MODEL_PATH = "BestWishYsh/Helios-Distilled"
NUM_FRAMES = 240
HEIGHT = 384
WIDTH = 640
FPS = 24

PROMPTS = [
    "A cinematic low-angle tracking shot of a cyberpunk detective walking down a neon-lit alleyway in the rain, 8k, photorealistic.",
    "A dynamic drone shot sweeping over a futuristic sci-fi metropolis with flying cars and massive holographic advertisements, vivid colors, 4k."
]

frame_queue = asyncio.Queue(maxsize=100)

async def frame_streamer(source):
    print("Frame streamer task started.")
    while True:
        frame_rgb = await frame_queue.get()
        rtc_frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGB24, frame_rgb.tobytes())
        source.capture_frame(rtc_frame)
        await asyncio.sleep(1/FPS)

class AuthenticRecorder:
    def __init__(self, filename, fps=24):
        self.filename = filename
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.process = subprocess.Popen([
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{WIDTH}x{HEIGHT}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-pix_fmt', 'yuv420p', filename
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last_write_time = time.time()
        self.first_frame = True
        self.last_frame_bgr = None

    def add_frames(self, frames_bgr, current_fps_str=""):
        current_time = time.time()
        
        # Calculate gap before writing new chunk
        if not self.first_frame:
            elapsed = current_time - self.last_write_time
            num_gaps = int(elapsed / self.frame_interval)
            # Max duplicate 240 frames (10 secs) to prevent gigantic files if it hangs completely
            if num_gaps > 240: num_gaps = 240
            
            if num_gaps > 0 and self.last_frame_bgr is not None:
                for _ in range(num_gaps):
                    try:
                        self.process.stdin.write(self.last_frame_bgr.tobytes())
                    except Exception:
                        pass
        
        # Write the new chunk
        for frame in frames_bgr:
            f = frame.copy()
            if current_fps_str:
                cv2.putText(f, current_fps_str, (WIDTH - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            try:
                self.process.stdin.write(f.tobytes())
            except Exception:
                pass
            self.last_frame_bgr = f

        self.first_frame = False
        chunk_play_time = len(frames_bgr) * self.frame_interval
        # the chunk finishes playing after chunk_play_time
        self.last_write_time = current_time + chunk_play_time

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None

def generation_worker(loop, room):
    print("Loading Helios Pipeline in worker thread...")
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
    from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel

    transformer = HeliosTransformer3DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")

    scheduler = HeliosScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )

    pipeline = HeliosPipeline.from_pretrained(
        MODEL_PATH, transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Compiling transformer for max speed...")
    pipeline.transformer = torch.compile(pipeline.transformer, backend="inductor", mode="max-autotune")

    step_setting = [2, 2, 2] # WE MUST EDIT THIS FOR DIFFERENT RUNS
    step_str = "_".join(map(str, step_setting))

    # Run only once through prompts to save exactly what is requested
    for idx, prompt in enumerate(PROMPTS):
        print(f"Generating video for prompt: '{prompt}'...")
        
        try:
            data = prompt.encode('utf-8')
            asyncio.run_coroutine_threadsafe(
                room.local_participant.publish_data(payload=data, topic="prompt"), loop
            )
        except Exception as e:
            print(f"Failed to send prompt data: {e}")
        
        generator = pipeline(
            prompt=prompt, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
            guidance_scale=1.0, is_enable_stage2=True,
            pyramid_num_inference_steps_list=step_setting,
            is_amplify_first_chunk=True, output_type="pt"
        )
        
        prompt_letter = "A" if idx == 0 else "B"
        mp4_filename = f"benchmark_hardware_{step_str}_{prompt_letter}.mp4"
        recorder = AuthenticRecorder(mp4_filename, fps=FPS)
        
        try:
            start_time = time.time()
            for current_video_chunk in generator:
                chunk_time = time.time() - start_time
                fps = current_video_chunk.shape[2] / chunk_time if chunk_time > 0 else 0
                print(f"Rendered chunk of size {current_video_chunk.shape} at {fps:.2f} FPS! Streaming...")
                start_time = time.time()
                
                fps_str = f"FPS: {fps:.2f}"
                try:
                    fps_data = fps_str.encode('utf-8')
                    asyncio.run_coroutine_threadsafe(
                        room.local_participant.publish_data(payload=fps_data, topic="fps"), loop
                    )
                except Exception as e:
                    pass
                
                chunk_np = current_video_chunk.to(dtype=torch.float32).cpu().numpy()
                chunk_np = (chunk_np + 1.0) / 2.0
                chunk_np = np.clip(chunk_np, 0.0, 1.0)
                if chunk_np.ndim == 5:
                    chunk_np = chunk_np[0] 
                chunk_np = np.transpose(chunk_np, (1, 2, 3, 0))
                
                frames_bgr = []
                for i in range(chunk_np.shape[0]):
                    frame = chunk_np[i]
                    frame_bgr = (frame * 255).astype(np.uint8)[:, :, ::-1]
                    frames_bgr.append(frame_bgr)
                    
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    asyncio.run_coroutine_threadsafe(frame_queue.put(frame_rgb), loop).result()
                    
                recorder.add_frames(frames_bgr, fps_str)
                    
        except StopIteration as e:
            pass
        except Exception as e:
            print(f"Exception during streaming generation: {e}")
        finally:
            recorder.close()
            print(f"Finished writing {mp4_filename}")
            
    print("FINISHED ALL PROMPTS.")
    sys.exit(0)

async def main():
    token = api.AccessToken(LK_API_KEY, LK_API_SECRET) \
        .with_identity("helios-streamer") \
        .with_name("Helios Streamer") \
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME)) \
        .to_jwt()

    room = rtc.Room()
    print(f"Connecting to dedicated LiveKit room '{ROOM_NAME}'...")
    await room.connect(LK_URL, token)
    print("Connected to LiveKit.")

    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("helios-video", source)
    
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_CAMERA
    publication = await room.local_participant.publish_track(track, options)
    print("Published video track.")

    asyncio.create_task(frame_streamer(source))
    threading.Thread(target=generation_worker, args=(asyncio.get_running_loop(), room), daemon=True).start()

    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
