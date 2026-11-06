"""DragonLeap V4 frozen-video V2A experiment. Source repo's weights, not Foley.

No checkpoint downloads or persistent model cache in execute. Native Comfy model
management handles offload; phases explicitly release the preceding model.
"""
import gc
import time

import torch
import folder_paths
import nodes
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.model_management as mm
from comfy_extras.nodes_lt import LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent
from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEDecode
from comfy_extras.nodes_custom_sampler import BasicScheduler, CFGGuider, RandomNoise, SamplerCustomAdvanced
from .media import decode_reference, REFERENCE_FPS

FILES = {
    "model": ("diffusion_models", "LTX2/DasiwaLTX23_dragonleapV4_INT8ConvRot.safetensors"),
    "gemma": ("text_encoders", "gemma_3_12B_it_fp8_e4m3fn.safetensors"),
    "projection": ("text_encoders", "ltx-2.3_text_projection_bf16.safetensors"),
    "video_vae": ("vae", "LTX2/LTX23_video_vae_bf16.safetensors"),
    "audio_vae": ("vae", "LTX2/LTX23_audio_vae_bf16.safetensors"),
}


def release():
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def load_vae(key):
    sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise(*FILES[key]), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, dtype=torch.bfloat16, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def sample_frozen(model, positive, negative, video, audio, seed, refine):
    # zero video denoise mask means the encoded input remains fixed on EVERY
    # denoising step. This is NOT Text-to-Audio / AudioOnlyModel (which disables V2A).
    frozen = {"samples": video, "noise_mask": torch.zeros_like(video)}
    guider = CFGGuider.execute(model, positive, negative, 1.0).result[0]
    sampler = comfy.samplers.sampler_object("euler_cfg_pp")
    schedule = [(8, 1.0), (4, .4), (2, .2)] if refine else [(8, 1.0)]
    for i, (steps, denoise) in enumerate(schedule):
        mm.throw_exception_if_processing_interrupted()
        # Reattach the original video at every refinement pass, not sampled video.
        av_latent = LTXVConcatAVLatent.execute(frozen, audio).result[0]
        sigmas = BasicScheduler.execute(model, "linear_quadratic", steps, denoise).result[0]
        noise = RandomNoise.execute((seed + i) % (2**64)).result[0]
        sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, av_latent).result[0]
        audio = LTXVSeparateAVLatent.execute(sampled).result[1]
    return audio


@torch.inference_mode()
def generate(path, prompt, seed, max_side, refine):
    if not prompt.strip():
        raise ValueError("Describe the sound and the visible timing in the audio prompt")
    # Fail before decoding/loading another large model when a file is missing.
    for category, name in FILES.values():
        folder_paths.get_full_path_or_raise(category, name)
    began = time.monotonic()
    clip = vae = model = audio_vae = None
    positive = negative = video = audio = frames = None
    release()  # WAN and LTX must not compete for VRAM.
    try:
        array, info = decode_reference(path, max_side, mm.throw_exception_if_processing_interrupted)
        frames = torch.from_numpy(array).float().div_(255)
        del array
        print(f"[ltx-audio] reference {frames.shape[2]}x{frames.shape[1]} / {len(frames)} at 24 fps; source {info['duration']:.3f}s unchanged", flush=True)
        clip = nodes.DualCLIPLoader().load_clip(FILES["gemma"][1], FILES["projection"][1], "ltxv")[0]
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        # CFG=1: no negative evaluation. No misleading negative prompt widget.
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        positive, negative = LTXVConditioning.execute(positive, negative, REFERENCE_FPS).result
        clip = None
        release()
        vae = load_vae("video_vae")
        video = vae.encode_tiled(frames, tile_x=512, tile_y=512, overlap=64, tile_t=65, overlap_t=9).cpu()
        frame_count = len(frames)
        frames = vae = None
        release()
        audio_vae = load_vae("audio_vae")
        audio = LTXVEmptyLatentAudio.execute(frame_count, REFERENCE_FPS, 1, audio_vae).result[0]
        model = nodes.UNETLoader().load_unet(FILES["model"][1], "default")[0]
        audio = sample_frozen(model, positive, negative, video, audio, seed, refine)
        model = video = positive = negative = None
        release()
        decoded = LTXVAudioVAEDecode.execute(audio, audio_vae).result[0]
        decoded["waveform"] = decoded["waveform"].detach().cpu()
        print(f"[ltx-audio] completed {time.monotonic()-began:.2f}s; video not regenerated; V2A quality is experimental", flush=True)
        return decoded
    finally:
        clip = vae = model = audio_vae = positive = negative = video = audio = frames = None
        release()
