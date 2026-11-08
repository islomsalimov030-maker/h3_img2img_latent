"""MiniMax H3 • Latent From Image (img2img)

True low-denoise image-to-image for MiniMax H3 still workflows.

* Takes the empty H3 audio/video latent produced by an H3 Image Studio prepare node
  (Image to Image or Reference Edit) and VAE-encodes the source image into its video stream.
* Builds a SIGMAS schedule that starts exactly at the requested noise level and spreads the
  steps evenly in the model's own (shifted) timestep space, so even with sigma shift 12 a
  denoise of 0.3 gets all its steps in the 0.3 -> 0 range.

Use with SamplerCustomAdvanced (noise + guider + sampler + these sigmas + this latent).
denoise is the real flow-matching noise fraction: 0.3 = 30% noise mixed into the source.
"""
import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.nested_tensor

__version__ = "0.2.0"
DENSE = 1000


class H3LatentFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_latent": ("LATENT", {"tooltip": "Empty H3 AV latent from an Image Studio prepare node."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "image": ("IMAGE", {"tooltip": "Source image (first image of the batch is used)."}),
                "model": ("MODEL", {"tooltip": "Same model the sampler uses (after sigma shift / LoRA)."}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 200,
                                  "tooltip": "Steps spent between the start noise level and 0."}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01,
                                      "tooltip": "Noise fraction mixed into the source. 0.2 subtle, 0.35 face fix, 0.6+ strong."}),
                "always_encode": ("BOOLEAN", {"default": False,
                    "tooltip": "Encode init into target video even at denoise=1. References stay in conditioning."}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "STRING")
    RETURN_NAMES = ("h3_latent", "sigmas", "info")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/img2img"

    @staticmethod
    def _schedule(model, scheduler, steps, denoise):
        dense = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, DENSE).cpu().float()
        target = float(denoise) * float(dense[0])
        k = int((dense > target).sum().item())          # first index with sigma <= target
        k = max(0, min(k, DENSE - 1))
        idx = torch.linspace(k, DENSE, steps + 1).round().long().clamp(0, DENSE)
        idx = torch.unique_consecutive(idx)
        sig = dense[idx]
        sig[-1] = 0.0
        return sig

    def run(self, h3_latent, vae, image, model, steps, scheduler, denoise,
            always_encode=False, mask=None):
        sigmas = self._schedule(model, scheduler, steps, denoise)
        out = dict(h3_latent)
        if denoise >= 0.999 and not always_encode:
            if mask is not None:
                out["noise_mask"] = self._mask(h3_latent["samples"], mask)
            return (out, sigmas, f"denoise 1.0 -> full generation from noise ({len(sigmas)-1} steps)")

        samples = h3_latent["samples"]
        nested = hasattr(samples, "tensors")
        video = samples.tensors[0] if nested else samples
        b, c, t, h, w = video.shape

        img = image[:1, :, :, :3].movedim(-1, 1)
        img = F.interpolate(img, size=(h * 16, w * 16), mode="bicubic", align_corners=False, antialias=True)
        img = img.clamp(0, 1).movedim(1, -1)

        frames = int(h3_latent.get("h3_natural_frames", 1 + 4 * (t - 1)))
        enc = vae.encode(img.repeat(max(1, frames), 1, 1, 1))
        if enc.ndim == 4:
            enc = enc.unsqueeze(2)
        if enc.shape[2] != t:
            enc = enc[:, :, :1].repeat(1, 1, t, 1, 1)
        enc = enc[:, :c].to(device=video.device, dtype=video.dtype)
        if enc.shape[0] != b:
            enc = enc[:1].repeat(b, 1, 1, 1, 1)

        out["samples"] = comfy.nested_tensor.NestedTensor([enc] + list(samples.tensors[1:])) if nested else enc
        info = (f"denoise {denoise:.2f}: {len(sigmas)-1} steps, sigmas "
                + ", ".join(f"{float(s):.3f}" for s in sigmas)
                + f" | latent {tuple(enc.shape)}")
        if mask is not None:
            out["noise_mask"] = self._mask(out["samples"], mask)
        if always_encode:
            shapes = [tuple(x.shape) for x in out["samples"].tensors] if nested else [tuple(enc.shape)]
            mask_shapes = ([tuple(x.shape) for x in out["noise_mask"].tensors]
                           if mask is not None and nested else
                           tuple(out["noise_mask"].shape) if mask is not None else None)
            # Core 0.34.6/current: references are minimax_refs in CONDITIONING,
            # never temporal slots in samples. All video slots are targets.
            info += (f" | version {__version__}; samples={shapes}; target_frames={list(range(t))}; "
                     f"reference_frames=[] (conditioning minimax_refs, not samples); "
                     f"cell_px={1344 / w:g}; noise_mask={mask_shapes}")
        return (out, sigmas, info)

    @staticmethod
    def _mask(samples, mask):
        nested = hasattr(samples, "tensors")
        streams = samples.tensors if nested else [samples]
        video = streams[0]
        if video.ndim != 5:
            raise ValueError("Expected H3 target video [B,C,T,H,W]; unknown layout, stop")
        b, c, t, h, w = video.shape
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] not in (1, b):
            raise ValueError("mask must be [H,W] or [1|B,H,W]")
        spatial = F.interpolate(mask[:, None].float(), size=(h, w), mode="area")
        spatial = spatial.to(device=video.device, dtype=video.dtype).clamp(0, 1)
        video_mask = spatial[:, :, None].expand(b, c, t, h, w).clone()
        if nested:
            return comfy.nested_tensor.NestedTensor([video_mask] + [torch.zeros_like(x) for x in streams[1:]])
        return video_mask


NODE_CLASS_MAPPINGS = {"H3LatentFromImage": H3LatentFromImage}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LatentFromImage": "MiniMax H3 • Latent From Image (img2img)"}
