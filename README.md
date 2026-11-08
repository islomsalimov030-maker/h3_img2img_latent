# H3 Latent From Image 0.2.0

MIT. New widget `always_encode` defaults to False. Default behavior without a
mask is unchanged, including the denoise>=.999 early return and info strings.
With True, init is VAE-encoded at every denoise, including 1; sigmas unchanged.

Layout verified against ComfyUI v0.34.6 and current core:
`MiniMaxH3ReferenceToVideo` outputs NestedTensor(video,audio). Video samples
[B,24,T,H/16,W/16] contain only target frames 0..T-1. References are separate
`minimax_refs` blocks in CONDITIONING, not frame slots in samples.

Optional `mask` is area-resized to H_lat/W_lat and expanded over all target
channels/frames. Returned noise_mask matches video/audio shapes; all non-video
streams are zero. References in CONDITIONING are never modified. Do not add
SetLatentNoiseMask after this node.

Info with always_encode: version, sample shapes, target/reference indices,
cell_px=1344/W_lat, noise-mask shapes. Reference indices in samples are []
because core conditions references separately. Unknown layouts fail explicitly.

Zero mask locks sampling to encoded init; VAE RGB reconstruction is still lossy.
Measure SSIM rather than assuming exact RGB preservation.

CPU tests (torch only): python -m unittest discover -s . -p test_latent.py -v

Sources:
https://github.com/Comfy-Org/ComfyUI/blob/v0.34.6/comfy_extras/nodes_minimax_h3.py
https://github.com/Comfy-Org/ComfyUI/blob/v0.34.6/comfy/samplers.py
