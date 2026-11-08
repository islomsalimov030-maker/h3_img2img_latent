"""CPU tests with stub schedule and real torch, no ComfyUI/GPU/models."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


class Nested:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True


comfy = types.ModuleType('comfy')
comfy.samplers = types.ModuleType('comfy.samplers')
comfy.samplers.SCHEDULER_NAMES = ['simple']
comfy.nested_tensor = types.ModuleType('comfy.nested_tensor')
comfy.nested_tensor.NestedTensor = Nested
with patch.dict(sys.modules, {'comfy': comfy, 'comfy.samplers': comfy.samplers,
                             'comfy.nested_tensor': comfy.nested_tensor}):
    spec = importlib.util.spec_from_file_location('h3_tested', Path(__file__).with_name('__init__.py'))
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)


class VAE:
    def encode(self, image):
        return torch.full((1, 24, 2, 4, 6), float(image.mean()))


class LatentTests(unittest.TestCase):
    def setUp(self):
        self.n = node.H3LatentFromImage()
        self.latent = {'samples': Nested([torch.zeros(1, 24, 2, 4, 6), torch.zeros(1, 32, 2, 9)])}
        self.image = torch.full((1, 64, 96, 3), .6)
        self.schedule = patch.object(node.H3LatentFromImage, '_schedule', return_value=torch.tensor([1., .5, 0.]))
        self.schedule.start()
        self.addCleanup(self.schedule.stop)

    def run_node(self, denoise, **kwargs):
        return self.n.run(self.latent, VAE(), self.image, None, 2, 'simple', denoise, **kwargs)

    def test_default_full_noise_legacy_exact(self):
        out, sig, info = self.run_node(1.)
        self.assertIs(out['samples'], self.latent['samples'])
        self.assertEqual(info, 'denoise 1.0 -> full generation from noise (2 steps)')
        self.assertTrue(torch.equal(sig, torch.tensor([1., .5, 0.])))

    def test_default_low_noise_legacy_bytes_and_info(self):
        out, _, info = self.run_node(.35)
        img = F.interpolate(self.image.movedim(-1, 1), size=(64, 96), mode='bicubic', align_corners=False, antialias=True)
        expected = VAE().encode(img.clamp(0, 1).movedim(1, -1).repeat(5, 1, 1, 1))
        self.assertEqual(out['samples'].tensors[0].numpy().tobytes(), expected.numpy().tobytes())
        self.assertIs(out['samples'].tensors[1], self.latent['samples'].tensors[1])
        self.assertEqual(info, 'denoise 0.35: 2 steps, sigmas 1.000, 0.500, 0.000 | latent (1, 24, 2, 4, 6)')

    def test_encode_at_full_noise_matches_vae(self):
        out, _, info = self.run_node(1., always_encode=True)
        expected = VAE().encode(self.image.repeat(5, 1, 1, 1))
        self.assertTrue(torch.allclose(out['samples'].tensors[0], expected))
        self.assertTrue(torch.count_nonzero(out['samples'].tensors[0]))
        self.assertIn('target_frames=[0, 1]', info)
        self.assertIn('reference_frames=[]', info)
        self.assertIn('cell_px=224', info)

    def test_mask_video_shape_area_resize_other_streams_zero(self):
        mask = torch.zeros(1, 8, 12); mask[:, :4] = 1
        out, _, info = self.run_node(1., always_encode=True, mask=mask)
        masks = out['noise_mask'].tensors
        self.assertEqual([m.shape for m in masks], [x.shape for x in out['samples'].tensors])
        resized = F.interpolate(mask[:, None], size=(4, 6), mode='area')
        self.assertTrue(torch.equal(masks[0][:, :, 0], resized.expand(1, 24, 4, 6)))
        self.assertTrue(torch.equal(masks[0][:, :, 0], masks[0][:, :, 1]))
        self.assertEqual(torch.count_nonzero(masks[1]), 0)
        self.assertIn('noise_mask=', info)

    def test_zero_mask_does_not_touch_conditioning_references(self):
        refs = [torch.randn(1, 24, 1, 4, 6), torch.randn(1, 24, 2, 4, 6)]
        snapshots = [x.clone() for x in refs]
        out, _, _ = self.run_node(1., always_encode=True, mask=torch.zeros(64, 96))
        self.assertEqual(sum(int(torch.count_nonzero(m)) for m in out['noise_mask'].tensors), 0)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(refs, snapshots)))
        # References are not temporal slots in samples; core keeps them in cond.
        self.assertNotIn('references', out)


if __name__ == '__main__':
    unittest.main()
