import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from datasets.crowd import Base, Crowd_sh, CustomDataset
from datasets.density import gen_gaussian_map
from losses.ot_loss import OT_Loss
from train_helper_ALTGVT import Trainer, train_collate


class DensityTests(unittest.TestCase):
    def test_gaussian_counts_and_boundaries(self):
        for points in [[], [[0, 0]], [[31.9, 31.9]],
                       [[0, 0], [31, 31]], [[8, 8], [8, 8], [20, 20]]]:
            density = gen_gaussian_map(32, 32, points)
            self.assertEqual(density.shape, (4, 4))
            self.assertEqual(density.dtype, np.float32)
            self.assertTrue(np.isfinite(density).all())
            self.assertTrue((density >= 0).all())
            self.assertAlmostEqual(float(density.sum()), len(points), places=5)

    def test_gaussian_is_smooth_and_adaptive(self):
        smooth = gen_gaussian_map(64, 64, [[32, 32]], 1)
        self.assertGreater(np.count_nonzero(smooth), 1)
        narrow = gen_gaussian_map(64, 64, [[16, 16], [48, 48]], 1, beta=0.1)
        wide = gen_gaussian_map(64, 64, [[16, 16], [48, 48]], 1, beta=0.5)
        self.assertGreater(narrow.max(), wide.max())
        coarse = gen_gaussian_map(64, 64, [[32, 32]], 8)
        np.testing.assert_allclose(coarse, smooth.reshape(8, 8, 8, 8).sum(axis=(1, 3)))

    def test_all_transform_paths_and_flip(self):
        image = Image.new('RGB', (32, 32))
        for cls in [Base, Crowd_sh, CustomDataset]:
            dataset = cls.__new__(cls)
            Base.__init__(dataset, '', 32)
            for mode in ['tv', 'l2']:
                dataset.density_loss = mode
                for points in [np.empty((0, 2)), np.array([[1., 2.], [20., 21.]])]:
                    with patch('datasets.crowd.random.random', return_value=0.0):
                        original = dataset.train_transform(image, points.copy())
                    with patch('datasets.crowd.random.random', return_value=1.0):
                        flipped = dataset.train_transform(image, points.copy())
                    self.assertEqual(original[2].shape, (1, 4, 4))
                    self.assertAlmostEqual(original[2].sum().item(), len(points), places=5)
                    torch.testing.assert_close(flipped[2], original[2].flip(-1))
                    torch.testing.assert_close(flipped[1][:, 0], 31 - original[1][:, 0])
                if mode == 'tv':
                    self.assertEqual(torch.count_nonzero(original[2]).item(), 2)
                else:
                    self.assertGreater(torch.count_nonzero(original[2]).item(), 2)

    def test_loss_values_and_gradients(self):
        trainer = Trainer(SimpleNamespace(density_loss='l2', wl2=2., wtv=0.01))
        trainer.mse = nn.MSELoss()
        trainer.tv_loss = nn.L1Loss(reduction='none')
        outputs = torch.ones(2, 1, 4, 4, requires_grad=True)
        normed = outputs / outputs.sum(dim=(1, 2, 3), keepdim=True)
        target = torch.zeros_like(outputs)
        counts = torch.tensor([0., 2.])
        target[1, 0, 0, 0] = 2
        actual = trainer.compute_density_loss(outputs, normed, target, counts)
        torch.testing.assert_close(actual, ((outputs - target) ** 2).mean() * 2)
        actual.backward()
        self.assertTrue(torch.isfinite(outputs.grad).all())
        trainer.args.density_loss = 'tv'
        expected = ((normed - target / (counts[:, None, None, None] + 1e-6)).abs()
                    .sum(dim=(1, 2, 3)) * counts).mean() * 0.01
        torch.testing.assert_close(
            trainer.compute_density_loss(outputs, normed, target, counts), expected)
        with self.assertRaises(ValueError):
            trainer.compute_density_loss(outputs, normed, target[:, :, :2], counts)

    def test_training_batch_both_modes(self):
        class TinyDensityModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1)

            def forward(self, images):
                density = nn.functional.softplus(
                    nn.functional.avg_pool2d(self.conv(images), 8))
                return density, density / (density.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)

        for mode in ['l2', 'tv']:
            dataset = Base('', 32)
            dataset.density_loss = mode
            batch = train_collate([
                dataset.train_transform(Image.new('RGB', (32, 32)), points)
                for points in [np.empty((0, 2)), np.array([[10., 12.], [24., 25.]])]
            ])
            args = SimpleNamespace(density_loss=mode, wl2=1., wtv=0.01,
                                   wot=0.01, max_epoch=1)
            trainer = Trainer(args)
            trainer.device = torch.device('cpu')
            trainer.model = TinyDensityModel()
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.mse, trainer.mae = nn.MSELoss(), nn.L1Loss()
            trainer.tv_loss = nn.L1Loss(reduction='none')
            trainer.ot_loss = OT_Loss(32, 8, 0, trainer.device, num_of_iter_in_ot=10)
            trainer.epoch = 0
            trainer.dataloaders = {'train': [batch]}
            trainer.logger = logging.getLogger('density-test')
            before = trainer.model.conv.weight.detach().clone()
            with tempfile.TemporaryDirectory() as directory:
                trainer.save_dir = directory
                with patch('train_helper_ALTGVT.wandb.log') as log:
                    trainer.train_epoch()
                logged = log.call_args[0][0]
                self.assertIn('train/' + mode + '_loss', logged)
                self.assertTrue(torch.isfinite(logged['train/TOTAL_loss']).all())
                self.assertFalse(list(Path(directory).glob('*.tar')))
                self.assertFalse(list(Path(directory).glob('*.pth')))
            self.assertFalse(torch.equal(before, trainer.model.conv.weight))
            for parameter in trainer.model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
