import argparse
import copy
import logging
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from torch import nn

from Networks.ALTGVT import GroupBlock
from Networks.dora import (
    ADAPTERS, adapter_state, attach_dora, configure_training, routers, set_route,
)
from utils.dora_checkpoint import (
    file_identity, load_baseline, load_for_inference, read_checkpoint,
    restore_adapter, save_adapter,
)
from utils.dora_options import resolve_training_options
from losses.ot_loss import OT_Loss
from train_helper_ALTGVT import Trainer
from train import add_training_arguments, parse_args
from utils.training_state import restore_random_state


SETTINGS = {'dora-day': {'rank': 2, 'alpha': 2.0},
            'dora-night': {'rank': 3, 'alpha': 4.0}}


class SmallTwins(nn.Module):
    """Actual Twins local/global attention with a small head for CPU checks."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleList([
            GroupBlock(8, 2, qkv_bias=True, ws=2),
            GroupBlock(8, 2, qkv_bias=True, ws=1, sr_ratio=2),
        ])])
        self.norm = nn.BatchNorm1d(8)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        for block in self.blocks[0]:
            x = block(x, 4, 4)
        return self.head(self.norm(x.reshape(-1, 8)))


class DoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SmallTwins().eval()
        self.inputs = torch.randn(2, 16, 8)

    def assert_state_equal(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_identity_routing_and_key_unchanged(self):
        before = self.model(self.inputs).detach()
        local_key = self.model.blocks[0][0].attn.qkv.weight.detach()[8:16].clone()
        global_key = self.model.blocks[0][1].attn.kv.weight.detach()[:8].clone()
        attach_dora(self.model, SETTINGS)
        self.assertEqual(len(routers(self.model)), 3)
        for route in ['baseline'] + list(ADAPTERS):
            set_route(self.model, route)
            torch.testing.assert_close(self.model(self.inputs), before)
        # A nonzero direction update and magnitude change must still leave K untouched.
        with torch.no_grad():
            for module in routers(self.model):
                for branch in module.branches.values():
                    for part in branch:
                        part.B.fill_(0.1)
                        part.magnitude.mul_(1.2)
        for route in ADAPTERS:
            set_route(self.model, route)
            self.assertTrue(torch.equal(self.model.blocks[0][0].attn.qkv.weight[8:16], local_key))
            self.assertTrue(torch.equal(self.model.blocks[0][1].attn.kv.weight[:8], global_key))

    def test_only_active_adapter_updates_and_buffers_stay_frozen(self):
        attach_dora(self.model, SETTINGS)
        for name in ADAPTERS:
            configure_training(self.model, name)
            before = copy.deepcopy(self.model.state_dict())
            active_before = adapter_state(self.model, name)
            other = ADAPTERS[1] if name == ADAPTERS[0] else ADAPTERS[0]
            calls = []
            handles = [module.branches[other].register_forward_hook(
                lambda *unused: calls.append(True)) for module in routers(self.model)]
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad], lr=0.01)
            self.model(self.inputs).square().mean().backward()
            optimizer.step()
            self.assertFalse(calls)
            for handle in handles:
                handle.remove()
            self.assertFalse(self.model.norm.training)
            for key, value in self.model.state_dict().items():
                if '.branches.' + name + '.' not in key:
                    self.assertTrue(torch.equal(before[key], value), key)
            self.assertTrue(any(not torch.equal(value, adapter_state(self.model, name)[key])
                                for key, value in active_before.items()))
            for key, value in self.model.named_parameters():
                if value.requires_grad:
                    self.assertIn('.branches.' + name + '.', key)
                    self.assertTrue(torch.isfinite(value.grad).all())
                else:
                    self.assertIsNone(value.grad)

    def test_adapter_roundtrip_and_compatibility_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / 'baseline.pth'
            torch.save(self.model.state_dict(), baseline)
            identity = file_identity(baseline)
            attach_dora(self.model, SETTINGS)
            paths = {}
            for name in ADAPTERS:
                configure_training(self.model, name)
                optimizer = torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad], lr=0.01)
                self.model(self.inputs).square().mean().backward()
                optimizer.step()
                paths[name] = root / (name + '.pth')
                save_adapter(paths[name], self.model, name, identity, {},
                             epoch=0, optimizer_state_dict=optimizer.state_dict())
                checkpoint = read_checkpoint(paths[name])
                self.assertTrue(all('.branches.' + name + '.' in key
                                    for key in checkpoint['adapter_state_dict']))
                self.assertNotIn('model_state_dict', checkpoint)
                loaded = load_for_inference(SmallTwins(), paths[name])
                torch.testing.assert_close(loaded(self.inputs), self.model(self.inputs))
                configure_training(loaded, name)
                fresh_optimizer = torch.optim.AdamW(
                    [p for p in loaded.parameters() if p.requires_grad], lr=0.01)
                fresh_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.assertTrue(fresh_optimizer.state)
            dual = load_for_inference(SmallTwins(), baseline,
                                      day_path=paths['dora-day'], night_path=paths['dora-night'])
            for name in ADAPTERS:
                set_route(dual, name)
                set_route(self.model, name)
                torch.testing.assert_close(dual(self.inputs), self.model(self.inputs))
            baseline_model = load_for_inference(SmallTwins(), baseline)
            set_route(dual, 'baseline')
            torch.testing.assert_close(dual(self.inputs), baseline_model(self.inputs))
            checkpoint = read_checkpoint(paths['dora-day'])
            with self.assertRaisesRegex(ValueError, 'different baseline'):
                restore_adapter(dual, checkpoint, 'dora-day', {'sha256': 'wrong'})
            with self.assertRaisesRegex(ValueError, 'not a dora-night'):
                restore_adapter(dual, checkpoint, 'dora-night', identity)
            changed = copy.deepcopy(checkpoint)
            changed['adapter_settings']['alpha'] += 1
            with self.assertRaisesRegex(ValueError, 'rank/alpha'):
                restore_adapter(dual, changed, 'dora-day', identity)
            with self.assertRaisesRegex(ValueError, 'No trained adapter'):
                load_for_inference(SmallTwins(), baseline, route='dora-night')
            with self.assertRaisesRegex(ValueError, 'not an adapter'):
                load_baseline(SmallTwins(), paths['dora-day'])

    def test_adapter_trainer_saves_best_weights_and_resume_checkpoint(self):
        class CountingTwins(SmallTwins):
            def forward(self, x):
                density = nn.functional.softplus(super().forward(x)).reshape(x.size(0), 1, 4, 4)
                return density, density / (density.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)

        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory, 'base.pth')
            model = CountingTwins()
            torch.save(model.state_dict(), baseline)
            attach_dora(model, SETTINGS)
            args = SimpleNamespace(stage='dora-night', max_epoch=0, val_epoch=1,
                                   val_start=0, density_loss='l2', wl2=1., wot=0.01, wtv=0.01)
            trainer = Trainer(args)
            trainer.model = model
            configure_training(model, args.stage)
            trainer.optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.01)
            trainer.device = torch.device('cpu')
            trainer.start_epoch = 0
            trainer.best_mae = trainer.best_mse = float('inf')
            trainer.logger = logging.getLogger('adapter-trainer-test')
            trainer.save_dir = directory
            trainer.baseline_identity = file_identity(baseline)
            trainer.ot_loss = OT_Loss(32, 8, 0, trainer.device, num_of_iter_in_ot=10)
            trainer.mse, trainer.mae = nn.MSELoss(), nn.L1Loss()
            trainer.tv_loss = nn.L1Loss(reduction='none')
            target = torch.zeros(2, 1, 4, 4)
            target[1, 0, 0, 0] = 1
            trainer.dataloaders = {'train': [(
                self.inputs, [torch.empty(0, 2), torch.tensor([[4., 4.]])], target)]}

            def validate():
                trainer.save_best_model(1., 2.)

            with patch('train_helper_ALTGVT.wandb.log'), patch.object(trainer, 'val_epoch', validate):
                trainer.train()
            best = read_checkpoint(trainer.best_model_path)
            self.assertEqual(best['best_mae'], 1.)
            self.assertEqual(best['epoch'], 0)
            self.assertNotIn('optimizer_state_dict', best)
            self.assertEqual(Path(trainer.best_model_path).name, 'dora-night_best_mae-1.00_epoch-1.pth')
            self.assertFalse(list(Path(directory).glob('*.tar')))
            checkpoint = read_checkpoint(trainer.latest_checkpoint_path)
            self.assertEqual(Path(trainer.latest_checkpoint_path).name, 'dora-night_last_epoch-1.ckpt')
            self.assertIn('optimizer_state_dict', checkpoint)
            self.assertIn('random_state', checkpoint)
            self.assertEqual(checkpoint['best_mae'], 1.)
            restored = load_for_inference(CountingTwins(), trainer.best_model_path)
            torch.testing.assert_close(restored(self.inputs)[0], model(self.inputs)[0])

    def test_stage_options_are_independent(self):
        parser = argparse.ArgumentParser()
        add_training_arguments(parser)
        args = parser.parse_args([
            '--stage', 'dora-night', '--baseline-checkpoint', 'baseline.pth',
            '--dora-night-train-dir', 'night/train', '--dora-night-val-dir', 'night/val',
            '--dora-day-lr', '0.01', '--dora-night-lr', '0.002', '--dora-night-epochs', '3',
            '--dora-day-val-start', '0', '--dora-night-val-start', '0',
        ])
        args.resume = ''
        resolve_training_options(args, parser)
        self.assertEqual(args.lr, 0.002)
        self.assertEqual(args.max_epoch, 2)
        self.assertEqual(args.dora_day_lr, 0.01)

    def test_only_one_best_file_per_stage_and_save_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory, 'baseline.pth')
            torch.save(self.model.state_dict(), baseline)
            for stage in ['baseline'] + list(ADAPTERS):
                trainer = Trainer(SimpleNamespace(stage=stage))
                trainer.model = copy.deepcopy(self.model)
                if stage != 'baseline':
                    attach_dora(trainer.model, SETTINGS)
                    trainer.baseline_identity = file_identity(baseline)
                trainer.save_dir = str(Path(directory, stage))
                Path(trainer.save_dir).mkdir()
                trainer.logger = logging.getLogger('best-only-test')
                trainer.best_mae = trainer.best_mse = float('inf')
                trainer.epoch = 0
                first = Path(trainer.save_best_model(5., 6.))
                self.assertIn('mae-5.00_epoch-1.pth', first.name)
                self.assertIsNone(trainer.save_best_model(5., 6.))
                self.assertIsNone(trainer.save_best_model(7., 8.))
                trainer.epoch = 4
                second = Path(trainer.save_best_model(4., 5.))
                self.assertFalse(first.exists())
                self.assertTrue(second.exists())
                self.assertIn('mae-4.00_epoch-5.pth', second.name)
                self.assertEqual(list(Path(trainer.save_dir).glob('*.pth')), [second])
                trainer.epoch = 5
                with patch('train_helper_ALTGVT.torch.save', side_effect=OSError('disk full')):
                    with self.assertRaises(OSError):
                        trainer.save_best_model(3., 4.)
                self.assertEqual(trainer.best_mae, 4.)
                self.assertEqual(trainer.best_model_path, str(second))
                self.assertTrue(second.exists())
                self.assertFalse(list(Path(trainer.save_dir).glob('*.tmp')))

    def test_setup_training_and_resume_integration(self):
        class CountingTwins(SmallTwins):
            def forward(self, x):
                density = nn.functional.softplus(super().forward(x)).reshape(x.size(0), 1, 4, 4)
                return density, density / (density.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)

        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, *unused):
                pass

            def __len__(self):
                return 2

            def __getitem__(self, index):
                target = torch.zeros(1, 4, 4)
                target[0, 0, 0] = 1
                return torch.ones(16, 8), torch.tensor([[4., 4.]]), target

        class CPUSetupTorch:
            # Exercise setup's GPU-only path using CPU tensors, without altering torch globally.
            cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)

            @staticmethod
            def device(unused):
                return torch.device('cpu')

            def __getattr__(self, name):
                return getattr(torch, name)

        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory, 'baseline.pth')
            torch.save(CountingTwins().state_dict(), baseline)
            options = [
                'train.py', '--stage', 'dora-day', '--save-dir', directory,
                '--dora-day-train-dir', 'day/train', '--dora-day-val-dir', 'day/val',
                '--dora-day-epochs', '1', '--batch-size', '2', '--num-workers', '0',
                '--dora-day-val-start', '0', '--dora-night-val-start', '0',
                '--density-loss', 'l2',
            ]
            with patch.object(sys, 'argv', options + ['--baseline-checkpoint', str(baseline)]):
                args = parse_args()
            args.crop_size = 32
            with patch('train_helper_ALTGVT.torch', CPUSetupTorch()), \
                    patch('train_helper_ALTGVT.Crowd_sh', FakeDataset), \
                    patch('train_helper_ALTGVT.ALTGVT.alt_gvt_large', side_effect=lambda **kw: CountingTwins()), \
                    patch('train_helper_ALTGVT.wandb.init'), patch('train_helper_ALTGVT.wandb.log'):
                trainer = Trainer(args)
                trainer.setup()
                frozen = {key: value.clone() for key, value in trainer.model.state_dict().items()
                          if '.branches.dora-day.' not in key}

                def validate():
                    trainer.save_best_model(0.5, 0.5)

                with patch.object(trainer, 'val_epoch', validate):
                    trainer.train()
                for key, value in frozen.items():
                    self.assertTrue(torch.equal(trainer.model.state_dict()[key], value), key)
                self.assertFalse(list(Path(trainer.save_dir).glob('*.tar')))
                last = Path(trainer.latest_checkpoint_path)
                expected_random = (random.random(), np.random.random(), torch.rand(3))
                with patch.object(sys, 'argv', options + [
                        '--resume', str(last), '--dora-day-epochs', '2', '--dora-day-lr', '1e-5']):
                    resumed_args = parse_args()
                resumed_args.crop_size = 32
                resumed = Trainer(resumed_args)
                resumed.setup()
                self.assertEqual(random.random(), expected_random[0])
                self.assertEqual(np.random.random(), expected_random[1])
                torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
                self.assertEqual(resumed.start_epoch, 1)
                self.assertEqual(resumed.best_mae, 0.5)
                self.assertEqual(resumed.optimizer.param_groups[0]['lr'], 1e-5)
                self.assertTrue(resumed.optimizer.state)
                self.assert_state_equal(adapter_state(trainer.model, 'dora-day'),
                                        adapter_state(resumed.model, 'dora-day'))
                with patch.object(resumed, 'val_epoch'):
                    resumed.train()
                self.assertEqual(resumed.epoch, 1)
                self.assertFalse(list(Path(resumed.save_dir).glob('*.tar')))
                self.assertEqual(Path(resumed.latest_checkpoint_path).name, 'dora-day_last_epoch-2.ckpt')
                self.assertTrue(last.exists())  # Source runs are never deleted when resuming.
                baseline_options = [
                    'train.py', '--stage', 'baseline', '--save-dir', directory,
                    '--max-epoch', '0', '--val-start', '5', '--num-workers', '0',
                    '--batch-size', '2', '--density-loss', 'l2',
                ]
                with patch.object(sys, 'argv', baseline_options):
                    baseline_args = parse_args()
                baseline_args.crop_size = 32
                baseline_trainer = Trainer(baseline_args)
                baseline_trainer.setup()
                baseline_trainer.train()  # A CKPT must exist even before the first validation.
                baseline_last = Path(baseline_trainer.latest_checkpoint_path)
                self.assertTrue(baseline_last.exists())
                self.assertIsNone(baseline_trainer.best_model_path)
                with patch.object(sys, 'argv', baseline_options + [
                        '--resume', str(baseline_last), '--max-epoch', '1']):
                    baseline_resume_args = parse_args()
                baseline_resume_args.crop_size = 32
                baseline_resumed = Trainer(baseline_resume_args)
                baseline_resumed.setup()
                self.assertEqual(baseline_resumed.start_epoch, 1)
                self.assertTrue(baseline_resumed.optimizer.state)
                self.assert_state_equal(baseline_trainer.model.state_dict(),
                                        baseline_resumed.model.state_dict())
                baseline_resumed.train()
                self.assertEqual(Path(baseline_resumed.latest_checkpoint_path).name,
                                 'baseline_last_epoch-2.ckpt')

    def test_latest_checkpoint_retention_failure_and_random_state_all_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory, 'baseline.pth')
            torch.save(self.model.state_dict(), base)
            for stage in ['baseline'] + list(ADAPTERS):
                trainer = Trainer(SimpleNamespace(stage=stage))
                trainer.model = copy.deepcopy(self.model)
                if stage != 'baseline':
                    attach_dora(trainer.model, SETTINGS)
                    trainer.baseline_identity = file_identity(base)
                configure_training(trainer.model, stage)
                trainer.optimizer = torch.optim.AdamW(
                    [p for p in trainer.model.parameters() if p.requires_grad], lr=0.001)
                trainer.save_dir = str(Path(directory, stage))
                Path(trainer.save_dir).mkdir()
                trainer.logger = logging.getLogger('latest-checkpoint-test')
                trainer.best_mae = trainer.best_mse = float('inf')
                trainer.epoch = 0
                trainer.save_latest_checkpoint()
                first = Path(trainer.latest_checkpoint_path)
                checkpoint = read_checkpoint(first)
                self.assertEqual(checkpoint['stage'], stage)
                self.assertEqual(checkpoint['epoch'], 0)
                self.assertFalse(list(Path(trainer.save_dir).glob('*.pth')))
                if stage == 'baseline':
                    self.assertIn('model_state_dict', checkpoint)
                else:
                    self.assertNotIn('model_state_dict', checkpoint)
                    self.assertEqual(checkpoint['adapter_name'], stage)
                expected = (random.random(), np.random.random(), torch.rand(2))
                restore_random_state(checkpoint['random_state'])
                self.assertEqual(random.random(), expected[0])
                self.assertEqual(np.random.random(), expected[1])
                torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)
                trainer.epoch = 1
                trainer.save_best_model(3., 4.)
                trainer.save_latest_checkpoint()
                latest = Path(trainer.latest_checkpoint_path)
                self.assertFalse(first.exists())
                self.assertEqual(latest.name, stage + '_last_epoch-2.ckpt')
                self.assertEqual(len(list(Path(trainer.save_dir).glob('*.pth'))), 1)
                self.assertEqual(len(list(Path(trainer.save_dir).glob('*.ckpt'))), 1)
                self.assertEqual(read_checkpoint(latest)['best_mae'], 3.)
                trainer.epoch = 2
                with patch('train_helper_ALTGVT.torch.save', side_effect=OSError('disk full')):
                    with self.assertRaises(OSError):
                        trainer.save_latest_checkpoint()
                self.assertEqual(trainer.latest_checkpoint_path, str(latest))
                self.assertTrue(latest.exists())
                self.assertFalse(list(Path(trainer.save_dir).glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
