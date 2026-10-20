import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

import test_image_patch as testing
from Networks.dora import attach_dora, routers
from utils.dora_checkpoint import file_identity, save_adapter
from utils.inference_router import FilenameRouter, make_router


class ImageModel(nn.Module):
    """Small real DoRA projections producing stride-eight density patches."""
    def __init__(self):
        super().__init__()
        attention = nn.Module()
        attention.qkv = nn.Linear(3, 9, bias=False)
        nn.init.ones_(attention.qkv.weight)
        block = nn.Module()
        block.attn = attention
        self.blocks = nn.ModuleList([block])
        self.calls = []

    def forward(self, image):
        self.calls.append(routers(self)[0].route)
        pooled = nn.functional.avg_pool2d(image, 8).permute(0, 2, 3, 1)
        qkv = self.blocks[0].attn.qkv(pooled)
        density = qkv[..., :3].abs().sum(-1).unsqueeze(1)
        return density, density / (density.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)


class RouterTests(unittest.TestCase):
    def model(self):
        model = ImageModel()
        attach_dora(model, {name: {'rank': 2, 'alpha': 2.}
                            for name in ('dora-day', 'dora-night')})
        with torch.no_grad():
            for router in routers(model):
                for part in router.branches['dora-night']:
                    part.magnitude.mul_(2)
        model.loaded_adapters = ('dora-day', 'dora-night')
        return model.eval()

    def test_filename_rules_unknown_and_ambiguous(self):
        router = FilenameRouter()
        for filename in ['IMG_1.jpg', '/tmp/IMG_2', 'DAY_3.JPG']:
            self.assertEqual(router.predict(None, filename).route, 'dora-day')
        for filename in ['night_IMG_1.jpg', 'NIGHT_2.jpg']:
            self.assertEqual(router.predict(None, filename).route, 'dora-night')
        with self.assertRaisesRegex(ValueError, 'Cannot identify'):
            router.predict(None, 'other.jpg')
        fallback = FilenameRouter(unknown_route='baseline').predict(None, 'other.jpg')
        self.assertEqual((fallback.route, fallback.domain), ('baseline', 'unknown'))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            FilenameRouter(['IMG_'], ['IMG_'], 'baseline').predict(None, 'IMG_1')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            FilenameRouter([''], ['night_'])

    def test_fixed_and_future_router_modes(self):
        fixed = make_router('fixed', 'baseline')
        self.assertEqual(fixed.predict(None, 'night_IMG_1').route, 'baseline')
        self.assertEqual(fixed.predict(None, 'night_IMG_1').domain, 'night')
        with self.assertRaisesRegex(NotImplementedError, 'reserved'):
            make_router('classifier')

    def test_image_level_routing_patches_and_metrics(self):
        model = self.model()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        images = [(torch.ones(1, 3, 32, 48), torch.tensor([10]), [name])
                  for name in ['IMG_1', 'night_IMG_1', 'day_2']]
        router = FilenameRouter()
        with patch.object(router, 'predict', wraps=router.predict) as predict:
            records, metrics = testing.evaluate(model, images, torch.device('cpu'), 32, 1, router)
        self.assertEqual(predict.call_count, 3)
        self.assertEqual(model.calls, ['dora-day'] * 2 + ['dora-night'] * 2 + ['dora-day'] * 2)
        self.assertAlmostEqual(records[1]['predicted_count'], 2 * records[0]['predicted_count'])
        grouped = {metric['group']: metric for metric in metrics}
        self.assertEqual(grouped['all']['count'], 3)
        self.assertEqual(grouped['day']['count'], 2)
        self.assertEqual(grouped['night']['count'], 1)
        self.assertIsNone(grouped['unknown']['mae'])
        self.assertAlmostEqual(grouped['all']['mae'], np.mean([abs(row['error']) for row in records]))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        with tempfile.TemporaryDirectory() as root:
            output = Path(testing.save_results(root, 'filename', records, metrics))
            with (output / 'routing.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]['route'], 'dora-night')
            self.assertEqual(rows[1]['confidence'], '')
            self.assertTrue((output / 'metrics.csv').exists())
            self.assertTrue((output / 'ALGVT_sha_test.txt').exists())

    def test_missing_adapter_and_empty_dataset_fail(self):
        model = self.model()
        model.loaded_adapters = ('dora-day',)
        image = [(torch.ones(1, 3, 32, 32), torch.tensor([1]), ['night_IMG_1'])]
        with self.assertRaisesRegex(ValueError, 'No trained adapter'):
            testing.evaluate(model, image, torch.device('cpu'), 32, 1, FilenameRouter())
        self.assertFalse(model.calls)
        with self.assertRaisesRegex(ValueError, 'No test images'):
            testing.evaluate(model, [], torch.device('cpu'), 32, 1, FilenameRouter())

    def test_full_test_entry_loads_three_files_and_mixed_validation_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'mix' / 'val'
            split.mkdir(parents=True)
            for name in ['IMG_1', 'night_IMG_1']:
                Image.new('RGB', (48, 32), 'white').save(split / (name + '.jpg'))
                np.save(split / (name + '.npy'), np.array([[8., 8.]]))
            base = root / 'baseline.pth'
            torch.save(ImageModel().state_dict(), base)
            model = self.model()
            identity = file_identity(base)
            paths = []
            for name in ('dora-day', 'dora-night'):
                path = root / (name + '.pth')
                save_adapter(path, model, name, identity, {})
                paths.append(str(path))
            args = testing.parser.parse_args([
                '--device', 'cpu', '--num-workers', '0', '--model-path', str(base),
                '--dora-day-checkpoint', paths[0], '--dora-night-checkpoint', paths[1],
                '--router-mode', 'filename', '--data-path', str(root / 'mix'), '--split', 'val',
                '--crop-size', '32', '--batch-size', '1', '--output-dir', str(root / 'results'),
            ])
            with patch('test_image_patch.ALTGVT.alt_gvt_large', side_effect=lambda **kw: ImageModel()):
                result = testing.test(args)
            self.assertEqual([row['route'] for row in result['records']], ['dora-day', 'dora-night'])
            self.assertTrue(Path(result['output_dir'], 'routing.csv').exists())
            self.assertAlmostEqual(result['records'][1]['predicted_count'],
                                   2 * result['records'][0]['predicted_count'])


if __name__ == '__main__':
    unittest.main()
