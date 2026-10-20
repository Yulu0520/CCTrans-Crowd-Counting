"""Portable adapter-only checkpoints with strict baseline compatibility checks."""

import hashlib
from pathlib import Path

import torch

from Networks.dora import ADAPTERS, adapter_state, attach_dora, load_adapter_state, set_route


FORMAT = 'cctrans-dora-v1'


def file_identity(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(Path(path).resolve()), 'sha256': digest.hexdigest()}


def read_checkpoint(path):
    return torch.load(path, map_location='cpu')


def is_adapter(checkpoint):
    return isinstance(checkpoint, dict) and checkpoint.get('format') == FORMAT


def load_baseline(model, path):
    checkpoint = read_checkpoint(path)
    if is_adapter(checkpoint):
        raise ValueError('A trained baseline checkpoint is required, not an adapter')
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint


def save_adapter(path, model, name, baseline, train_config, **training_state):
    torch.save(dict(
        format=FORMAT, adapter_name=name, baseline=baseline,
        adapter_settings=model.dora_settings[name], target_modules=model.dora_targets,
        adapter_state_dict=adapter_state(model, name), train_config=train_config,
        **training_state,
    ), path)


def restore_adapter(model, checkpoint, name, baseline):
    if not is_adapter(checkpoint) or checkpoint['adapter_name'] != name:
        raise ValueError('Checkpoint is not a ' + name + ' adapter')
    if checkpoint['baseline']['sha256'] != baseline['sha256']:
        raise ValueError('Adapter was trained against a different baseline checkpoint')
    if checkpoint['adapter_settings'] != model.dora_settings[name]:
        raise ValueError('Adapter rank/alpha mismatch for ' + name)
    if checkpoint['target_modules'] != model.dora_targets:
        raise ValueError('Adapter attention targets do not match')
    load_adapter_state(model, name, checkpoint['adapter_state_dict'])


def load_for_inference(model, model_path, baseline_path='', day_path='', night_path='', route='auto'):
    primary = read_checkpoint(model_path)
    adapters = {}
    if is_adapter(primary):
        adapters[primary['adapter_name']] = primary
        baseline_path = baseline_path or primary['baseline']['path']
        default_route = primary['adapter_name']
    else:
        baseline_path = baseline_path or model_path
        default_route = 'baseline'
    for name, path in zip(ADAPTERS, [day_path, night_path]):
        if path:
            if name in adapters:
                raise ValueError('Adapter supplied twice: ' + name)
            adapters[name] = read_checkpoint(path)
    if not is_adapter(primary) and str(baseline_path) == str(model_path):
        model.load_state_dict(primary.get('model_state_dict', primary), strict=True)
    else:
        load_baseline(model, baseline_path)
    del primary
    if adapters:
        identity = file_identity(baseline_path)
        settings = {name: {'rank': 4, 'alpha': 4.0} for name in ADAPTERS}
        for name, checkpoint in adapters.items():
            if not is_adapter(checkpoint) or checkpoint['adapter_name'] != name:
                raise ValueError('Expected a ' + name + ' checkpoint')
            settings[name] = checkpoint['adapter_settings']
        attach_dora(model, settings)
        for name, checkpoint in adapters.items():
            restore_adapter(model, checkpoint, name, identity)
    route = default_route if route == 'auto' else route
    if route != 'baseline' and route not in adapters:
        raise ValueError('No trained adapter supplied for route: ' + route)
    set_route(model, route)
    model.loaded_adapters = tuple(adapters)
    model.inference_route = route
    model.eval()
    return model


def add_inference_arguments(parser):
    parser.add_argument('--adapter-route', choices=['auto', 'baseline'] + list(ADAPTERS),
                        default='auto', help='auto uses the adapter named by the model checkpoint')
    parser.add_argument('--baseline-checkpoint', default='', help='baseline path override after moving files')
    parser.add_argument('--dora-day-checkpoint', default='')
    parser.add_argument('--dora-night-checkpoint', default='')
