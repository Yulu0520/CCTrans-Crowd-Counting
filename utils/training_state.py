"""Serializable random state for epoch-boundary training checkpoints."""

import random

import numpy as np
import torch


def capture_random_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {
            'algorithm': numpy_state[0], 'keys': numpy_state[1].tolist(),
            'position': numpy_state[2], 'has_gauss': numpy_state[3],
            'cached_gaussian': numpy_state[4],
        },
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
    }


def restore_random_state(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state['algorithm'],
                         np.asarray(numpy_state['keys'], dtype=np.uint32),
                         numpy_state['position'], numpy_state['has_gauss'],
                         numpy_state['cached_gaussian']))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Restoring CUDA random state requires the same number of visible GPUs')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    torch.backends.cudnn.benchmark = state['cudnn_benchmark']
    torch.backends.cudnn.deterministic = state['cudnn_deterministic']
