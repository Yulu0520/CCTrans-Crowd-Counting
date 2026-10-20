"""Independent day/night DoRA branches for Twins Q/V projections."""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize


ADAPTERS = ('dora-day', 'dora-night')
ROUTES = ('baseline',) + ADAPTERS


class DoRASlice(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('DoRA rank and alpha must be positive and finite')
        self.scaling = alpha / rank
        self.A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        self.magnitude = nn.Parameter(weight.detach().norm(dim=1))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        direction = weight + self.scaling * (self.B @ self.A)
        denominator = direction.norm(dim=1).clamp_min(torch.finfo(weight.dtype).eps).detach()
        return direction * (self.magnitude / denominator).unsqueeze(1)


class DualProjection(nn.Module):
    def __init__(self, weight, slices, settings):
        super().__init__()
        self.slices = slices
        self.route = 'baseline'
        self.branches = nn.ModuleDict({
            name: nn.ModuleList([
                DoRASlice(weight[start:end], **settings[name]) for start, end in slices
            ]) for name in ADAPTERS
        })

    def forward(self, weight):
        if self.route == 'baseline':
            return weight
        pieces, previous = [], 0
        for (start, end), adapter in zip(self.slices, self.branches[self.route]):
            pieces.extend([weight[previous:start], adapter(weight[start:end])])
            previous = end
        pieces.append(weight[previous:])
        return torch.cat(pieces, dim=0)


def projections(model):
    """Enumerate only attention Q/V weights, including packed QKV and split Q/KV."""
    found = []
    for name, module in model.named_modules():
        if not name.startswith('blocks.') or not name.endswith('.attn'):
            continue
        if hasattr(module, 'qkv'):
            layer = module.qkv
            dim = layer.in_features
            if layer.out_features != 3 * dim:
                raise ValueError('Unsupported QKV projection: ' + name)
            found.append((name + '.qkv', layer, [(0, dim), (2 * dim, 3 * dim)]))
        elif hasattr(module, 'q') and hasattr(module, 'kv'):
            dim = module.q.out_features
            if module.kv.out_features != 2 * dim:
                raise ValueError('Unsupported KV projection: ' + name)
            found.extend([(name + '.q', module.q, [(0, dim)]),
                          (name + '.kv', module.kv, [(dim, 2 * dim)])])
        else:
            raise ValueError('Unsupported attention projection: ' + name)
    if not found:
        raise ValueError('No Twins attention projections found')
    return found


def attach_dora(model, settings):
    layers = projections(model)
    for name, layer, slices in layers:
        if not isinstance(layer, nn.Linear) or parametrize.is_parametrized(layer, 'weight'):
            raise ValueError('Expected an unadapted Linear projection: ' + name)
    for _, layer, slices in layers:
        parametrize.register_parametrization(
            layer, 'weight', DualProjection(layer.weight, slices, settings))
    model.dora_settings = {name: dict(settings[name]) for name in ADAPTERS}
    model.dora_targets = [name for name, _, _ in layers]


def routers(model):
    return [module for module in model.modules() if isinstance(module, DualProjection)]


def set_route(model, route):
    if route not in ROUTES:
        raise ValueError('Unknown adapter route: ' + route)
    modules = routers(model)
    if not modules and route != 'baseline':
        raise ValueError('DoRA adapters are not attached')
    for module in modules:
        module.route = route


def configure_training(model, stage):
    set_route(model, stage)
    if stage == 'baseline':
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    # Also freeze BatchNorm buffers, dropout and stochastic depth in the baseline.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for module in routers(model):
        module.branches[stage].train()
        for parameter in module.branches[stage].parameters():
            parameter.requires_grad_(True)


def adapter_state(model, name):
    marker = '.branches.' + name + '.'
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if marker in key}


def load_adapter_state(model, name, state):
    expected = adapter_state(model, name)
    if state.keys() != expected.keys():
        raise ValueError('Adapter parameter names do not match the model')
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError('Adapter parameter shape mismatch: ' + key)
    model.load_state_dict(state, strict=False)
