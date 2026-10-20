"""Independent training options for each day/night adapter."""

import math

from Networks.dora import ADAPTERS


def settings_from_args(args):
    return {name: {'rank': getattr(args, name.replace('-', '_') + '_rank'),
                   'alpha': getattr(args, name.replace('-', '_') + '_alpha')}
            for name in ADAPTERS}


def resolve_training_options(args, parser):
    for name in ADAPTERS:
        prefix = name.replace('-', '_')
        values = {key: getattr(args, prefix + '_' + key) for key in
                  ['rank', 'alpha', 'lr', 'weight_decay', 'epochs', 'val_epoch', 'val_start']}
        if (values['rank'] <= 0 or values['epochs'] <= 0 or values['val_epoch'] <= 0
                or not 0 <= values['val_start'] < values['epochs']):
            parser.error(name + ': rank/epochs/validation interval must be positive; '
                         'validation start must be within the stage')
        if any(not math.isfinite(values[key]) for key in ['alpha', 'lr', 'weight_decay']):
            parser.error(name + ': hyperparameters must be finite')
        if values['alpha'] <= 0 or values['lr'] <= 0 or values['weight_decay'] < 0:
            parser.error(name + ': alpha/lr must be positive and weight decay nonnegative')
    if args.stage == 'baseline':
        if any(getattr(args, name.replace('-', '_') + '_checkpoint') for name in ADAPTERS):
            parser.error('Adapter checkpoints require an adapter training stage')
        if args.baseline_checkpoint:
            parser.error('Use --resume to initialize or resume baseline training')
        return
    prefix = args.stage.replace('-', '_')
    if not args.baseline_checkpoint and not args.resume:
        parser.error('Adapter training needs --baseline-checkpoint or --resume')
    if args.resume and any(getattr(args, name.replace('-', '_') + '_checkpoint') for name in ADAPTERS):
        parser.error('Use --resume or adapter weight checkpoints, not both')
    if not all(getattr(args, prefix + '_' + split + '_dir') for split in ['train', 'val']):
        parser.error('Provide --' + args.stage + '-train-dir and --' + args.stage + '-val-dir')
    args.lr = getattr(args, prefix + '_lr')
    args.weight_decay = getattr(args, prefix + '_weight_decay')
    # Preserve the existing inclusive zero-based Trainer loop while exposing epoch counts.
    args.max_epoch = getattr(args, prefix + '_epochs') - 1
    args.val_epoch = getattr(args, prefix + '_val_epoch')
    args.val_start = getattr(args, prefix + '_val_start')
