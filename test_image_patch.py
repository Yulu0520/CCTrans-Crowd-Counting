import argparse
import csv
from datetime import datetime
from pathlib import Path
import torch
import os
import numpy as np
import datasets.crowd as crowd
from Networks import ALTGVT
from utils.dora_checkpoint import load_for_inference
from Networks.dora import set_route
from utils.inference_router import make_router
import torch.nn.functional as F

def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h//factor*factor), int(w//factor*factor)
    img_tensor = F.interpolate(img_tensor, (h, w), mode='bilinear', align_corners=True)

    return img_tensor
def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(img_tensor, (min_size, int(min_size / h * w)), mode='bilinear', align_corners=True)
        else:
            img_tensor = F.interpolate(img_tensor, (int(min_size / w * h), min_size), mode='bilinear', align_corners=True)
    return img_tensor

parser = argparse.ArgumentParser(description='Test ')

# 在這裡填入權重與資料路徑的 default，即可直接執行 python test_image_patch.py。
parser.add_argument('--model-path', type=str, default='ckpts/ALTGVT/CCTrans_12-1-input-256_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0_density-tv_wl2-1.0_beta-0.3_sigma-15.0/baseline/1020-105707-269876/best_model_mae-58.87_epoch-261.pth',
                    help='baseline full-model checkpoint, or the main adapter checkpoint')
parser.add_argument('--dora-day-checkpoint', type=str, default='',
                    help='day adapter-only checkpoint')
parser.add_argument('--dora-night-checkpoint', type=str, default='',
                    help='night adapter-only checkpoint')
parser.add_argument('--data-path', type=str, default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/day',
                    help='dataset root; use --split val for a mix/val directory')

# baseline：只用 baseline；dora-day／dora-night：baseline 加上指定的 DoRA。
# auto：model-path 是 adapter 時選該 adapter，否則只用 baseline。
parser.add_argument('--adapter-route', choices=['auto', 'baseline', 'dora-day', 'dora-night'],
                    default='auto', help='select which loaded branch to evaluate')
parser.add_argument('--baseline-checkpoint', type=str, default='',
                    help='optional baseline override when model-path points to an adapter')

# fixed：整次測試固定分支；filename：每張原圖依檔名選路。
# classifier：預留未來影像 router，尚未實作。
parser.add_argument('--router-mode', choices=['fixed', 'filename', 'classifier'],
                    default='fixed', help='how to choose an adapter for each full image')
parser.add_argument('--day-prefixes', nargs='+', default=['IMG_', 'day_'],
                    help='case-insensitive day filename prefixes')
parser.add_argument('--night-prefixes', nargs='+', default=['night_IMG_', 'night_'],
                    help='case-insensitive night filename prefixes')
parser.add_argument('--unknown-route', choices=['error', 'baseline'], default='error',
                    help='filename mode: fail or use baseline for an unrecognized name')
parser.add_argument('--split', default='val',
                    help='subdirectory under data-path, e.g. val or train; omitted keeps legacy dataset layout')
parser.add_argument('--output-dir', default='test_results',
                    help='parent directory for timestamped metrics and per-image routing CSVs')

parser.add_argument('--device', default='0', help='GPU index, or cpu')
parser.add_argument('--num-workers', type=int, default=1, help='DataLoader workers; use 0 for no subprocesses')
parser.add_argument('--batch-size', type=int, default=4,
                        help='train batch size')
parser.add_argument('--crop-size', type=int, default=256,
                    help='the crop size of the train image')
parser.add_argument('--dataset', type=str, default='sha',
                    help='dataset name: qnrf, nwpu, sha, shb, custom')
parser.add_argument('--pred-density-map-path', type=str, default='inference_results',
                    help='save predicted density maps when pred-density-map-path is not empty.')

def predict_density(model, inputs, device, crop_size, batch_size):
    """Keep the existing patch/overlap calculation, with one route for all patches."""
    if inputs.size(0) != 1:
        raise ValueError('Route and stitch one original image at a time')
    if min(inputs.shape[-2:]) < crop_size:
        raise ValueError('Input image is smaller than crop-size; resize before patch inference')
    crop_imgs, crop_masks = [], []
    b, c, h, w = inputs.size()
    rh, rw = crop_size, crop_size
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
            mask = torch.zeros([b, 1, h, w]).to(device)
            mask[:, :, gis:gie, gjs:gje].fill_(1.0)
            crop_masks.append(mask)
    crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

    crop_preds = []
    nz, bz = crop_imgs.size(0), batch_size
    for i in range(0, nz, bz):
        gs, gt = i, min(nz, i + bz)
        crop_pred, _ = model(crop_imgs[gs:gt])

        _, _, h1, w1 = crop_pred.size()

        crop_pred = F.interpolate(crop_pred, size=(h1 * 8, w1 * 8), mode='bilinear', align_corners=True) / 64

        crop_preds.append(crop_pred)
    crop_preds = torch.cat(crop_preds, dim=0)

    # splice them to the original size
    idx = 0
    pred_map = torch.zeros([b, 1, h, w]).to(device)
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
            idx += 1
    # for the overlapping area, compute average value
    mask = crop_masks.sum(dim=0).unsqueeze(0)
    outputs = pred_map / mask
    return outputs


def summarize(records):
    if not records:
        raise ValueError('No test images found')
    metrics = []
    for domain in ['all', 'day', 'night', 'unknown']:
        selected = records if domain == 'all' else [row for row in records if row['domain'] == domain]
        errors = np.array([row['error'] for row in selected], dtype=np.float64)
        metrics.append(dict(group=domain, count=len(selected),
                            mae=float(np.abs(errors).mean()) if selected else None,
                            rmse=float(np.sqrt(np.square(errors).mean())) if selected else None))
    return metrics


def evaluate(model, dataloader, device, crop_size, batch_size, router):
    if crop_size <= 0 or crop_size % 8 or batch_size <= 0:
        raise ValueError('crop-size must be a positive multiple of 8; batch-size must be positive')
    records = []
    with torch.no_grad():
        for inputs, count, names in dataloader:
            filename = names[0]
            # Decide once, before creating patches; the same route serves every patch.
            decision = router.predict(inputs, filename)
            if decision.route != 'baseline' and decision.route not in model.loaded_adapters:
                raise ValueError('No trained adapter loaded for route: ' + decision.route)
            set_route(model, decision.route)
            outputs = predict_density(model, inputs.to(device), device, crop_size, batch_size)
            predicted_count = outputs.sum().item()
            gt_count = count[0].item()
            row = dict(filename=filename, domain=decision.domain, route=decision.route,
                       router_source=decision.source, confidence=decision.confidence,
                       gt_count=gt_count, predicted_count=predicted_count,
                       error=gt_count - predicted_count)
            records.append(row)
            print('Image: {} | route: {} | GT: {} | predicted: {:.3f} | error: {:.3f}'.format(
                filename, decision.route, gt_count, predicted_count, row['error']))
    return records, summarize(records)


def save_results(output_dir, router_mode, records, metrics):
    directory = Path(output_dir) / (router_mode + '_' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'routing.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'count', 'mae', 'rmse'])
        writer.writeheader()
        writer.writerows(metrics)
    with (directory / 'ALGVT_sha_test.txt').open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write('{} {} {} {}\n'.format(
                row['filename'], row['gt_count'], row['predicted_count'], row['error']))
    print('Results saved to:', directory)
    return str(directory)


def test(args, isSave=True):
    if args.router_mode == 'classifier':
        make_router('classifier')  # Fail before loading files; future router factory extension point.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda')
    crop_size = args.crop_size
    dataset_name = args.dataset.lower()
    default_split = {'sha': 'test_data', 'shb': 'test_data', 'qnrf': 'test', 'nwpu': 'val'}
    split = args.split if args.split is not None else default_split.get(dataset_name, '')
    data_path = os.path.join(args.data_path, split)
    if dataset_name == 'qnrf':
        dataset = crowd.Crowd_qnrf(data_path, crop_size, 8, method='val')
    elif dataset_name == 'nwpu':
        dataset = crowd.Crowd_nwpu(data_path, crop_size, 8, method='val')
    elif dataset_name in ('sha', 'shb'):
        dataset = crowd.Crowd_sh(data_path, crop_size, 8, method='val')
    elif dataset_name == 'custom':
        dataset = crowd.CustomDataset(data_path, crop_size, downsample_ratio=8, method='test')
    else:
        raise NotImplementedError(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    model = ALTGVT.alt_gvt_large(pretrained=False)
    load_for_inference(
        model, args.model_path, args.baseline_checkpoint,
        args.dora_day_checkpoint, args.dora_night_checkpoint,
        args.adapter_route if args.router_mode == 'fixed' else 'baseline',
    )
    if args.router_mode == 'filename' and set(model.loaded_adapters) != {'dora-day', 'dora-night'}:
        raise ValueError('filename routing requires both trained dora-day and dora-night checkpoints')
    router = make_router(args.router_mode, model.inference_route,
                         args.day_prefixes, args.night_prefixes, args.unknown_route)
    model.to(device)
    model.eval()
    records, metrics = evaluate(model, dataloader, device, crop_size, args.batch_size, router)
    for metric in metrics:
        if metric['count']:
            print('{}: images={}, MAE={:.4f}, RMSE={:.4f}'.format(
                metric['group'], metric['count'], metric['mae'], metric['rmse']))
        else:
            print('{}: images=0, MAE/RMSE=N/A'.format(metric['group']))
    directory = save_results(args.output_dir, args.router_mode, records, metrics) if isSave else None
    return dict(records=records, metrics=metrics, output_dir=directory)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.model_path or not args.data_path:
        parser.error('Fill in the model-path and data-path defaults above, or provide them on the command line')
    test(args, isSave=True)
