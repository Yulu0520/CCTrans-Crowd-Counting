import os
import json
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh, CustomDataset

# from models import vgg19
from Networks import ALTGVT
from Networks.dora import ADAPTERS, attach_dora, configure_training
from utils.dora_options import settings_from_args
from utils.dora_checkpoint import (
    file_identity, load_baseline, read_checkpoint, restore_adapter, save_adapter,
)
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import AverageMeter
from utils.training_state import capture_random_state, restore_random_state
import utils.log_utils as log_utils
import wandb
from tqdm.auto import tqdm


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[
        1
    ]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.stage = getattr(args, "stage", "baseline")
        self.best_model_path = None
        self.latest_checkpoint_path = None
        self.resumed_best_model_path = None

    def setup(self):
        args = self.args
        sub_dir = (
            "ALTGVT/{}_12-1-input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}".format(
                args.run_name,
                args.crop_size,
                args.wot,
                args.wtv,
                args.reg,
                args.num_of_iter_in_ot,
                args.norm_cood,
            )
        )

        sub_dir += "_density-{}_wl2-{}_beta-{}_sigma-{}".format(
            args.density_loss, args.wl2, args.gaussian_beta, args.gaussian_sigma
        )
        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S-%f")
        self.save_dir = os.path.join(args.save_dir, sub_dir, self.stage, time_str)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, "train-{:s}.log".format(time_str))
        )
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info("using {} gpus".format(self.device_count))
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8
        if self.stage == "baseline":
            train_root = os.path.join(args.data_dir, "train")
            val_root = os.path.join(args.data_dir, "val")
        else:
            prefix = self.stage.replace("-", "_")
            train_root = getattr(args, prefix + "_train_dir")
            val_root = getattr(args, prefix + "_val_dir")
        if args.dataset.lower() == "qnrf":
            self.datasets = {
                x: Crowd_qnrf(
                    {"train": train_root, "val": val_root}[x], args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "nwpu":
            self.datasets = {
                x: Crowd_nwpu(
                    {"train": train_root, "val": val_root}[x], args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "sha" or args.dataset.lower() == "shb":
            self.datasets = {
                "train": Crowd_sh(
                    train_root,
                    args.crop_size,
                    downsample_ratio,
                    "train",
                ),
                "val": Crowd_sh(
                    val_root,
                    args.crop_size,
                    downsample_ratio,
                    "val",
                ),
            }
        elif args.dataset.lower() == "custom":
            self.datasets = {
                "train": CustomDataset(
                    (args.data_dir if self.stage == "baseline" else train_root),
                    args.crop_size, downsample_ratio, method="train"
                ),
                "val": CustomDataset(
                    (args.data_dir if self.stage == "baseline" else val_root),
                    args.crop_size, downsample_ratio, method="valid"
                ),
            }
        else:
            raise NotImplementedError

        train_dataset = self.datasets["train"]
        train_dataset.density_loss = args.density_loss
        train_dataset.gaussian_beta = args.gaussian_beta
        train_dataset.gaussian_sigma = args.gaussian_sigma

        self.dataloaders = {
            x: DataLoader(
                self.datasets[x],
                collate_fn=(train_collate if x ==
                            "train" else default_collate),
                batch_size=(args.batch_size if x == "train" else 1),
                shuffle=(True if x == "train" else False),
                num_workers=args.num_workers * self.device_count,
                pin_memory=(True if x == "train" else False),
            )
            for x in ["train", "val"]
        }
        self.start_epoch = 0
        self.best_mae = np.inf
        self.best_mse = np.inf
        resume = read_checkpoint(args.resume) if args.resume else None
        if resume is not None and resume.get("stage", self.stage) != self.stage:
            raise ValueError("Resume checkpoint belongs to a different training stage")
        self.model = ALTGVT.alt_gvt_large(
            pretrained=(self.stage == "baseline" and resume is None))
        if self.stage == "baseline":
            if args.resume:
                load_baseline(self.model, args.resume)
        else:
            baseline_path = args.baseline_checkpoint
            if not baseline_path and resume is not None:
                baseline_path = resume.get("baseline", {}).get("path", "")
            if not baseline_path:
                raise ValueError("Provide a trained baseline checkpoint")
            load_baseline(self.model, baseline_path)
            self.baseline_identity = file_identity(baseline_path)
            attach_dora(self.model, settings_from_args(args))
            for name in ADAPTERS:
                path = getattr(args, name.replace("-", "_") + "_checkpoint")
                if path:
                    restore_adapter(self.model, read_checkpoint(path), name, self.baseline_identity)
            if resume is not None:
                restore_adapter(self.model, resume, self.stage, self.baseline_identity)
        self.model.to(self.device)
        configure_training(self.model, self.stage)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        if resume is not None and "optimizer_state_dict" in resume:
            self.optimizer.load_state_dict(resume["optimizer_state_dict"])
            # CLI learning rate and decay deliberately override the saved optimizer settings.
            for group in self.optimizer.param_groups:
                group.update(lr=args.lr, weight_decay=args.weight_decay)
            self.start_epoch = resume["epoch"] + 1
            self.best_mae = resume.get("best_mae", np.inf)
            self.best_mse = resume.get("best_mse", np.inf)
            self.resumed_best_model_path = resume.get("best_model_path")
        self.logger.info("stage=%s, trainable parameters=%d", self.stage,
                         sum(parameter.numel() for parameter in trainable))
        config = dict(vars(args))
        if self.stage != "baseline":
            config["baseline"] = self.baseline_identity
        with open(os.path.join(self.save_dir, "config.json"), "w") as stream:
            json.dump(config, stream, indent=2)
        if args.wandb:
            self.wandb_run = wandb.init(config=config, project="CTTrans", name=args.run_name + "/" + self.stage)
        else:
            wandb.init(mode="disabled")

        self.ot_loss = OT_Loss(
            args.crop_size,
            downsample_ratio,
            args.norm_cood,
            self.device,
            args.num_of_iter_in_ot,
            args.reg,
        )
        self.tv_loss = nn.L1Loss(reduction="none").to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        # Best metrics are restored above when resuming.
        if resume is not None and "optimizer_state_dict" in resume and "random_state" in resume:
            restore_random_state(resume["random_state"])

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info(
                "-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5
            )
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
            self.save_latest_checkpoint()

    def compute_density_loss(self, outputs, outputs_normed, target, counts):
        if outputs.shape != target.shape:
            raise ValueError("Prediction and density target shapes differ: {} vs {}".format(
                outputs.shape, target.shape
            ))
        if self.args.density_loss == "l2":
            # Mean over batch, channel and pixels; targets retain their head count.
            return self.mse(outputs, target) * self.args.wl2
        target_normed = target / (counts[:, None, None, None] + 1e-6)
        return (
            self.tv_loss(outputs_normed, target_normed).sum(dim=(1, 2, 3)) * counts
        ).mean() * self.args.wtv

    def train_epoch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_density_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        configure_training(self.model, self.stage)

        progress = tqdm(
            self.dataloaders["train"],
            desc=f"Epoch {self.epoch}/{self.args.max_epoch} Train",
            unit="batch", dynamic_ncols=True,
        )
        for step, (inputs, points, gt_discrete) in enumerate(progress):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs)
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(
                    outputs_normed, outputs, points
                )
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(
                    outputs.sum(1).sum(1).sum(1),
                    torch.from_numpy(gd_count).float().to(self.device),
                )
                epoch_count_loss.update(count_loss.item(), N)

                density_loss = self.compute_density_loss(
                    outputs, outputs_normed, gt_discrete,
                    torch.from_numpy(gd_count).float().to(self.device),
                )
                epoch_density_loss.update(density_loss.item(), N)

                loss = ot_loss + count_loss + density_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = (
                    torch.sum(outputs.view(N, -1),
                              dim=1).detach().cpu().numpy()
                )
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
                progress.set_postfix(
                    loss=f"{epoch_loss.get_avg():.3f}",
                    count=f"{epoch_count_loss.get_avg():.3f}",
                    ot=f"{epoch_ot_loss.get_avg():.3f}",
                    **{self.args.density_loss: f"{epoch_density_loss.get_avg():.3f}"},
                    mae=f"{epoch_mae.get_avg():.2f}",
                    refresh=False,
                )

                # log wandb
                wandb.log(
                    {
                        "train/TOTAL_loss": loss,
                        "train/count_loss": count_loss,
                        "train/ot_loss": ot_loss,
                        "train/{}_loss".format(self.args.density_loss): density_loss,
                        "train/pred_err": pred_err,
                    },
                    step=self.epoch,
                )

        self.logger.info(
            "Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, "
            "Count Loss: {:.2f}, {} Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch,
                epoch_loss.get_avg(),
                epoch_ot_loss.get_avg(),
                epoch_wd.get_avg(),
                epoch_ot_obj_value.get_avg(),
                epoch_count_loss.get_avg(),
                self.args.density_loss.upper(),
                epoch_density_loss.get_avg(),
                np.sqrt(epoch_mse.get_avg()),
                epoch_mae.get_avg(),
                time.time() - epoch_start,
            )
        )

    def save_latest_checkpoint(self):
        """Keep one complete epoch-boundary resume file, independently of best weights."""
        path = os.path.join(self.save_dir, "{}_last_epoch-{}.ckpt".format(self.stage, self.epoch + 1))
        temporary_path = path + ".tmp"
        best_path = self.best_model_path or self.resumed_best_model_path
        state = dict(
            stage=self.stage, epoch=self.epoch,
            best_mae=float(self.best_mae), best_mse=float(self.best_mse),
            best_model_path=os.path.abspath(best_path) if best_path else None,
            optimizer_state_dict=self.optimizer.state_dict(),
            random_state=capture_random_state(),
        )
        try:
            if self.stage == "baseline":
                torch.save(dict(state, model_state_dict=self.model.state_dict(),
                                train_config=vars(self.args).copy()), temporary_path)
            else:
                save_adapter(temporary_path, self.model, self.stage, self.baseline_identity,
                             vars(self.args).copy(), **state)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        previous_path = self.latest_checkpoint_path
        self.latest_checkpoint_path = path
        if previous_path and previous_path != path and os.path.isfile(previous_path):
            os.remove(previous_path)
        self.logger.info("Saved resume checkpoint: epoch %d -> %s", self.epoch + 1, path)

    def save_best_model(self, mae, mse):
        """Save a new best before removing this run's previous best checkpoint."""
        if not np.isfinite(mae) or mae >= self.best_mae:
            return None
        prefix = "best_model" if self.stage == "baseline" else self.stage + "_best"
        filename = "{}_mae-{:.2f}_epoch-{}.pth".format(prefix, mae, self.epoch + 1)
        model_path = os.path.join(self.save_dir, filename)
        temporary_path = model_path + ".tmp"
        try:
            if self.stage == "baseline":
                torch.save(self.model.state_dict(), temporary_path)
            else:
                save_adapter(
                    temporary_path, self.model, self.stage, self.baseline_identity,
                    vars(self.args).copy(), epoch=self.epoch,
                    best_mae=float(mae), best_mse=float(mse),
                )
            os.replace(temporary_path, model_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        previous_path = self.best_model_path
        self.best_model_path = model_path
        self.best_mae, self.best_mse = float(mae), float(mse)
        if previous_path and previous_path != model_path and os.path.isfile(previous_path):
            os.remove(previous_path)
        self.logger.info("Saved best weights: epoch %d, MAE %.4f, RMSE %.4f -> %s",
                         self.epoch + 1, mae, mse, model_path)
        return model_path

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        val_abs_error = 0.0
        val_squared_error = 0.0
        progress = tqdm(
            self.dataloaders["val"],
            desc=f"Epoch {self.epoch}/{args.max_epoch} Val",
            unit="image", dynamic_ncols=True,
        )
        for inputs, count, name in progress:
            with torch.no_grad():
                # nputs = cal_new_tensor(inputs, min_size=args.crop_size)
                inputs = inputs.to(self.device)
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = map(
                    lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks)
                )

                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    crop_pred, _ = self.model(crop_imgs[gs:gt])

                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = (
                        F.interpolate(
                            crop_pred,
                            size=(h1 * 8, w1 * 8),
                            mode="bilinear",
                            align_corners=True,
                        )
                        / 64
                    )

                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                # for the overlapping area, compute average value
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
                val_abs_error += abs(res)
                val_squared_error += res * res
                progress.set_postfix(
                    mae=f"{val_abs_error / len(epoch_res):.2f}",
                    rmse=f"{np.sqrt(val_squared_error / len(epoch_res)):.2f}",
                    refresh=False,
                )
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.logger.info(
            "Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch, mse, mae, time.time() - epoch_start
            )
        )

        # log wandb
        wandb.log({"val/MSE": mse, "val/MAE": mae}, step=self.epoch)

        model_path = self.save_best_model(mae, mse)
        if model_path and args.wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_path)
            self.wandb_run.log_artifact(artifact)


def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h // factor * factor), int(w // factor * factor)
    img_tensor = F.interpolate(
        img_tensor, (h, w), mode="bilinear", align_corners=True)

    return img_tensor


def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(
                img_tensor,
                (min_size, int(min_size / h * w)),
                mode="bilinear",
                align_corners=True,
            )
        else:
            img_tensor = F.interpolate(
                img_tensor,
                (int(min_size / w * h), min_size),
                mode="bilinear",
                align_corners=True,
            )
    return img_tensor


if __name__ == "__main__":
    import torch

    print(torch.__file__)
    x = torch.ones(1, 3, 768, 1152)
    y = tensor_spilt(x)
    print(y.size())
