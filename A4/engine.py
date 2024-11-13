import math
import sys
import time
import os
from datetime import datetime

from tqdm import tqdm
import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, ckpt_dir, best_loss,
                    val_steps, val_loader_fn, scaler=None):
    # model.train()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    n_steps = len(data_loader)
    pbar = tqdm(data_loader, position=0, leave=True)
    for step_id, (images, targets) in enumerate(pbar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        global_step = step_id + epoch * n_steps

        writer.add_scalar(f'train/loss', loss_value, global_step)
        for loss_name, loss_val in loss_dict_reduced.items():
            writer.add_scalar(f'train/{loss_name}', loss_val.item(), global_step)
        
        save_ckpt = False
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            save_ckpt = True

        if val_steps > 0 and (global_step + 1) % val_steps == 0:
            val_loader = val_loader_fn()
            coco_evaluator = evaluate(model, val_loader, device=device)
            for metric_name, metric_eval in coco_evaluator.coco_eval.items():
                for metric_id, metric_val in enumerate(metric_eval.stats):
                    writer.add_scalar(f'val/{metric_name}_{metric_id}', metric_val, global_step)
            model.train()

        if global_step % 100 == 0:
            save_ckpt = True

        if save_ckpt:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pth")
            timestamp = datetime.now().strftime("%y/%m/%d %H:%M:%S.%f")
            model_dict = {
                'model': model.state_dict(),
                'loss': loss_value,
                'step': global_step,
                'epoch': epoch,
                'timestamp': timestamp,
            }
            pbar.set_description(f'{timestamp} :: epoch: {epoch} step: {global_step} loss: {loss_value}') #loss should be 0.1
            torch.save(model_dict, ckpt_path)

        # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return best_loss


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in tqdm(data_loader, desc='validating', position=0, leave=True):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        # evaluator_time = time.time()
        coco_evaluator.update(res)
        
        # evaluator_time = time.time() - evaluator_time
        # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
