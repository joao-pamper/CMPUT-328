
import numpy as np

import os
import torch
import functools
from typing import Any, Callable, Optional

import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks 

from engine import train_one_epoch

def to_np(self):
    arr = self.cpu().detach().numpy()
    setattr(self, 'arr', arr)
    return arr

torch.Tensor.__repr__ = to_np

import torch.utils
import torch.utils.data


class Params:
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    step_size = 3
    gamma = 0.1
    num_epochs = 100

    val_steps = 100
    val_samples = 100

    batch_size = 8
    ckpt_path = 'ckpt'
    v2 = 0
    no_resize = 1

class MNISTDDRGB(torch.utils.data.Dataset):
    def __init__ (self, imgs, labels, masks, bboxes, transforms):
        self.transforms = transforms
        self.imgs = imgs
        self.labels = labels
        self.bboxes = bboxes
        self.masks = masks

    def __getitem__(self, idx):
        img = self.imgs[idx, ...].reshape((64, 64, 3)).astype(np.uint8)
        img = img.transpose((2, 0, 1))

        y1, y2 = self.labels[idx, ...].squeeze()

        bbox_1 = self.bboxes[idx, 0, :].squeeze().astype(np.int32)
        bbox_2 = self.bboxes[idx, 1, :].squeeze().astype(np.int32)

        instance_mask = self.masks[idx, :].reshape((64, 64))

        mask_1, mask_2 = instance_mask == 1, instance_mask == 2

        masks = np.stack((mask_1, mask_2), axis=0)
        boxes = np.stack((bbox_1, bbox_2), axis=0)
        labels = torch.tensor((y1, y2,), dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((2,), dtype=torch.int64)

        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = idx
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return self.imgs.shape[0]
    
def mask_to_bbox(mask):
    y, x = np.where(mask)
    min_x, max_x = np.amin(x), np.amax(x)
    min_y, max_y = np.amin(y), np.amax(y)
    return [min_x, min_y, max_x, max_y]

def instance_to_semantic(mask, label, bbox):
    im_h, im_w = mask.shape[:2]
    inst_bbox_1 = mask_to_bbox(mask)
    min_x = bbox[0] - inst_bbox_1[0]
    min_y = bbox[1] - inst_bbox_1[1]
    max_x = min_x + im_w
    max_y = min_y + im_h

    instance_mask_1_full = np.zeros((64, 64), dtype=bool)
    instance_mask_1_full[min_y:max_y, min_x:max_x] = mask

    return instance_mask_1_full

# def maskrcnn_resnet18_fpn_v2(
#     *,
#     weights: Optional[MaskRCNN_ResNet50_FPN_V2_Weights] = None,
#     progress: bool = True,
#     num_classes: Optional[int] = None,
#     weights_backbone: Optional[ResNet50_Weights] = None,
#     trainable_backbone_layers: Optional[int] = None,
#     **kwargs: Any,
# ) -> MaskRCNN:


def get_model_instance_segmentation(num_classes, v2, **kwargs):
    if v2:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            **kwargs
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT",
            **kwargs
        )
    
    #model = torch.hub.load("ultralytics/yolov5", "yolov5s", autoshape=False, classes=num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_val_loader(val_data, val_samples, val_transforms, batch_size):
    val_images = val_data['images']

    sample_idxs = np.random.choice(range(val_images.shape[0]), val_samples, replace=False)
    val_images = val_data['images'][sample_idxs, ...]
    val_labels = val_data['labels'][sample_idxs, ...]
    val_bboxes = val_data['bboxes'][sample_idxs, ...]
    val_instance_masks = val_data['instance_masks'][sample_idxs, ...]

    val_dataset = MNISTDDRGB(val_images, val_labels, val_instance_masks, val_bboxes, val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utils.collate_fn
    )
    return val_loader

def get_transform(train):
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

import utils 

def main():
    params = Params()

    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    train_data = np.load("train.npz")

    train_images = train_data['images']
    train_labels = train_data['labels']
    train_bboxes = train_data['bboxes']
    train_instance_masks = train_data['instance_masks']

    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = MNISTDDRGB(train_images, train_labels, train_instance_masks, train_bboxes, train_transform)

    from torch.utils.tensorboard import SummaryWriter

    tb_path = os.path.join(params.ckpt_path, 'tb')
    os.makedirs(tb_path, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_path)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    model_params = dict(
        num_classes=10,
        v2=params.v2, 
        min_size = 128,
        max_size = 128,
        trainable_backbone_layers = 1
    )
    # if params.no_resize:
    #     model_params.update(dict(
    #         min_size=64,
    #         max_size=64,
    #     ))

    model = get_model_instance_segmentation(**model_params)

    model.to(device)
    
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        model_params,
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params.step_size,
        gamma=params.gamma
    )

    best_loss = None

    val_data = np.load(f"valid.npz")
    val_loader_fn = functools.partial(get_val_loader, val_data, params.val_samples, val_transform, params.batch_size)

    for epoch in range(params.num_epochs):
        best_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, writer, params.ckpt_path,
                                    best_loss, params.val_steps, val_loader_fn)
        lr_scheduler.step()

if __name__ == "__main__":
    main()


    #1:23