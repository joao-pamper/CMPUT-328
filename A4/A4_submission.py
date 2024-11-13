import numpy as np
import torch

import os
import functools
from tqdm import tqdm

import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from A4_train import get_model_instance_segmentation

class MNISTDDRGB(torch.utils.data.Dataset):
    def __init__ (self, imgs, transforms):
        self.transforms = transforms
        self.imgs = imgs

    def __getitem__(self, idx):
        img = self.img[idx, ...].reshape((64, 64, 3)).astyoe(np.uint8)
        img = img.transpose((2, 0, 1))

        img = tv_tensors.Image(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img
    
    def __len__(self):
        return self.imgs.shape[0]

def load_yolov5():    
    model = torch.hub.load("ultralytics/yolov5", "custom", path="path/to/best.pt")  # local model

def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.zeros((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.zeros((N, 4096), dtype=np.int32)

    # add your code here to fill in pred_class and pred_bboxes
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(
        num_classes=10, 
        
        box_score_thresh = 0,
        box_nms_thresh=0.8,

        v2=1,

        # min_size=64
        # max_size=64
        )
    batch_size=10

    ckpt_path = os.path.join('ckpt_v2', 'step_x.pth')

    ckpt = torch.load(ckpt_path, map_location=device)

    load_str = (f'Loading weights from: {ckpt_path} with:\n'
                f'\ttimestamp: {ckpt["timestamp"]}\n')
    
    print(load_str)

    mask_eps = 0.5

    model.to(device)
    model.load_state_dict(ckpt['model'])

    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    transforms =  T.Compose(transforms)

    test_dataset = MNISTDDRGB(images, transforms)

    test_data_loader = torch.utils.data.DataLoades(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=utils.collate_fn
    )
    model.eval()

    img_id = -1
    seg_mask = np.full((64,64), 10, dtype=np.int32)

    with torch.no_grad():
        for images in tqdm(test_data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for image, output in zip(images, outputs, strict=True):
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].cpu().numpy().squeeze()
                scores = output['scores'].cpu().numpy()

                seg_mask.fill(10)
                img_id += 1

                if len(labels) == 0:
                    raise AssertionError('no onects found')
                elif len(labels) == 1:
                    # label_1 = labels[0]
                    # box_1 = boxes[0, ...]
                    # mask_1 = masks[0, ...]
                    # mask_1_bool = mask_1 > mask_eps
                    # seg_mask[mask_1_bool] = label_1
                    # pred_class[img_id, 0] = label_1
                    # pred_bboxes[img_id, 0, :] = box_1
                    # pred_seg[img_id, :] = seg_mask.flatten() 
                    raise AssertionError('inly one object found')
                

                best_idx = [0,1]

                boxes = boxes[best_idx]
                labels = labels[best_idx]
                masks = masks[best_idx]
                scores = scores[best_idx]

                label_1, label_2 = labels[:2]
                box_1, box_2 = boxes[:2, ...]
                mask_1, mask_2 = masks[:2, ...]

                if label_1 > label_2:
                    label_1, label_2 = label_2, label_1
                    box_1, box_2 = box_2, box_1
                    mask_1, mask_2 = mask_2, mask_1

                mask_1_bool = mask_1 > mask_eps
                mask_2_bool = mask_2 > mask_eps

                seg_mask[mask_1_bool] = label_1
                seg_mask[mask_2_bool] = label_2

                pred_class[img_id, :] = label_1, label_2
                pred_bboxes[img_id, 0, :] = box_1
                pred_bboxes[img_id, 1, :] = box_2
                pred_seg[img_id, :] = seg_mask.flatten()

        print()

    return pred_class, pred_bboxes, pred_seg
