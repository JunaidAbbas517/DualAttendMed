import os
import config_infer as config

if hasattr(config, 'GPU'):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    print(f'Setting CUDA_VISIBLE_DEVICES={config.GPU} (GPU {config.GPU} will be visible as device 0)')

import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from models import WSDAN_CAL
from datasets import get_trainval_datasets
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
from grand_cam_utils import GradCAM, show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt
from cam_generator import CAMGenerator, extract_backbone_from_trained_model
from torchvision import transforms

assert torch.cuda.is_available()

device = torch.device("cuda:0")
print(f'Using device: {device}')
torch.backends.cudnn.benchmark = True

cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 2))
crop_metric = TopKAccuracyMetric(topk=(1, 2))
drop_metric = TopKAccuracyMetric(topk=(1, 2))

best_acc = 0.0

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def calculate_iou(predicted_attention, ground_truth_mask, threshold=0.5, use_top_percent=None):
    if use_top_percent is not None:
        flat_att = predicted_attention.flatten().cpu().numpy()
        num_pixels = len(flat_att)
        threshold_idx = int(num_pixels * (1 - use_top_percent))
        if threshold_idx < num_pixels and threshold_idx >= 0:
            threshold_value = np.partition(flat_att, threshold_idx)[threshold_idx]
        else:
            threshold_value = flat_att.min()
        threshold_value = torch.tensor(threshold_value, dtype=predicted_attention.dtype, device=predicted_attention.device)
        pred_binary = (predicted_attention >= threshold_value).float()
    else:
        pred_binary = (predicted_attention >= threshold).float()
    intersection = (pred_binary * ground_truth_mask).sum()
    union = pred_binary.sum() + ground_truth_mask.sum() - intersection
    iou = intersection / (union + 1e-8)
    return iou.item() if isinstance(iou, torch.Tensor) and iou.numel() == 1 else iou


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() +
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])
    return torch.stack(heat_attention_maps, dim=1)


def main():
    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)
    validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.workers, pin_memory=True)
    num_classes = validate_dataset.num_classes if hasattr(validate_dataset, 'num_classes') else 2
    logs = {}
    start_epoch = 0
    net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    checkpoint = torch.load(config.ckpt)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    print('Network loaded from {}'.format(config.ckpt))

    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    if config.visual_path is not None:
        visualization(net)

    test(data_loader=validate_loader, net=net, num_classes=num_classes)


def visualize(**kwargs):
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()
    start_time = time.time()
    net.eval()
    savepath = config.visual_path
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X = X.to(device)
            predictions, attention_maps = net.visualize(X)
            attention_maps = torch.max(attention_maps, dim=1, keepdim=True)[0]
            attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
            attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())
            heat_attention_maps = generate_heatmap(attention_maps)
            raw_image = X.cpu() * STD + MEAN
            heat_attention_image = raw_image * 0.2 + heat_attention_maps * 0.8
            for batch_idx in range(X.size(0)):
                rimg = ToPILImage(raw_image[batch_idx])
                haimg = ToPILImage(heat_attention_image[batch_idx])
                rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
                haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * config.batch_size + batch_idx)))
            print('iter %d / %d done!' % (i, len(data_loader)))


def visualization(model):
    model.eval()
    target_layers = [model.attentions]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    image_path = ""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).cuda()
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 1
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.imsave('cam_nodouc_M32_438_224.jpg', visualization)
    plt.show()


def test(**kwargs):
    global best_acc
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    num_classes = kwargs.get('num_classes', 2)
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()
    cam_masks = None

    if hasattr(config, 'use_cam_iou') and config.use_cam_iou:
        print("\n" + "="*80)
        print("Using trained DualAttendMed model's backbone (better than random baseline)")
        print("="*80)
        if hasattr(config, 'cam_masks_save_path') and config.cam_masks_save_path:
            if os.path.exists(config.cam_masks_save_path):
                print(f"Loading pre-computed CAM masks from {config.cam_masks_save_path}")
                cam_data = np.load(config.cam_masks_save_path)
                cam_masks = cam_data['cam_masks']
                print(f"Loaded {len(cam_masks)} CAM masks")
            else:
                print("Extracting ResNet-152 backbone from trained DualAttendMed model...")
                baseline_model = extract_backbone_from_trained_model(net, num_classes)
                baseline_model.to(device)
                baseline_model.eval()
                cam_generator = CAMGenerator(baseline_model, device=device)
                cam_top_percent = getattr(config, 'cam_top_percent', 0.2)
                cam_masks, _ = cam_generator.generate_cam_masks_for_dataset(
                    data_loader,
                    save_path=config.cam_masks_save_path,
                    top_percent=cam_top_percent
                )
                print(f"Generated {len(cam_masks)} CAM masks (top {cam_top_percent*100}% pixels)")
                del baseline_model, cam_generator
                torch.cuda.empty_cache()
        else:
            print("Extracting ResNet-152 backbone from trained DualAttendMed model...")
            baseline_model = extract_backbone_from_trained_model(net, num_classes)
            baseline_model.to(device)
            baseline_model.eval()
            cam_generator = CAMGenerator(baseline_model, device=device)
            cam_top_percent = getattr(config, 'cam_top_percent', 0.2)
            cam_masks, _ = cam_generator.generate_cam_masks_for_dataset(
                data_loader,
                save_path=None,
                top_percent=cam_top_percent
            )
            print(f"Generated {len(cam_masks)} CAM masks (top {cam_top_percent*100}% pixels)")
            del baseline_model, cam_generator
            torch.cuda.empty_cache()

        print("="*80 + "\n")

    start_time = time.time()
    net.eval()

    with torch.no_grad():
        sum_pred = []
        sum_y = []
        y_pred_ = []
        iou_values = []
        cam_mask_idx = 0

        for i, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)
            X_m = torch.flip(X, [3])
            y_pred_raw, y_pred_aux_raw, _, attention_map = net(X)
            X_m = torch.flip(X, [3])
            y_pred_raw_m, y_pred_aux_raw_m, _, attention_map_m = net(X_m)
            if attention_map.dim() == 3:
                attention_map = attention_map.unsqueeze(1)
            if attention_map_m.dim() == 3:
                attention_map_m = attention_map_m.unsqueeze(1)
            crop_images = batch_augment(X, attention_map, mode='crop', theta=0.5, padding_ratio=0.1)
            crop_images_m = batch_augment(X_m, attention_map_m, mode='crop', theta=0.5, padding_ratio=0.1)
            y_pred_crop, y_pred_aux_crop, _, _ = net(crop_images)
            y_pred_crop_m, y_pred_aux_crop_m, _, _ = net(crop_images_m)
            y_pred_m = (y_pred_raw_m + y_pred_crop_m) / 2.
            y_pred = (y_pred + y_pred_m) / 2.
            y_pred_aux = (y_pred_aux_raw + y_pred_aux_crop) / 2.
            y_pred_aux_m = (y_pred_aux_raw_m + y_pred_aux_crop_m) / 2.
            y_pred_aux = (y_pred_aux + y_pred_aux_m) / 2.
            for batch_idx in range(y_pred.shape[0]):
                y_pred_.append(np.argmax(y_pred[batch_idx].cpu().numpy()))

            if cam_masks is not None:
                attention_map_batch = attention_map
                if attention_map_batch.dim() == 4:
                    attention_map_batch = attention_map_batch.squeeze(1)
                elif attention_map_batch.dim() == 2:
                    attention_map_batch = attention_map_batch.unsqueeze(0)
                attention_map_batch = attention_map_batch.cpu().numpy()
                for b in range(attention_map_batch.shape[0]):
                    att_map = attention_map_batch[b]
                    att_map_min = att_map.min()
                    att_map_max = att_map.max()
                    if att_map_max > att_map_min + 1e-8:
                        att_map = (att_map - att_map_min) / (att_map_max - att_map_min + 1e-8)
                    else:
                        att_map = np.zeros_like(att_map)
                    if cam_mask_idx < len(cam_masks):
                        cam_mask = cam_masks[cam_mask_idx]
                        att_map_tensor = torch.from_numpy(att_map).float()
                        cam_mask_tensor = torch.from_numpy(cam_mask).float()
                        if att_map_tensor.shape != cam_mask_tensor.shape:
                            target_size = cam_mask_tensor.shape
                            att_map_tensor = F.interpolate(
                                att_map_tensor.unsqueeze(0).unsqueeze(0),
                                size=target_size,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        cam_top_percent = getattr(config, 'cam_top_percent', 0.2)
                        iou = calculate_iou(att_map_tensor, cam_mask_tensor, use_top_percent=cam_top_percent)
                        iou_values.append(iou)
                        cam_mask_idx += 1

            epoch_acc = raw_metric(y_pred, y)
            sum_y.extend(y.cpu().numpy())
            if i % 5 == 0:
                print('Batch {}/{}: Accuracy ({:.2f}, {:.2f})'.format(i, len(data_loader), epoch_acc[0], epoch_acc[1]))

        sum_pred = np.array(y_pred_)
        sum_y = np.array(sum_y)
        accuracy = accuracy_score(sum_y, sum_pred) * 100.0

        if len(iou_values) > 0:
            mean_iou = np.mean(iou_values)
            std_iou = np.std(iou_values)
            print('\n' + '='*80)
            print('='*80)
            print('  Accuracy: {:.2f}%'.format(accuracy))
            print('  IoU (CAM-based): {:.4f} ± {:.4f}'.format(mean_iou, std_iou))
            print('        (CAM maps from trained ResNet-152 backbone, top {}% pixels)'.format(
                getattr(config, 'cam_top_percent', 0.2) * 100))
            print('='*80)
        else:
            print('\n' + '='*80)
            print('  Accuracy: {:.2f}%'.format(accuracy))
            print('='*80)


if __name__ == '__main__':
    main()
