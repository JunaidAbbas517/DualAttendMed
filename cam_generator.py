import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class CAMGenerator:
    def __init__(self, model, device='cuda'):
        self.model = model.eval()
        self.device = device
        self.model.to(device)

        if hasattr(model, 'layer4'):
            self.target_layer = model.layer4
        elif hasattr(model, 'backbone'):
            backbone = model.backbone
            if isinstance(backbone, nn.Sequential):
                for module in reversed(list(backbone.children())):
                    if hasattr(module, 'expansion'):
                        self.target_layer = module
                        break
                else:
                    self.target_layer = list(backbone.children())[-1]
            else:
                self.target_layer = backbone.layer4 if hasattr(backbone, 'layer4') else backbone
        elif hasattr(model, 'features') and hasattr(model.features, 'layer4'):
            self.target_layer = model.features.layer4
        else:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.target_layer = module

        self.feature_maps = None
        self.hook_handle = None

    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    def generate_cam(self, images, labels=None):
        images = images.to(self.device)
        batch_size = images.size(0)

        if self.hook_handle is None:
            self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

        with torch.no_grad():
            outputs = self.model(images)

            if labels is None:
                labels = torch.argmax(outputs, dim=1)
            else:
                labels = labels.to(self.device)

            feature_maps = self.feature_maps

            if hasattr(self.model, 'fc'):
                weights = self.model.fc.weight
            elif hasattr(self.model, 'classifier'):
                if isinstance(self.model.classifier, nn.Sequential):
                    for layer in self.model.classifier:
                        if isinstance(layer, nn.Linear):
                            weights = layer.weight
                            break
                else:
                    weights = self.model.classifier.weight
            else:
                raise ValueError("Cannot find classifier layer in model")

            cam_maps = []
            for i in range(batch_size):
                class_weights = weights[labels[i]]
                class_weights = class_weights.view(-1, 1, 1)
                cam = (class_weights * feature_maps[i]).sum(dim=0)
                cam = F.relu(cam)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cam_maps.append(cam.cpu().numpy())

        return np.array(cam_maps)

    def threshold_cam_to_mask(self, cam_maps, top_percent=0.2):
        if cam_maps.ndim == 2:
            cam_maps = cam_maps[np.newaxis, ...]
            squeeze_output = True
        else:
            squeeze_output = False

        binary_masks = []
        for cam in cam_maps:
            flat_cam = cam.flatten()
            threshold_idx = int(len(flat_cam) * (1 - top_percent))
            threshold = np.partition(flat_cam, threshold_idx)[threshold_idx]
            binary_mask = (cam >= threshold).astype(np.float32)
            binary_masks.append(binary_mask)

        binary_masks = np.array(binary_masks)
        if squeeze_output:
            binary_masks = binary_masks[0]

        return binary_masks

    def generate_cam_masks_for_dataset(self, data_loader, save_path=None, top_percent=0.2):
        all_cam_masks = []
        all_labels = []

        for i, (images, labels) in enumerate(tqdm(data_loader, desc="Generating CAM")):
            cam_maps = self.generate_cam(images, labels)
            cam_masks = self.threshold_cam_to_mask(cam_maps, top_percent=top_percent)
            all_cam_masks.append(cam_masks)
            all_labels.append(labels.numpy())

        all_cam_masks = np.concatenate(all_cam_masks, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            np.savez(save_path, cam_masks=all_cam_masks, labels=all_labels)

        return all_cam_masks, all_labels

    def __del__(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()


def create_baseline_resnet152(num_classes, pretrained=True, checkpoint_path=None):
    model = models.resnet152(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    return model


def extract_backbone_from_trained_model(trained_model, num_classes):
    if hasattr(trained_model, 'module'):
        model = trained_model.module
    else:
        model = trained_model

    if not hasattr(model, 'features'):
        raise ValueError("Model does not have 'features' attribute.")

    backbone = model.features

    if hasattr(model, 'num_features'):
        num_features = model.num_features
    else:
        last_layer = list(backbone.children())[-1]
        if hasattr(last_layer, 'expansion'):
            num_features = 512 * last_layer.expansion
        else:
            num_features = 2048

    class CAMModel(nn.Module):
        def __init__(self, backbone, num_features, num_classes):
            super(CAMModel, self).__init__()
            self.backbone = backbone
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(num_features, num_classes)
            self.layer4 = list(backbone.children())[-1]

        def forward(self, x):
            x = self.backbone(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    cam_model = CAMModel(backbone, num_features, num_classes)
    nn.init.normal_(cam_model.fc.weight, 0, 0.01)
    nn.init.zeros_(cam_model.fc.bias)

    return cam_model
