import os
import config_distributed as config


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
from torchvision import transforms
import random
from models import WSDAN_CAL
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment, Auxiliary_Loss_v2, AttentionAlignmentLoss, DiversityLoss
from datasets import get_trainval_datasets
import math

from torch.cuda.amp import autocast, GradScaler  

cross_entropy_loss = nn.CrossEntropyLoss() 
center_loss = CenterLoss() 
auxi_loss = Auxiliary_Loss_v2(M=config.num_attentions) 
attention_alignment_loss = AttentionAlignmentLoss() 
diversity_loss = DiversityLoss(margin=0.1) 

loss_container = AverageMeter(name='loss')
top1_container = AverageMeter(name='top1')
top5_container = AverageMeter(name='top5')

raw_metric = TopKAccuracyMetric(topk=(1, 2))
crop_metric = TopKAccuracyMetric(topk=(1, 2))
drop_metric = TopKAccuracyMetric(topk=(1, 2))

best_acc = 0.0

def main():

    if hasattr(config, 'GPU'):
        os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
        torch.cuda.set_device(0)
        print(f'Using GPU {config.GPU} (visible as device 0)')
    else:
        torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

    num_classes = train_dataset.num_classes

    logs = {}
    start_epoch = 0
    net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    feature_center = torch.randn(num_classes, config.num_attentions * net.num_features).cuda() * 0.01
   
    if config.ckpt and os.path.isfile(config.ckpt):

        checkpoint = torch.load(config.ckpt)

        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        state_dict = checkpoint['state_dict']

        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

        logging.info('Network loaded from {}'.format(config.ckpt))
        print('Network loaded from {} @ {} epoch'.format(config.ckpt, start_epoch))

        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].cuda()
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    net.cuda()

    learning_rate = config.learning_rate
    print('begin with', learning_rate, 'learning rate')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    scaler = GradScaler()  

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=config.workers, pin_memory=True, drop_last=True)
    validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size * 4,
                                shuffle=False,
                                num_workers=config.workers, pin_memory=True, drop_last=True)

    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')

    print('\n' + '='*80)
    print('DualAttendMed Training Started')
    print('='*80)
    print(f'Model: ResNet-152 with {config.num_attentions} attention maps')
    print(f'Total Epochs: {config.epochs} | Batch Size: {config.batch_size}')
    print(f'Training Samples: {len(train_dataset)} | Validation Samples: {len(validate_dataset)}')
    print(f'Initial Learning Rate: {config.learning_rate}')
    print(f'GPU: {config.GPU}')
    print('='*80 + '\n')

    for epoch in range(start_epoch, config.epochs):

        callback.on_epoch_begin()
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        print(f'\n{"="*80}')
        print(f'EPOCH {epoch + 1}/{config.epochs} | Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('='*80)

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches', ncols=120)
        pbar.set_description(f'Epoch {epoch + 1}/{config.epochs}')

        train(epoch=epoch,
              logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar,
              scaler=scaler)

        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar,
                 epoch=epoch)

        torch.cuda.synchronize()

        callback.on_epoch_end(logs, net, feature_center=feature_center)
        pbar.close()

def adjust_learning(optimizer, epoch, iter):

    base_lr = config.learning_rate

    if hasattr(config, 'lr_decay_factor') and hasattr(config, 'lr_decay_epochs'):
        decay_factor = config.lr_decay_factor
        decay_epochs = config.lr_decay_epochs
        lr = base_lr * (decay_factor ** (epoch // decay_epochs))
    else:
        base_rate = 0.9
        base_duration = 2.0
        lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(**kwargs):

    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']
    scaler = kwargs.get('scaler', None)
    
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    start_time = time.time()
    net.train()
    batch_len = len(data_loader)
    for i, (X, y) in enumerate(data_loader):

        float_iter = float(i) / batch_len
        adjust_learning(optimizer, epoch, float_iter)
        now_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        X = X.cuda()
        y = y.cuda()

        forward_output = net(X, return_attention_weights=True)
        if len(forward_output) == 6:
            y_pred_raw, y_pred_aux, feature_matrix, attention_map, channel_weights, attention_maps_full = forward_output
        else:
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = forward_output
            attention_maps_full = attention_map 

        feature_center_batch = F.normalize(feature_center[y], dim=-1)

        center_momentum = feature_matrix.detach() - feature_center_batch

        feature_center[y] = config.beta * torch.mean(center_momentum, dim=0) + feature_center_batch

        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=0.5, padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=0.5)

        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)
        forward_output_refined = net(aug_images, return_attention_weights=True)
        if len(forward_output_refined) == 6:
            y_pred_aug, y_pred_aux_aug, _, _, _, attention_maps_refined = forward_output_refined
        else:
            y_pred_aug, y_pred_aux_aug, _, _ = forward_output_refined
            attention_maps_refined = None

        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)

        loss_cls_coarse = cross_entropy_loss(y_pred_raw, y)
        loss_cls_fine = cross_entropy_loss(y_pred_aug, y_aug)
        loss_cls_aux = cross_entropy_loss(y_pred_aux, y_aux)

        loss_cls = (loss_cls_coarse + loss_cls_fine * 2.0 + loss_cls_aux * 3.0) / 6.0

        if attention_maps_full is not None:
            loss_att = attention_alignment_loss(attention_maps_full)
        else:
            loss_att = torch.tensor(0.0, device=X.device)

        if attention_maps_full is not None:
            loss_div = diversity_loss(attention_maps_full)
        else:
            loss_div = torch.tensor(0.0, device=X.device)

        feature_matrix_norm = F.normalize(feature_matrix, dim=-1)
        aux_loss_value = auxi_loss(feature_matrix_norm, feature_center_batch)

        aux_loss_max = 10.0 if epoch < 5 else 5.0
        aux_loss_value = torch.clamp(aux_loss_value, max=aux_loss_max)

        batch_loss = lambda_cls * loss_cls + \
                     lambda_att * loss_att + \
                     lambda_div * loss_div + \
                     aux_loss_value * 0.01

        if scaler is not None:
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())

            loss_cls_val = loss_cls.item() if isinstance(loss_cls, torch.Tensor) else loss_cls
            loss_att_val = loss_att.item() if isinstance(loss_att, torch.Tensor) else loss_att
            loss_div_val = loss_div.item() if isinstance(loss_div, torch.Tensor) else loss_div

            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)

        aux_loss_display = aux_loss_value.item() if isinstance(aux_loss_value, torch.Tensor) else aux_loss_value
        batch_info = 'Loss: {:.3f} | Cls: {:.3f} Att: {:.3f} Div: {:.3f} Aux: {:.3f} | Train Acc: {:.2f}% (Top-1) {:.2f}% (Top-2) | LR: {:.5f}'.format(
            epoch_loss, loss_cls_val, loss_att_val, loss_div_val, aux_loss_display,
            epoch_raw_acc[0], epoch_raw_acc[1], now_lr)

        pbar.update()
        pbar.set_postfix_str(batch_info)

    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):

    global best_acc
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()

    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X = X.cuda()
            y = y.cuda()

            y_pred_raw, y_pred_aux, _, attention_map = net(X)

            if attention_map.dim() == 3:
                attention_map = attention_map.unsqueeze(1)
            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.5, padding_ratio=0.1)

            y_pred_crop3, y_pred_aux_crop3, _, _ = net(crop_images3)

            y_pred = (y_pred_raw + y_pred_crop3) / 2.
            y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.

            batch_loss = cross_entropy_loss(y_pred, y)

            epoch_loss = loss_container(batch_loss.item())

            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)

    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    val_info = 'Val Acc: {:.2f}%'.format(epoch_acc[0])

    pbar.set_postfix_str('{} | {}'.format(logs['train_info'], val_info))

    if epoch_acc[0] > best_acc:
        best_acc = epoch_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if aux_acc[0] > best_acc:
        best_acc = aux_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if epoch % 10 == 0:
        save_model(net, logs, 'model_epoch%d.pth' % epoch)

    print('\n' + '='*80)
    print(f'EPOCH {epoch + 1}/{config.epochs} VALIDATION RESULTS')
    print('='*80)
    print(f'  Validation Loss:     {epoch_loss:.4f}')
    print(f'  Validation Accuracy: {epoch_acc[0]:.2f}% (Top-1) | {epoch_acc[1]:.2f}% (Top-2)')
    print(f'  Auxiliary Accuracy:  {aux_acc[0]:.2f}% (Top-1) | {aux_acc[1]:.2f}% (Top-2)')
    print(f'  Best Accuracy So Far: {best_acc:.2f}%')
    if epoch_acc[0] > best_acc or aux_acc[0] > best_acc:
        print('  NEW BEST MODEL SAVED!')
    print('='*80 + '\n')
    

    batch_info = 'Val Loss: {:.4f} | Val Acc: {:.2f}% (Top-1) {:.2f}% (Top-2) | Aux Acc: {:.2f}% (Top-1) {:.2f}% (Top-2) | Best: {:.2f}%'.format(
        epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_acc)

    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


def save_model(net, logs, ckpt_name):

    save_path = os.path.join(config.save_dir, ckpt_name)
    torch.save({'logs': logs, 'state_dict': net.state_dict()}, save_path)
    print(f'Model saved to: {save_path}')


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
