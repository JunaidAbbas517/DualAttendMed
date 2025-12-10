import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms




class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)
    

class Auxiliary_Loss_v2(nn.Module):
    def __init__(self,M,alpha=0.05,margin=1,inner_margin=[0.1]):
        super().__init__()
  
        self.M = M
        self.alpha = torch.tensor(alpha)
       
     
        self.margin=torch.Tensor(margin)
   
        self.register_buffer('inner_margin',torch.Tensor(inner_margin))

    def forward(self,feature_matrix,feature_centers):
        feature_centers = feature_centers.detach().cuda()
        feature_matrix = feature_matrix.cuda()
        inner_margin=self.inner_margin[0]
        
     
        feature_matrix_norm = F.normalize(feature_matrix, dim=-1)
        feature_centers_norm = F.normalize(feature_centers, dim=-1)
        
  
        diff_norm = torch.norm(feature_matrix_norm - feature_centers_norm, dim=-1)
      
        intra_class_loss = F.relu(diff_norm - inner_margin.cuda())
        intra_class_loss = torch.mean(intra_class_loss)
      
        intra_class_loss = torch.clamp(intra_class_loss, max=10.0)

        batch_size = feature_centers_norm.shape[0]

        num_features = feature_centers_norm.shape[1] // self.M
        feature_centers_reshaped = feature_centers_norm.reshape(batch_size, self.M, num_features)
        
        inter_class_loss=0
        for j in range(self.M):
            for k in range(j+1,self.M):
            
                dist = torch.norm(feature_centers_reshaped[:,j] - feature_centers_reshaped[:,k], dim=1)  # (batch_size,)
            
                inter_class_loss+= F.relu(self.margin.cuda()-torch.mean(dist), inplace=False)
       
        inter_class_loss = inter_class_loss / self.M / self.alpha
 
        inter_class_loss = torch.clamp(inter_class_loss, max=100.0)
     
        return intra_class_loss+inter_class_loss


class AttentionAlignmentLoss(nn.Module):
    """
  
    """
    def __init__(self):
        super(AttentionAlignmentLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, attention_maps, target_attention_maps=None):
        """
        
        """
        if target_attention_maps is None:
     
            target_attention_maps = torch.mean(attention_maps, dim=1, keepdim=True)
            target_attention_maps = target_attention_maps.expand_as(attention_maps)
        
  
        loss = self.mse_loss(attention_maps, target_attention_maps)
        return loss


class DiversityLoss(nn.Module):
    """

    """
    def __init__(self, margin=0.1):
        super(DiversityLoss, self).__init__()
    
        self.margin = margin
    
    def forward(self, attention_maps):
        """
        Args:
       
        """
        if attention_maps.dim() == 4:
        
            B, M, H, W = attention_maps.shape
            total_loss = 0
            for b in range(B):
                att_maps = attention_maps[b]  # (M, H, W)
                loss = self._compute_diversity(att_maps)
                total_loss += loss
            return total_loss / B
        else:
            return self._compute_diversity(attention_maps)
    
    def _compute_diversity(self, attention_maps):
        """
 
        """
        M = attention_maps.shape[0]
        total_loss = 0
        count = 0
        
        
        for i in range(M):
            for j in range(i + 1, M):
            
                hadamard = attention_maps[i] * attention_maps[j] 
            
                frobenius_norm_sq = torch.sum(hadamard ** 2)
           
                loss = F.relu(self.margin - frobenius_norm_sq)
                total_loss += loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=attention_maps.device)


class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples



class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)


def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c  
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])  
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
           

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


##################################
# transform in dataset
##################################
def get_transform(resize, phase='train'):

    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[0] / 0.875))),#[0] [1]
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[0] / 0.875))),#[0] [1]
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
