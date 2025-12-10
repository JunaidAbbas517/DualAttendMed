


GPU = '2' 
workers = 4 
batch_size = 32 

image_size = (224, 224) 
net = 'resnet152' 
num_attentions = 32  
beta = 5e-2  

visual_path = None 


tag = 'CT' 


ckpt = './test_CT_224_M8/dualattendmed-resnet152/model_bestacc.pth'


use_cam_iou = True 
cam_baseline_model_path = None  

cam_top_percent = 0.2  
cam_masks_save_path = '/test_CT_224_M8/dualattendmed-resnet152/cam_masks.npz'  # 
