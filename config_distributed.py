##################################################
# Training Config - DualAttendMed 
##################################################


GPU = '2'  # Use GPU 2

workers = 0  

epochs = 40 

batch_size = 16   #8 #32 #64       
learning_rate = 1e-3       

lr_decay_factor = 0.8       
lr_decay_epochs = 10       

image_size =(224,224) 
net = 'resnet152'
num_attentions = 32 
beta = 5e-2              

tag = 'CT'       

use_cross_validation = False 

save_dir = './test_CT_224_M8/dualattendmed-resnet152/'
model_name = 'dualattendmed_model.ckpt'
log_name = 'dualattendmed_train.log'

ckpt = False
