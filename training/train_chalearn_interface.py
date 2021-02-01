import sys
from tqdm import tqdm
from map_object import Map
from train_chalearn import run_train

general_lr = "0.0005:0.2:70"
vgg16_lr = "0.00005:0.2:20"

LR = general_lr #vgg16_lr 
TRAINING_EPOCHS = 70
BATCH_SIZE = 128
PREPROCESSING = "vggface2"
AUGMENTATION = "default" 

net = sys.argv[1]
out_path = sys.argv[2]
gpu = sys.argv[3]

params = Map(
	net=net,
	out_path=out_path,
	lr=LR,
	n_training_epochs=TRAINING_EPOCHS,
	batch_size=BATCH_SIZE,
	selected_gpu=gpu,
	preprocessing=PREPROCESSING,
	augmentation=AUGMENTATION,
	momentum=False
)
run_train(params)

