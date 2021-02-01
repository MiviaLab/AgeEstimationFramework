import sys
from tqdm import tqdm
from datetime import datetime
from map_object import Map
from train_adience import run_train

datetime_str = datetime.today().strftime('%Y%m%d_%H%M%S')
net = sys.argv[1]
out_path = sys.argv[2]


general_lr = "0.0005:0.2:20"
vgg16_lr = "0.000005:0.2:20"

LR = vgg16_lr if 'vgg16' in net else general_lr
WEIGHT_DECAY = 0.0005
MOMENTUM = True
TRAINING_EPOCHS = 35
BATCH_SIZE = 32
PREPROCESSING = "vggface2"
AUGMENTATION = "default"

train_fold_sequence = [
	["fold_0", "fold_1", "fold_2", "fold_3"],
	["fold_1" ,"fold_2", "fold_3", "fold_4"],
	["fold_2", "fold_3", "fold_4", "fold_0"],
	["fold_3", "fold_4", "fold_0", "fold_1"],
	["fold_4", "fold_0", "fold_1", "fold_2"]
]

#val_fold_sequence = [["fold_3"],["fold_4"],["fold_0"],["fold_1"],["fold_2"]]
val_fold_sequence = [[],[],[],[],[]]
test_fold_sequence = ["fold_4","fold_0","fold_1","fold_2","fold_3"]

for train_folds, val_fold in zip(train_fold_sequence, val_fold_sequence):
	params = Map(
		datetime=datetime_str,
		net=net,
		out_path=out_path,
		train_folds=train_folds,
		val_folds=val_fold,
		lr=LR,
		n_training_epochs=TRAINING_EPOCHS,
		batch_size=BATCH_SIZE,
		preprocessing=PREPROCESSING,
		augmentation=AUGMENTATION,
		weight_decay=WEIGHT_DECAY,
		momentum=MOMENTUM
	)
	print("Training folds", train_folds)
	print("Validation folds", val_fold)
	run_train(params)
	import keras.backend as K
	K.clear_session()
