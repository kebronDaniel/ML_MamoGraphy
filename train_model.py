

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator

from pyimagesearch import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers.core import Lambda
from keras.optimizers import Adam
import keras.backend as k
import tensorflow as tf
from tensorflow.python.framework import graph_util
print(keras.__version__)
print(tf.__version__)

# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 8
BS = 32
image_shape=(255,255,3);

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(255, 255),
	color_mode="rgb",
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(255, 255),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(255, 255),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

def build_network(is_training=True):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=image_shape,  padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',name="2_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name="1_pool"))

    model.add(Conv2D(64, (3, 3), padding='same',name="3_conv"))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3, 3), padding='same',name="4_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name="2_pool"))

    model.add(Conv2D(128,(3, 3),padding='same',name="5_conv"))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3),padding='same',name="6_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name="3_pool"))

    model.add(Conv2D(256,(3, 3), padding='same',name="7_conv"))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same',name="8_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name="4_pool"))

    model.add(Flatten())
    model.add(Dense(512,name="fc_1"))
    model.add(Activation('relu'))
   
    
    if (is_training):
        #model.add(Dense(512, activation='relu'))
        #model.add(Dropout(0.5, name="drop_1"))
        model.add(Lambda(lambda x:k.dropout(x,level=0.5),name="drop_1"))
        
        
       
    model.add(Dense(3,name="fc_2"))
    model.add(Activation('softmax',name="class_result"))
    model.summary()
    return model
#tf.reset_default_graph()
#sess = tf.Session()
#k.set_session(sess)
model=build_network()

history_dict = {}
#model.compile(loss='sparse_categorical_crossentropy',optimizer = Adam(),metrics=['accuracy'])


# initialize our ResNet model and compile it
#model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	#(64, 128, 256, 512), reg=0.0005)
#opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])


H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
	)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) )

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
#print(classification_report(testGen.classes, predIdxs,
	#target_names=testGen.class_indices.keys()))

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])