#! /usr/bin/python3
# coding=utf-8
#####################

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import random

BATCH_SIZE = 32

# tf.compat.v1.enable_eager_execution()

# create a folder to store resized flower images
resize_img = './calibration/'
if os.path.exists(resize_img) is False:
	os.mkdir(resize_img)

# Go through the example images and resize to 128 x128
flower_pathes = ['./flowers/daisy', './flowers/dandelion', './flowers/rose', './flowers/sunflower', './flowers/tulip']

# image index
img_index = 0
# image label
label = 0
imagepaths, labels = list(), list()

f = open("calibration.txt", "w+")

for flower_path in flower_pathes:
	flower_list = os.listdir(flower_path)
	for i in range(0, len(flower_list)):
		img_path = os.path.join(flower_path, flower_list[i])
		#image_index start with 1
		img_index = img_index + 1
		if os.path.isfile(img_path):
			if (i < 500):
				try:
					img = cv2.imread(img_path)
					res = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
					res_imgpath = resize_img + str(img_index).zfill(8) + '.jpg'
					cv2.imwrite(res_imgpath, res)
					f.write(res_imgpath + " " + str(label) +"\n")
					imagepaths.append(img_path)
					labels.append([label])
				except Exception as e:
					print("Not image file") #pass
					print(img_path)
					os.remove(img_path)
					print("Delete the file")
	#label start with 0
	label += 1

f.close()

# random shuffle
imagepath = imagepaths[0]
img_path_label = list(zip(imagepaths, labels))
random.shuffle(img_path_label)
imagepaths, labels = zip(*img_path_label)
imagepaths = list(imagepaths)
labels = list(labels)

# Convert to tensor
def _read_img_function(imagepath): 
	image = tf.read_file(imagepath)
	image_tensor = tf.image.decode_jpeg(image)
	image_resize = tf.image.resize_images(image_tensor, [128, 128])
	image_resize = image_resize / 255.0
	return image_resize

# Create dataset
imagepaths_ds = tf.data.Dataset.from_tensor_slices(imagepaths)
label_ds = tf.data.Dataset.from_tensor_slices(labels)
image_ds = imagepaths_ds.map(_read_img_function)
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = image_label_ds.shuffle(buffer_size=img_index)
ds = ds.repeat()

ds = ds.batch(BATCH_SIZE)

#print("After batch: ", repr(ds))
ds = ds.prefetch(buffer_size=4)

print("======================================================")
#print(tf.compat.v1.data.get_output_shapes(dataset))

# Set the checkpoint
#checkpoint_path = "./check_point/float_model.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback

#cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#    save_weights_only=True,
#    verbose=1,
#    period=10)


model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(128, 128, 3)),
    keras.layers.Conv2D(32, (3,3), padding="same", activation=tf.nn.relu, input_shape=(128, 128, 3)),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, (3,3), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])



'''
model = keras.Sequential([
	#keras.layers.Flatten(input_shape=(128, 128, 3)),
	keras.layers.Conv2D(32, (3,3), padding="same", input_shape=(128, 128, 3)),
	keras.layers.MaxPool2D(pool_size=(2,2)),
	keras.layers.Conv2D(64, (3,3), padding="same"),
	keras.layers.MaxPool2D(pool_size=(2,2)),
	keras.layers.Conv2D(128, (3,3), padding="same"),
	keras.layers.MaxPool2D(pool_size=(2,2)),
	keras.layers.Flatten(),
	#keras.layers.Dense(512, activation=tf.nn.relu),
	#keras.layers.Dense(200, activation=tf.nn.relu),
	keras.layers.Dense(100),
	keras.layers.Dense(5)
])
'''

model.compile(optimizer=keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

steps = int(img_index / BATCH_SIZE)
#model.fit(ds, steps_per_epoch= steps, epochs=10, callbacks = [cp_callback])
model.fit(ds, steps_per_epoch= steps, epochs=10)

model.save("flower_classification_weights.h5")

