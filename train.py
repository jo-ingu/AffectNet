# Modules
import tensorflow as tf
import numpy as np
import os
import sys
import csv
from keras.applications import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from custom_data_generator import CustomDataGenerator

# data preprocessing
def process_data(paths, labels):
    
    labels_out = []
    paths_out = []
    count = 0
   
    for paths_index, emotion in enumerate(labels):
        if emotion > 7:
            continue
        labels_out.append(emotion)
        paths_out.append(paths[paths_index])
        
        count += 1
        print('Processed:', count, end='\r')
        
    weights = class_weight.compute_class_weight('balanced', np.unique(labels_out), labels_out)
    weights = dict(enumerate(weights))
    labels_out = to_categorical(labels_out, num_classes=8)
    
    print('Processed:', count)
    return paths_out, labels_out, weights

# load labels and paths and preprocess
t_paths = np.load('Desktop/jo/AffectNet/training_paths.npy')
t_labels = np.load('Desktop/jo/AffectNet/training_labels.npy')
t_paths, t_labels, t_weights = process_data(t_paths, t_labels)

v_paths = np.load('Desktop/jo/AffectNet/validation_paths.npy')
v_labels = np.load('Desktop/jo/AffectNet/validation_labels.npy')
v_paths, v_labels, v_weights = process_data(v_paths, v_labels)

# Model
base_model = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(8, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])

# parameters and preparation of training
batch_size = 64
epochs = 10
train_gen = CustomDataGenerator(t_paths, t_labels, batch_size, shuffle=True, augment=True)
validation_gen = CustomDataGenerator(v_paths, v_labels, batch_size)
t_steps = len(t_labels) // batch_size
v_steps = len(v_labels) // batch_size

# train
history = model.fit(train_gen, steps_per_epoch=t_steps, class_weight=t_weights,
         epochs = epochs, validation_data=validation_gen, validation_steps=v_steps)
