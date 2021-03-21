import os
import tensorflow as tf
import shutil
import json
import random
import datetime

import nvd_helper
import importlib
def helper_reload():
    importlib.reload(nvd_helper)

# model_type='int' 'string' 'bool'

# put everything in classes

# class for raw nvd
# class for datasets and preprocessing (int and string superclasses)
# class for model


string=['cvssv3_properties']
label=['cvssv3_base_score']
nvd_properties=(string, label)
directory='./raw_nvd/'
modelstr='nvd'
for i in nvd_properties:
    for j in i:
        modelstr+='_'+j


nvd=nvd_helper.load_json_nvd(directory)

cve_item=nvd['nvdcve-1.1-2019.json']['CVE_Items'][0]

items_vector=nvd_helper.extract(nvd, nvd_properties)

# labelmap=list(set([i[1] for i in items_vector]))
labelmap=list(set([i[1][0] for i in items_vector]))
labelmap={labelmap[i]:i for i in range(len(labelmap))}
print(labelmap)


dataset_blncd=nvd_helper.balance(labelmap, items_vector)


"""
legacy code

legacy = tf.keras.preprocessing.text_dataset_from_directory(
    F"NVD_severity/train",
    label_mode='int',
    batch_size=32,
    validation_split=0.2, # 80% will be used for training, 20% will be used for validation
    subset='training',
    seed=42)
"""

# variables/settings
batch_size= 32
shuffle_size= 5000
max_features= 5000
embedding_dim= 64
epoch_num= 3

# make dataset

# average length of strings
def avgLen(x): return len(x[0])
ds_len=list(map(avgLen, dataset_blncd))

train_ds, val_ds, test_ds = nvd_helper.make_ds(dataset_blncd, batch_size, shuffle_size)

# preprocess

train_ds, val_ds, test_ds = nvd_helper.preprocess(train_ds, val_ds, test_ds, batch_size, ds_len, max_features)


# compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1, embedding_dim), # embedding layer creates an efficient, dense representation in which similar words have a similar encoding, improves through training
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(), # reduces the length of each input vector to the average sequence length of all vectors
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(labelmap.keys()))])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer='adam', 
    metrics=['categorical_accuracy'])

# model.summary()


history=model.fit(
    train_ds,
    epochs=epoch_num,
    validation_data=val_ds)

# import instead of train
# model = tf.keras.models.load_model('nvd_sev.h5')


# evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# export the model
# includes the TextVectorization layer in the model
def export():
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')])
    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])
    return export_model

# save model
model.save(F'{modelstr}.h5')


"""
legacy code

# manually confirm accuracy
highPred=export_model.predict(class2)
medPred=export_model.predict(class1)
lowPred=export_model.predict(class0)
numItems=len(highPred)

stats=[0,0,0]
for i in highPred:
    if i[0] == max(i):
        stats[0]+=1

for i in medPred:
    if i[1] == max(i):
        stats[1]+=1

for i in lowPred:
    if i[2] == max(i):
        stats[2]+=1

avg_p=sum(stats)/3/14672
"""


"""
Settings results
0.3689
3 epoch
max_features = 5000
embedding_dim = 64

0.3130
3 epoch
max_features = 10000
embedding_dim = 128

0.3418
3 epoch
max_features= 100000
embedding_dim= 512

0.3703
3 epoch
max_features = 5000
embedding_dim = 64
2x additional 64 nodes

Todo
"""
