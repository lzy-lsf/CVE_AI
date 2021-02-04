import os
import tensorflow as tf
import shutil
import json
import random

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import nvd_load

print('imports done')

print("loading data to memory")
nvd=nvd_load.load_json_nvd('./raw_nvd_limited/')

print("extracting information")
# first keyword should be keyword used for output classification
items_vector=nvd_load.extract(nvd, ['description', 'bmv2_severity'])

labels=['HIGH', 'MEDIUM', 'LOW']

# balance data according to output labels
def balance(labels, items_vector):
    dataset_orig=[]
    for lab in labels:
        dataset_orig+=[[i for i in items_vector if i[1] == lab]]
        # dataset_orig comprises lists of items for each label/class
        # we assume the first element of an item denotes the label

# get maximum # of items per class
    lengths=[len(i) for i in dataset_orig]
    item_limit=min(lengths)
    print(F"items available per class {lengths}")
    print("item limit:",item_limit)
    dataset_blncd=[]
    for dataset in dataset_orig:
        dataset_blncd+=[random.choice(dataset) for i in range(item_limit)]
    random.shuffle(dataset_blncd)
    print(F"{len(dataset_blncd)} items added")
    return dataset_blncd
    # return dataset_blncd

print("data balancing")
dataset_blncd=balance(labels, items_vector)

# split batches
# split data and labels
# split train/val/test
def split(batch_size, dataset_blncd, train_val_ratio, train_test_ratio):
    i=0
    d_batch=[]
    l_batch=[]
    datasets={'train':([],[]), 'val':([],[]), 'test':([],[])}
    for label in dataset_blncd:
        for item in label:
            i+=1
            l_batch.append(item[0])
            d_batch.append(item[1])
            if i == batch_size:
                i=0
                print(random.random())
                if random.random() > train_test_ratio:
                    if random.random() > train_val_ratio:
                        datasets['train'][0].append(d_batch)
                        datasets['train'][1].append(l_batch)
                    else:
                        datasets['val'][0].append(d_batch)
                        datasets['val'][1].append(l_batch)
                else:
                    datasets['test'][0].append(d_batch)
                    datasets['test'][1].append(l_batch)
                d_batch=[]
                l_batch=[]
    print(F"{len(datasets['train'][0])} batches in train dataset")
    print(F"{len(datasets['val'][0])} batches in validate dataset")
    print(F"{len(datasets['test'][0])} batches in test dataset")
    print(F"train/test datasets ratio {train_test_ratio}")
    print(F"train/validate datasets ratio {train_val_ratio}")
    raw_ds=[]
    for key in datasets.keys():
        data_tensor=tf.data.Dataset.from_tensor_slices(datasets[key][0])
        label_tensor=tf.data.Dataset.from_tensor_slices(datasets[key][1])
        raw_ds+=[tf.data.Dataset.zip((data_tensor, label_tensor))]
    return raw_ds
    # merge datasets into train/val/test datasets

raw_ds=split(32, dataset_blncd, 0.2, 0.5)
"""

print(raw_ds[0].class_names)

# prepare the dataset for training
print('preparing dataset for training')

# TextVectorization layer converts text to lowercase and strips punctuation by default
max_features = 100000
embedding_dim = 512

def avgLen(x): return len(x[1])
items_vector_len=list(map(avgLen, items_vector))
avgLen = int(sum(items_vector_len)/len(items_vector_len))
sequence_length = avgLen # setting sequence length to average length

vectorize_layer = TextVectorization(
    # standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int', # each token has a corresponding number
    output_sequence_length=sequence_length)

# adapt: map strings (words) to integers
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# check result, preprocess some examples
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch (32 reviews and labels) from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))
# print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
# print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# apply to all sets, train, val, test must be the same
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# tuning performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# compile the model
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim), # embedding layer creates an efficient, dense representation in which similar words have a similar encoding, improves through training
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(), # reduces the length of each input vector to the average sequence length of all vectors
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(labels))])

model.summary()

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer='adam', 
    metrics=['categorical_accuracy'])

# import instead of train
# model = tf.keras.models.load_model('nvd_sev.h5')

# train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5)

# evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# export the model
# includes the TextVectorization layer in the model
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

# save model
model.save('nvd_sev.h5')

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


""
Settings results

0.4009
128 nodes
64 nodes
max_features = 100000
embedding_dim = 512

Todo
"""
