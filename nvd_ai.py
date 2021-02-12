import os
import tensorflow as tf
import shutil
import json
import random

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import nvd_helper
print('imports done')

print("loading data to memory")
nvd=nvd_helper.load_json_nvd('./raw_nvd/')

print("extracting information")
# first keyword should be output-labels
nvd_properties=['description', 'bmv2_severity']
items_vector=nvd_helper.extract(nvd, nvd_properties)
del(nvd)

labelmap={'HIGH':2, 'MEDIUM':1, 'LOW':0}

print("data balancing")
dataset_blncd=nvd_helper.balance(labelmap, items_vector)
del(items_vector)


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


string_ds=[i[0] for i in dataset_blncd]
label_ds=[i[1] for i in dataset_blncd]
del(dataset_blncd)

label_ds=tf.data.Dataset.from_tensor_slices(label_ds)
string_ds=tf.data.Dataset.from_tensor_slices(string_ds)
dataset=tf.data.Dataset.zip((string_ds, label_ds))

# variables
batch_size=32
shuffle_size=5000
val_ratio=0.2
max_features = 5000
embedding_dim = 64

dataset=dataset.batch(batch_size)

train_ds=dataset.shard(num_shards=2, index=0)
test_ds=dataset.shard(num_shards=2, index=1)
train_ds=train_ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
# shuffle_size only for datasets which are too large and don't fit in memory

val_ds=train_ds.take(int(len(string_ds)*val_ratio))
train_ds=train_ds.skip(int(len(string_ds)*val_ratio)

print('preparing dataset for training')

# TextVectorization layer converts text to lowercase and strips punctuation by default

def avgLen(x): return len(x[0])
items_vector_len=list(map(avgLen, dataset_blncd))
sequence_length = int(sum(items_vector_len)/len(items_vector_len)) # setting sequence length to average length

vectorize_layer = TextVectorization(
    # standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int', # each token has a corresponding number
    output_sequence_length=sequence_length)

# adapt: map strings (words) to integers
# Make a text-only dataset (without labels), then call adapt
train_text = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# check result, preprocess some examples
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch (32 reviews and labels) from the dataset
"""
text_batch, label_batch = next(iter(train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
"""

# apply vectorization to all datasets so train, val, test are the same
train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

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
    layers.Dense(len(labelmap.keys()))])

# model.summary()

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer='adam', 
    metrics=['categorical_accuracy'])

# import instead of train
# model = tf.keras.models.load_model('nvd_sev.h5')

# train
history = model.fit(
    train_ds,
    validation_data=val_ds[1],
    epochs=3)

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
filestr=''
for i in nvd_properties:
    filestr+='_'+str(i)
model.save(F'nvd{filestr}.h5')


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

0.4009
128 nodes
64 nodes
max_features = 100000
embedding_dim = 512

Todo
"""
