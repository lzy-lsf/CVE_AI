def doImports():
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

# load in memory
print("loading data to memory")
nvd=nvd_load.load_json_nvd('./raw_nvd/')

print("extracting information")
items_vector=nvd_load.extract(nvd)

# folder structure
directory='NVD_severity'
if os.path.exists(directory):
    os.rename(directory, str(random.randint(100000,999999)))

severityClasses=['LOW', 'MEDIUM', 'HIGH']
os.makedirs(directory)
os.makedirs(directory+'/train')
os.makedirs(directory+'/test')
for folder in severityClasses:
    os.makedirs(directory+'/train/'+str(folder))
    os.makedirs(directory+'/test/'+str(folder))

# creating balanced train/test datasets
print("creating balanced train/test datasets")
highSevItems=[i for i in items_vector if i[0] == 'HIGH']
mediumSevItems=[i for i in items_vector if i[0] == 'MEDIUM']
lowSevItems=[i for i in items_vector if i[0] == 'LOW']

# get maximum # of items per class
lengthsOfSevClasses=[len(highSevItems), len(mediumSevItems), len(lowSevItems)]
maxItems=min(lengthsOfSevClasses)
print(F"items available per severity class {lengthsOfSevClasses}")
print("item limit:",maxItems)

# make datasets according to the limit
lowSevItems=[random.choice(lowSevItems)[1] for i in range(maxItems)]
mediumSevItems=[random.choice(mediumSevItems)[1] for i in range(maxItems)]
highSevItems=[random.choice(highSevItems)[1] for i in range(maxItems)]

# write datasets to train/test dirs (balanced)
reportSev={sev:0 for sev in severityClasses}
reportDir={'test':0, 'train':0}
for dataset,sev in [(highSevItems,'HIGH'), (mediumSevItems,'MEDIUM'), (lowSevItems,'LOW')]:
    filename=0
    for subdir in ['train', 'test']:
        datasetSub=[i for i in dataset[:len(dataset)//2]]
        for desc in datasetSub:
            filename+=1
            fp=open(F"{directory}/{subdir}/{sev}/{filename}.txt", 'w', encoding='utf-8')
            x=fp.write(desc.lower())
            fp.close()
            print(F'\rwriting files /{subdir}/{sev}/ {int(filename/maxItems*100)}%', end='', flush=True)
        datasetSub=dataset[len(dataset)//2:]
        reportDir[subdir]+=len(dataset)//2
    reportSev[sev]+=len(dataset)
print()
print(F"files in train: {reportDir['train']}\nfiles in test: {reportDir['test']}")
for key in reportSev.keys():
    print(F"{key} severity: {reportSev[key]} items")

print("Severity classes:")
for sevClass in range(len(severityClasses)):
    print(F"{sevClass} - {severityClasses[sevClass]}")

# import datasets
print('importing datasets from raw files')
batch_size=32
seed=42

# train dataset
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    F"{directory}/train",
    batch_size=batch_size,
    validation_split=0.2, # 80% will be used for training, 20% will be used for validation
    subset='training',
    seed=seed)

# validation dataset
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    F"{directory}/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

# test dataset
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    F"{directory}/test", 
    batch_size=batch_size)

print(raw_train_ds.class_names)

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
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)])

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
highPred=export_model.predict(highSevItems)
medPred=export_model.predict(mediumSevItems)
lowPred=export_model.predict(lowSevItems)
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
Settings results

0.4009
128 nodes
64 nodes
max_features = 100000
embedding_dim = 512

Todo
"""