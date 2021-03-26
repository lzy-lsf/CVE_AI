#!/usr/bin/env python
# coding: utf-8

# ## CVE AI
# #### Classify CVE Security Vulnerabilities with Tensorflow
# 
# This Jupyter Notebook will prepare and train one or multiple datasets which can be downloaded from the [NVD Website](https://nvd.nist.gov/vuln/data-feeds#JSON_FEED).
# 
# The model will be trained with the textual description of each vulnerability and the corresponding severity High, Medium or Low.
# 
# #### Training the model
# At the moment there are 19 datasets available (from 2002 until 2021). Training performance will decrease as you add more dataset. However, according to the statistics the more datasets are included the higher the accuracy will be. This was measured using the ```categorical_accuracy``` metric.
# 
# 

# In[1]:


# Shortcut for converting the notebook to a Python script
get_ipython().system("jupyter nbconvert --to script 'CVE Tensorflow AI.ipynb'")


# In[2]:


import os
import tensorflow as tf
import shutil
import json
import random

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# In[3]:


# Read the dataset into memory

# Multiple datasets from https://nvd.nist.gov/vuln/data-feeds#JSON_FEED can be read from the NVD_DATASET directory

# Example directory structure:
# NVD_DATASET
#    - nvdcve-1.1-2020.json
#    - nvdcve-1.1-2019.json


raw_dataset_directory="./NVD_DATASET/"
directory_contents=os.listdir(raw_dataset_directory)

nvd_dataset={}

for filename in directory_contents:
    if filename[-5:] == ".json":
        print(F"processing {filename}")
        fp=open(F"{raw_dataset_directory}/{filename}", "r", encoding="utf-8")
        nvd_dataset[filename]=json.load(fp)
        fp.close()
print("done")


# In[4]:


if False:
    print(F"sample: {nvd['nvdcve-1.1-2019.json']['CVE_Items'][0]}")
    print(F"\n# of items: {len(nvd['nvdcve-1.1-2019.json']['CVE_Items'])}")


# In[5]:


# Extract the relevant data from the dataset

# The textual description as well as the severity (HIGH, MEDIUM, or LOW) will be extracted from the dataset

# Unfortunately the data structure of the JSON data is inconsistent
# The statistics can be used to improve the data extraction

passed = 0
failed = 0
item_pairs=[]

for year in nvd_dataset.keys():
    for item in nvd_dataset[year]["CVE_Items"]:
        try:
            severity = item["impact"]["baseMetricV2"]["severity"] # severity
            description = item["cve"]["description"]["description_data"][0]["value"]
            item_pairs.append((severity, description))
            passed+=1
        except:
            failed+=1
print(F"passed: {passed}\nfailed: {failed}")


# In[6]:


# Create balanced train and test datasets


# sort
high_severity_items=[i for i in item_pairs if i[0] == "HIGH"]
medium_severity_items=[i for i in item_pairs if i[0] == "MEDIUM"]
low_severity_items=[i for i in item_pairs if i[0] == "LOW"]

# find out maximum possible item count for each severity class
length_arr=[len(high_severity_items), len(medium_severity_items), len(low_severity_items)]
item_num_limit=min(length_arr)
print(F"items available per severity class {length_arr}")
print("item limit:",item_num_limit)

# balance datasets
low_severity_items=[random.choice(low_severity_items)[1] for i in range(item_num_limit)]
medium_severity_items=[random.choice(medium_severity_items)[1] for i in range(item_num_limit)]
high_severity_items=[random.choice(high_severity_items)[1] for i in range(item_num_limit)]


# In[7]:


# The datasets will be written to disk


severity_categories=["LOW", "MEDIUM", "HIGH"]
nvd_prepared_dir="NVD_PROCESSED"

def create_directory_structure():
    if os.path.exists(nvd_prepared_dir):
        print(F"directory {nvd_prepared_dir} exists")
        return
    os.makedirs(nvd_prepared_dir)
    os.makedirs(nvd_prepared_dir+"/train")
    os.makedirs(nvd_prepared_dir+"/test")
    for folder in severity_categories:
        os.makedirs(nvd_prepared_dir+"/train/"+str(folder))
        os.makedirs(nvd_prepared_dir+"/test/"+str(folder))

def write_data_to_disk():
    reportSev={sev:0 for sev in severity_categories}
    reportDir={"test":0, "train":0}
    for dataset,sev in [(high_severity_items,"HIGH"), (medium_severity_items,"MEDIUM"), (low_severity_items,"LOW")]:
        filename=0
        for subdir in ["train", "test"]:
            datasetSub=[i for i in dataset[:len(dataset)//2]]
            for desc in datasetSub:
                filename+=1
                fp=open(F"{nvd_prepared_dir}/{subdir}/{sev}/{filename}.txt", "w", encoding="utf-8")
                x=fp.write(desc.lower())
                fp.close()
            datasetSub=dataset[len(dataset)//2:]
            reportDir[subdir]+=len(dataset)//2
        reportSev[sev]+=len(dataset)
    print(F"files in train: {reportDir['train']}\nfiles in test: {reportDir['test']}")
    for key in reportSev.keys():
        print(F"{key} severity: {reportSev[key]} items")
        
def save_datasets():
    create_directory_structure()
    write_data_to_disk()

if True: save_datasets()


# In[8]:


# Create Tensorflow training, validation and test datasets

# 80% Training
# 20% Validation

severity_categories=["LOW", "MEDIUM", "HIGH"]
print("\n\nSeverity categories:")
for sevClass in range(len(severity_categories)):
    print(F"{sevClass} - {severity_categories[sevClass]}")

batch_size=32
seed=42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    F"{nvd_prepared_dir}/train",
    batch_size=batch_size,
    validation_split=0.2, # 80% will be used for training, 20% will be used for validation
    subset="training",
    seed=seed)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    F"{nvd_prepared_dir}/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    F"{nvd_prepared_dir}/test", 
    batch_size=batch_size)


# In[9]:


# Preprocess text to vector

max_features = 100000
embedding_dim = 512

# Calculate the average length of all texts
def avgLen(x): return len(x[1])
item_pairs_len=list(map(avgLen, item_pairs))
avgLen = int(sum(item_pairs_len)/len(item_pairs_len))
sequence_length = avgLen

vectorize_layer = TextVectorization(
    # standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int", # each token has a corresponding number
    output_sequence_length=sequence_length)

# adapt: map strings (words) to integers
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# check result, preprocess some examples
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# In[10]:


# apply to all sets, train, val, test must be the same
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# tuning performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[11]:


# compile the model
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim), # embedding layer creates an efficient, dense representation in which similar words have a similar encoding, improves through training
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(), # reduces the length of each input vector to the average sequence length of all vectors
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(3)])

model.summary()

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer="adam", 
    metrics=["categorical_accuracy"])


# In[ ]:


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10)

# evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


# In[ ]:


export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation("sigmoid")])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=["accuracy"])


# In[ ]:


model.save("nvd_sev.tensorflow")


# In[15]:


# manually confirm accuracy
highPred=export_model.predict(high_severity_items)
medPred=export_model.predict(medium_severity_items)
lowPred=export_model.predict(low_severity_items)
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

avg_p=sum(stats)/3/numItems
print(avg_p,numItems)


# In[ ]:




