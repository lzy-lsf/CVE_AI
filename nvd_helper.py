import os
import json
import random
import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def load_json_nvd(directory):
    nvd={}
    dir_list=os.listdir(directory)
    for filename in dir_list:
        if '.json' in filename:
            fp=open(F"{directory}/{filename}", 'r', encoding='utf-8')
            nvd[filename]=json.load(fp)
            fp.close()
        print(F'\rloading data {int(dir_list.index(filename)/len(os.listdir(directory))*100)}%', end='', flush=True)
    print()
    return nvd


# properties are generally suitable for the input layer
# while scores and severity rankings are suitable for the output layer
def get_json_value(key, cve_item):
    if key == 'base_metric_v2_properties':
        return [cve_item['impact']['baseMetricV2'][i] for i in ['obtainAllPrivilege', 'obtainUserPrivilege', 'obtainOtherPrivilege', 'userInteractionRequired']]
    elif key == 'cvssv2_properties':
        return [cve_item['impact']['baseMetricV2']['cvssV2'][i] for i in ['accessComplexity', 'authentication', 'confidentialityImpact', 'integrityImpact', 'availabilityImpact']]
    elif key == 'cvssv2_base_score':
        return cve_item['impact']['baseMetricV2']['cvssV2']['baseScore']
    elif key == 'bmv2_severity':
        return cve_item['impact']['baseMetricV2']['severity']
    elif key == 'bmv2_exploitability_score':
        return cve_item['impact']['baseMetricV2']['exploitabilityScore']
    elif key == 'bmv2_impact_score':
        return cve_item['impact']['baseMetricV2']['impactScore']
    elif key == 'cvssv3_properties':
        return [cve_item['impact']['baseMetricV2']['cvssV3'][i] for i in ['attackVector', 'attackComplexity', 'privilegesRequired', 'userInteractionRequired', 'scope', 'confidentialityImpact', 'integrityImpact', 'availabilityImpact']]
    elif key == 'cvssv3_base_score':
        return cve_item['impact']['baseMetricV3']['cvssV3']['baseScore']
    elif key == 'cvssv3_base_severity':
        return cve_item['impact']['baseMetricV3']['cvssV3']['baseSeverity']
    elif key == 'bmv3_exploitability_score':
        return cve_item['impact']['baseMetricV3']['exploitabilityScore']
    elif key == 'bmv3_impact_score':
        return cve_item['impact']['baseMetricV3']['impactScore']
    elif key == 'description':
        return cve_item['cve']['description']['description_data'][0]['value']


# cve_item=nvd['nvdcve-1.1-2018.json']['CVE_Items'][0]

# extract a vector containing different combinations of data using this function
def extract(nvd, keys_get):
    passed = 0
    failed = 0
    items_vector=[]
    for year in nvd.keys():
        for cve_item in nvd[year]['CVE_Items']:
            new_item=[]
            for key in keys_get:
                cve_val=[]
                try:
                    value=get_json_value(key, cve_item)
                    if type(value) == list:
                        value=' '.join([str(i) for i in value])
                        # if type(value[0]) == str: # and not value[0].isdigit() add int support
                        #     value=[' '.join(value)]
                        # else:
                        #     value=[' '.join([str(i) for i in value])]
                    elif type(value) == int:
                        value=str(value)
                    # convert bools to binary ints
                    else:
                        value=str(value)
                    # print(value)
                        # todo find a way to handle INTS, make a seperate model for int input
                        
                    # print(F"original {new_item} append {value} key {key} keys {keys_get}")
                    new_item.append(value)
                    passed+=1
                except:
                    failed+=1
                break
            if len(new_item) == len(keys_get):
                new_item=[new_item[0]]+[' '.join(new_item[1:])]
                items_vector.append(new_item) # due to unknown reason we receive empty values sometimes
    print(F"passed: {passed}\nfailed: {failed}")
    return items_vector



# balance data according to output labels
def balance(labelmap, items_vector):
    """
    Balances items so that each label has the same number of items
    Transforms all labels to their respective label-indices
    
    items_vector is a list of tuples where the first element of the tuple represents the label

    Returns: A list containing all items as tuples (string, label)
    """
    dataset_orig=[]
    for lab in labelmap.keys():
        dataset_orig+=[[(i[0].lower(),labelmap[lab]) for i in items_vector if i[1] == lab]]

    # calculate item cap
    lengths=[len(i) for i in dataset_orig]
    item_limit=min(lengths)
    print(F"items available per label {lengths}")
    print("item limit:", item_limit)

    # randomly choose item_limit items from each label
    dataset_blncd=[]
    for label_group in dataset_orig:
        dataset_blncd+=[random.choice(label_group) for i in range(item_limit)]
    print(F"{len(dataset_blncd)} items added")
    random.shuffle(dataset_blncd)
    return dataset_blncd


def make_ds(dataset_blncd, batch_size, shuffle_size):
    string_ds=[i[0] for i in dataset_blncd]
    label_ds=[i[1] for i in dataset_blncd]

    label_ds=tf.data.Dataset.from_tensor_slices(label_ds)
    string_ds=tf.data.Dataset.from_tensor_slices(string_ds)
    dataset=tf.data.Dataset.zip((string_ds, label_ds))

    dataset=dataset.batch(batch_size)

    dataset_len=len(dataset)
    train_size=int(0.7 * dataset_len)
    val_size=int(0.15 * dataset_len)
    test_size=int(0.15 * dataset_len)

    train_ds = dataset.take(train_size).shuffle(shuffle_size, reshuffle_each_iteration=True)
    test_ds = dataset.skip(train_size)
    val_ds = dataset.skip(test_size)
    test_ds = dataset.take(test_size)

    print(F"dataset sizes (train/val/test) {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    return train_ds,val_ds,test_ds


def preprocess(train_ds, val_ds, test_ds, batch_size, ds_len, max_features):
    # TextVectorization layer converts text to lowercase and strips punctuation by default

    sequence_length=int(sum(ds_len)/len(ds_len)) # setting sequence length to average length

    vectorize_layer = TextVectorization(
        # standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int', # each token has a corresponding number
        output_sequence_length=sequence_length)

    # adapt: map strings (words) to integers
    # Make a text-only dataset (without labels), then call adapt
    train_text = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label
        
    """
    # retrieve a batch (32 reviews and labels) from the dataset
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

    return train_ds,val_ds,test_ds


# functions for preprocess
