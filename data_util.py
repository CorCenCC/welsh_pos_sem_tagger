#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:13:54 2019

@author: ignatius
"""

#@title Instantiating 'DataLoader' object and functions (RUN ME!) { display-mode: "form" }
import random
from collections import Counter
import zipfile, pickle, gensim, os, re
import numpy as np
import tensorflow as tf

class DataLoader():
    def __init__(self, data_file, emb_model, n_past_words=3):
        self.emb_model = emb_model
        self.n_past_words = n_past_words
        self.word_to_id, self.id_to_word = {}, {}
        self.tag_to_id, self.id_to_pos = {}, {}

        with open(data_file, 'r', encoding='utf8') as f:
            self.sentences = [line.strip().split() for line in f.readlines()]
        self.gen_vocab(self.sentences)

    def split_tags(self, sent):
        ws,ts = [],[]
        for wt in sent:
            w,t = wt.split('|',1)
            ws.append(w)
            ts.append(re.match(r'([^|]+\|!?[A-Z]+\d*)',t).group())
        return (ws,ts)
    
    def gen_vocab(self, sentences):
        words, tagset = [], set()
        for i, sent in enumerate(sentences):
            sent_words, sent_tags = self.split_tags(sent)
            words.extend(sent_words)
            tagset.update(sent_tags)
        word_counts     = Counter(words)
        words_to_keep   = [t[0] for t in word_counts.most_common()]
        self.word_to_id = {word: i for i, word in enumerate(words_to_keep)}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.tag_to_id  = {tag: i for i, tag in enumerate(tagset)}
        self.id_to_tag  = {v: k for k, v in self.tag_to_id.items()}
        self.words = words
        return word_counts, tagset

    def get_filtered_embeddings(self):
        intersection = self.emb_model.vocab & self.word_to_id.keys()
        return {w:self.emb_model[w] for w in intersection}
    
    def save_embeddings(self, emb_filename, embs):
        with open(emb_filename, 'w', encoding="utf8") as f:
            f.write(f"{len(embs)} {len(tuple(embs.values())[0])}\n")
            for w,vecs in embs.items():
                f.write(f"{w} {' '.join([str(v) for v in vecs])}\n")
    
    def save_vocab(self, vocab_filename):
        dicts = [self.word_to_id, self.tag_to_id, self.id_to_word,  self.id_to_tag]
        with open(vocab_filename, 'wb') as f:
            pickle.dump(dicts, f)
            
    def create_instances(self, sentences, n_prev_wds=3):
        self.instance_list = []
        for sent in sentences:
            tokens, tags = self.split_tags(sent)
            for i in range(len(tokens)):
                if tokens[i]!='unk' and tags[i]!='PUNCT':
                    new_instance = []
                    for j in range(i-n_prev_wds, i):
                        if j<0:
                            new_instance.append('-*-')
                        else:
                            new_instance.append(tokens[j])
                    new_instance.extend([tokens[i], tags[i]])
                    self.instance_list.append(new_instance)
        return self.instance_list

    def feature_to_tensor(self, feature, embs, size):
        return [(embs[word][:size] if word in embs else np.zeros(size)) for word in feature]
    
    def gen_features_and_labels(self, instances, size):
        self.features, self.labels = [],[]
        for instance in instances:
            features, tag = instance[:-1], instance[-1]
            self.features.append(self.feature_to_tensor(features, self.emb_model, size))
            self.labels.append(self.tag_to_id[tag])
        return self.features, self.labels

    def show_stats(self):
        print(f"{'sentences':>13s}:  {len(self.sentences)}")
        print(f"{'token size':>13s}: {len(self.words)}")
        print(f"{'vocab len:':>13s}: {len(set(self.words))}")
        print(f"{'model vocab':>13s}: {len(self.emb_model.vocab)}")
        print(f"{'model vecsize':>13s}: {self.emb_model.vector_size}")
        print(f"{'model oov':>13s}: {len(self.words) - len(self.emb_model.vocab)}")
        print(f"{'tagset size':>13s}: {len(self.tag_to_id)}")
        print(f"{'punctuations':>13s}: {1667}")  # FIXME!
        print(f"{'unknown tags':>13s}: {44}")   # FIXME!

# Use the 'filtered' or 'main' or 'zipped' vec_file
# For the first use, you may have to use the zipped version
data_basefolder = os.getcwd()+'/data/' # .zip or .vec #FIXME
def load_vec_file(data_file, vec_file):
    if os.path.exists(vec_file) and 'filtered' in vec_file:
        print('Loading filtered embedding models...', end="")
    else:
        if os.path.exists(vec_file) and vec_file.endswith('.vec'):
            print('Loading full embedding models...(may take a while...)', end="")
        elif os.path.exists(vec_file) and vec_file.endswith('.zip'):
            print('Extracting embedding zipped file (may take a few seconds)...', end="")
            with zipfile.ZipFile(vec_file,'r') as zip_ref:
                zip_ref.extractall(data_basefolder)
        else:
            print("FileError: Embedding file not found!")
            return None

    emb_model = gensim.models.KeyedVectors.load_word2vec_format(vec_file)

    print(' Done!\nLoading and processing the training data...', end="")
    dataloader = DataLoader(data_file=data_file, emb_model=emb_model)
#    print(' Done!\nCreating and saving the filtered embedding...!', end="")
#    filtered_embeddings = dataloader.get_filtered_embeddings()
#    dataloader.save_embeddings(os.path.join(data_basefolder,'welsh_fasttext_filtered_300.vecs'), filtered_embeddings)
    print(' Done!\nDataLoader() objected returned!')
    dataloader.save_vocab('vocab.pkl')
    return dataloader

def prepare_training_data(dataloader, nvecs, mini_batch_size, eval_split):
    word_count, tagset = dataloader.gen_vocab(dataloader.sentences)
    train_len = len(dataloader.sentences) - int(len(dataloader.sentences)*eval_split)
    print('Preparing training data... ', end="")
    random.seed(5) #FIXME: put parameter in the argument list
    random.shuffle(dataloader.sentences)
    trainset = dataloader.create_instances(dataloader.sentences[:train_len])
    eval_set = dataloader.create_instances(dataloader.sentences[train_len:])
#    testset  = dataloader.create_instances(dataloader.sentences[550:]) # FIXME!!!

    trainset_vectors = dataloader.gen_features_and_labels(trainset, nvecs)
    eval_set_vectors = dataloader.gen_features_and_labels(eval_set, nvecs)
#    testset_vectors  = dataloader.gen_features_and_labels(testset, nvecs)  # FIXME!!!

    #Prepare the training dataset...
    train_ds = tf.data.Dataset.from_tensor_slices((trainset_vectors[0],trainset_vectors[1]))
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
    train_ds = train_ds.shuffle(buffer_size=mini_batch_size * 10)
    train_ds = train_ds.batch(mini_batch_size)

    print('Done!\nPreparing evaluation data ...', end="")
    eval_insts = tf.convert_to_tensor(eval_set_vectors[0], dtype=tf.float32)
    eval_labels = tf.convert_to_tensor(eval_set_vectors[1], dtype=tf.int32)
    eval_ds = (eval_insts, eval_labels)
    print('Done!\nTraining and evaluation data returned.', end="")
    return train_ds, eval_ds, tagset