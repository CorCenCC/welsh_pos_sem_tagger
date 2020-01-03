#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:22:19 2019

@author: ignatius
"""
import model_setup as ms
import data_util as dt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pickle

tf.enable_eager_execution()

accuracy = tfe.metrics.Accuracy()

config = pickle.load(open('tagger.config', 'rb'))
vocab_dicts = pickle.load(open('vocab.pkl', 'rb'))

tagger = ms.configure_tagger(config['in_shape'],
                             config['output_length'],
                             config['dropout'],
                             config['batchnorm'])

checkpoint_dir = tf.train.latest_checkpoint('checkpoints/')

optimizer=tf.train.AdamOptimizer()

root = tf.train.Checkpoint(optimizer=optimizer, model=tagger,
                         optimizer_step=tf.train.get_or_create_global_step()) 
root = tfe.Saver(tagger.variables)

root.restore(checkpoint_dir)

dataloader = dt.load_vec_file(config['datafile'],
                              config['vecsfile'])

_, eval_set, _ = dt.prepare_training_data(dataloader,
                                nvecs=config['nvecs'],
                                mini_batch_size=config['mini_batch_size'],
                                eval_split=config['eval_split'])
eval_insts, eval_labels = eval_set

logits = tagger(eval_insts, training=False)

accuracy(tf.argmax(logits, axis=1, output_type=tf.int32), eval_labels)
# Compute the cross-entropy loss of the classification outputs on this batch
loss_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                            labels=eval_labels)
# Compute the average loss over the batch
eval_loss = tf.reduce_mean(loss_value)  
        # Compare most likely predicted label to actual label

print(f'\naccuracy: {100*accuracy.result():.3f}%\neval_loss: {eval_loss}')
print(f'eval_labels: {eval_labels}')
print(f'logits: {tf.argmax(logits, axis=1, output_type=tf.int32)}')