#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:29:14 2019

@author: ignatius
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
#import matplotlib.pyplot as plt

try:
    tf.enable_eager_execution()
    print('Running in Eager mode.')
except ValueError:
    print('Already running Eagerly')

# Only 1 hidden layer is enough
def configure_tagger(input_shape, output_length, dropout=0.0, batchnorm=True):
    print(f'\nTagger configuration:\n\t- input shape={input_shape}\n\t- dropout={dropout}\n\t- batchnorm={batchnorm}')
    if batchnorm:
        tagger = tf.keras.Sequential([
            # Input layer: a flattened 4x100 input matrix of word vectors
            tf.keras.layers.Flatten(input_shape=input_shape, name='flatten_input'), 
            tf.keras.layers.BatchNormalization(axis=1),
            # Hidden layer: 1 layer of 256 neurons + ReLU non-linearity activation
            tf.keras.layers.Dense(256, activation=tf.nn.relu, name='input_to_H1'),        
            tf.keras.layers.BatchNormalization(axis=1),
            # Dropout layer:
            tf.keras.layers.Dropout(dropout),        
            # Output layer: with neuron size as length of the tagset
            tf.keras.layers.Dense(output_length, name='H1_to_logits'),  #name='H2_to_logits'
        ])
    else:
        tagger = tf.keras.Sequential([
        # Input layer: a flattened 4x100 input matrix of word vectors
        tf.keras.layers.Flatten(input_shape=input_shape, name='flatten_input'), 
        # Hidden layer: 1 layer of 256 neurons + ReLU non-linearity activation
        tf.keras.layers.Dense(256, activation=tf.nn.relu, name='input_to_H1'),        
        # Dropout layer:
        tf.keras.layers.Dropout(dropout),
        # Output layer: with neuron size as length of the tagset
        tf.keras.layers.Dense(output_length, name='H1_to_logits'),  #name='H2_to_logits'
    ])
    print('Model successfully configured!\nTraining in progress...')
    return tagger

#The optimizer is responsible for controlling the learning rate
# train_tagger:(tagger, optimizer, num_epochs, result_point, eval_point)
def train_tagger(train_ds, eval_ds, model, 
                 optimizer=tf.train.AdamOptimizer(), 
                 num_epochs=100, result_point = 5, eval_point = 20):
   
    step_counter = tf.train.get_or_create_global_step()  # Just a variable that keeps track of how many training steps we've run
    checkpoint_prefix = 'checkpoints/ckpt'

    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step()) 
    
    summary_writer = tf.contrib.summary.create_file_writer('log')
    summary_writer.set_as_default()
    
    # Extract the eval instances and labels  
    eval_insts, eval_labels = eval_ds
    
    # Lists to store the loss and accuracy of every epoch
    epoch_losses, epoch_accuracies = [], []

    # Lists to store the evaluation losses and accuracies
    eval_losses, eval_accuracies = [], []

    for epoch in range(1,num_epochs+1):
        # Tensorflow provides a convenient API for tracking a number of metrics during training/evaluation
        loss_avg = tfe.metrics.Mean()
        accuracy = tfe.metrics.Accuracy()
        # Loop over our data pipeline
        for step, (instance_batch, label_batch) in enumerate(train_ds):
        # Initialise a GradientTape to track the operations
            with tf.GradientTape() as tape:
                # Compute the logits (un-normalised scores) of the current batch of examples 
                
                # using the neural network architecture we defined earlier
                logits = model(instance_batch, training=True)
                
                # Compute the cross-entropy loss of the classification outputs on this batch
                loss_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                            labels=label_batch)
                
                # Compute the average loss over the batch
                loss_value = tf.reduce_mean(loss_value)  

                # Add current batch loss to our loss metric tracker - note the function call semantics
                loss_avg(loss_value)

                # Compare most likely predicted label to actual label
                accuracy(tf.argmax(logits, axis=1, output_type=tf.int32), label_batch)

            # Play the tape backwards and get the gradient of the loss of the current batch
            # Note we're now outside the scope of the with-block above
            grads = tape.gradient(loss_value, model.variables)
            # Use the optimizer to apply the gradients to the tagger parameters along with
            # its internal learning rate
            optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)
            
        # Get the average loss and accuracy for the epoch
        epoch_loss = loss_avg.result()
        epoch_losses.append(epoch_loss)
        epoch_accuracy = accuracy.result()
        epoch_accuracies.append(epoch_accuracy)
        
        if (epoch%result_point==0 and epoch) or epoch==num_epochs:
            print(f"Epoch {epoch:02d}: Loss = {epoch_loss:.3f}, Accuracy = {epoch_accuracy:.3%}")

        logits = model(eval_insts, training=False)
        # Compute the cross-entropy loss of the classification outputs on this batch
        loss_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=eval_labels)

        # Compute the average loss over the batch
        eval_loss = tf.reduce_mean(loss_value)  

        # Compare most likely predicted label to actual label
        accuracy(tf.argmax(logits, axis=1, output_type=tf.int32), eval_labels)
        eval_losses.append(eval_loss)
        eval_accuracy = accuracy.result()
        eval_accuracies.append(eval_accuracy)

#        step_counter.assign_add(1)
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            tf.contrib.summary.scalar('training_accuracy', epoch_accuracy)
            tf.contrib.summary.scalar('training_loss', epoch_loss)
#            tf.contrib.summary.merge(['training_accuracy','training_loss'])
            tf.contrib.summary.scalar('evaluation_accuracy', eval_accuracy)
            tf.contrib.summary.scalar('evaluation_loss', eval_loss)
#            tf.contrib.summary.merge(['evaluation_accuracy','evaluation_loss'])

        if (epoch%eval_point==0 and epoch) or epoch==num_epochs:
            print(f"{42*'-'}\n-Eval {1+(epoch//eval_point):02d}: Loss = {eval_loss:.3f}, Accuracy = {eval_accuracy:.3%}\n{42*'='}\n")
        root.save(checkpoint_prefix)
#        tfe.Saver(model.weights).save(checkpoint_path, global_step=step_counter)
    return  epoch_accuracies, epoch_losses, eval_accuracies, eval_losses

def test_tagger(model, test_ds):
    test_insts, test_labels = test_ds
    logits = model(test_insts, training=False)
    # Compute the cross-entropy loss of the classification outputs on this batch
    loss_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=test_labels)

    # Compute the average loss over the batch
    test_loss = tf.reduce_mean(loss_value)  

    # Compare most likely predicted label to actual label
    accuracy = tfe.metrics.Accuracy()
    accuracy(tf.argmax(logits, axis=1, output_type=tf.int32), test_labels)
    
    return accuracy.result(), test_loss