#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 18:53:50 2019

@author: ignatius
"""
import model_setup as ms
import data_util as dt
import os, pickle

# Folders and files
data_basefolder = os.path.join('.','data') # .zip or .vec
filtered_vecs_fname = os.path.join(data_basefolder, 'welsh_fasttext_filtered_300.vec')
full_vecs_fname = os.path.join(data_basefolder, 'welsh_fasttext_300.vec')
#data_file = args.datafile or os.path.join(data_basefolder, 'cy_both_tagged.data')
data_file = os.path.join(data_basefolder, 'cy_both_tagged.data')

#n_epochs, respnt, evpnt = 5, 1, 2  #Code testing
#n_epochs, respnt, evpnt = 50, 5, 10 #epochs for parameter optimisation
respnt, evpnt = 5, 20 #experiment

#load and process the training data
dataloader = dt.load_vec_file(data_file, filtered_vecs_fname) #done

#parameters to optimise
#runs = [('nodropout_nobatchnorm_result.pkl', [100],[8], [0],  False),
#        ('wtdropout_nobatchnorm_result.pkl', [100],[8], [30], False),
#        ('nodropout_wtbatchnorm_result.pkl', [100],[8], [0],  True),
#        ('wtdropout_wtbatchnorm_result.pkl', [100],[8], [30], True)]

runs = [('wtdropout_nobatchnorm_200epochs.pkl', [100],[8], [30], False, 200),
        ('nodropout_wtbatchnorm_200epochs.pkl', [100],[8], [0], True, 200)]

for results_dump, nvecs_list, mini_batches, dropouts, batchnorm, n_epochs in runs:    
    all_results = {} # dictionary to store the results
    for nvecs in nvecs_list:
        for mini_batch in mini_batches:
            for dropout in dropouts:
                params = f'{nvecs}_{mini_batch}_{dropout}'
                print(f"\nConfiguration: 'nvecs'={nvecs}, 'mini_batch'={mini_batch}, 'dropout'={dropout}, 'batchnorm'={batchnorm}, \n{'-'*40}")
    
                #Prepare the 'train_set' and 'test_set'
                train_set, eval_set, tagset =\
                                dt.prepare_training_data(dataloader, nvecs=nvecs,
                                                         mini_batch_size=mini_batch, eval_split=0.1)

                #Configure a simple fully connected feed-forward neural-network 
                ## Only 1 hidden layer does well enough
                in_shape, output_length =\
                                train_set.output_shapes[0][1:], len(tagset)
                
                tagger = ms.configure_tagger(in_shape, output_length, drop_out=dropout/100, batchnorm=batchnorm)
                
                # results: [epoch_accuracies, epoch_losses, eval_accuracies, eval_losses]
                _ , results = ms.train_tagger(train_set, eval_set,
                                                  model=tagger, num_epochs=n_epochs,
                                                  result_point = respnt, eval_point = evpnt)
    
                # store only the results (drop d tagger) in the result dictionary
                print(f'Experiments successfully run!\nUpdating results with params = {params}...', end="")
                all_results[params] = results
                with open(results_dump, 'wb') as  output:
                    pickle.dump(all_results, output)
                print('Successfully updated!')
    
    print('Experiments successfully run!\nDumping final results in {results_dump}...', end="")
    with open(results_dump, 'wb') as  output:
        pickle.dump(all_results, output)
    print('Successfully updated!\nThanks for your patience.')
