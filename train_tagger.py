#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:27:44 2019

@author: ignatius
"""
import model_setup as ms
import data_util as dt
import os, pickle, argparse
argp = argparse.ArgumentParser(
       prog='train_tagger.py',
       description="""This script trains and persists the multi-tagger for 
                      'parts-of-speech' (POS) and semantic (SEM) tagging tasks
                       with embedding models as features. The training parameters
                       and their default values are as described below:""",
       epilog="""Functions used here are defined in the 2 key modules:
                - `data_util` which loads the raw data and embedding files as 
                    well as prepares the training instances
                - `model_setup` which sets the parameters to train the models
                    and plot the results.""")

data_basefolder = os.path.join('.','data')
datafile = os.path.join(data_basefolder, 'cy_both_tagged.data')
vecsfile = os.path.join(data_basefolder, 'welsh_fasttext_filtered_300.vec')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

argp.add_argument('-df', '--datafile', default=datafile,
                  help = "raw tagged text file. Default:'cy_both_tagged.data' in folder.")
argp.add_argument('-vf', '--vecsfile', default=vecsfile,
                  help = "embeddings file. Default:'welsh_fasttext_filtered_300.vec' in folder.")
argp.add_argument('-v', '--nvecs', default=5, type=int, 
                  help = 'number of vectors from embedding. Default: 100.')
argp.add_argument('-e', '--eval_split', default=0.1, type=float,
                  help = 'evaluation data percentage. Default: 0.1.')
argp.add_argument('-n', '--n_epochs', default=10, type=int,
                  help = 'number of training epochs. Default: 100.')
argp.add_argument('-m', '--mini_batch_size', default=8, type=int,
                  help = 'size of the mini_batch. Default: 8.')
argp.add_argument('-d', '--dropout', default=0.3, type=float,
                  help = 'dropout rate in percentage. Default: 0.3.')
argp.add_argument('-b', '--batchnorm', default=False, type=str2bool,
                  help = 'dropout rate in percentage. Default: False')
argp.add_argument('-rp', '--result_point', default=1, type=int,
                  help = 'numbers of steps before each training result display. Default: 1')
argp.add_argument('-ep', '--eval_point', default=10, type=int,
                  help = 'numbers of steps before each evaluation result display. Default: 10')
args = argp.parse_args()

#params = f'{nvecs}_{mini_batch}_{dropout}'
print(f"{'-'*40}\ndata file: {args.datafile}\nvector file: {args.vecsfile}")
print(f"""Training configuration:
    \t-'nvecs'={args.nvecs}; 'mini_batch_size'={args.mini_batch_size}, 
    \t-'dropout'={args.dropout},'batchnorm'={args.batchnorm},\n{'-'*40}""")

#load and process the training data
dataloader = dt.load_vec_file(args.datafile, args.vecsfile)
#print(dataloader.show_stats()) #Remove later

#Prepare the 'train_set' and 'test_set'
train_set, eval_set, tagset =\
                     dt.prepare_training_data(dataloader, nvecs=args.nvecs, 
                                    mini_batch_size = args.mini_batch_size,
                                    eval_split=args.eval_split)

#Configure a simple fully connected feed-forward neural-network 
## 1 hidden layer does well enough
in_shape, output_length = train_set.output_shapes[0][1:], len(tagset)

config = {'datafile':args.datafile,'vecsfile':args.vecsfile, 'nvecs':args.nvecs,
          'eval_split':args.eval_split, 'n_epochs':args.n_epochs,
          'mini_batch_size':args.mini_batch_size, 'dropout':args.dropout, 
          'batchnorm':args.batchnorm,'result_point':args.result_point,
          'eval_point':args.eval_point, 'in_shape': in_shape,
          'output_length': output_length}

tagger = ms.configure_tagger(config['in_shape'],
                             config['output_length'],
                             config['dropout'],
                             config['batchnorm'])

with open('tagger.config', 'wb') as config_dump:
    pickle.dump(config, config_dump)

test_acc, test_loss = ms.test_tagger(tagger, eval_set)
print(f"Testing before training:\n\t-acc  = {100*test_acc:.2f}%\n\t-loss = {test_loss:.3f}")

# results: [epoch_accuracies, epoch_losses, eval_accuracies, eval_losses]
results = ms.train_tagger(train_set, eval_set, tagger,
                              num_epochs=args.n_epochs,
                              result_point = args.result_point, 
                              eval_point = args.eval_point)

#Old tracking script. May be removed later.
result_dump = 'result_dump.pkl'
with open(result_dump, 'wb') as  save_result:
    pickle.dump(results, save_result)

print(f"Training details successfully dumped in '{result_dump}'!")
print(f"Model training checkpoints stored in the 'checkpoint' folder")

test_acc, test_loss = ms.test_tagger(tagger, eval_set)
print(f"Testing after training:\n\t-acc  = {100*test_acc:.2f}%\n\t-loss = {test_loss:.3f}")
