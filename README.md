# Welsh Part-of-Speech and Semantic Tagger 

This is a simple embedding based multi-task tagger for the Welsh part-of-speech and semantic tagging implemented using a feedforward network in Tensorflow.

### Data
The gold standard data, `cy_both_tagged.data`, comprises 611 manually tagged sentences (14,876 tokens), i.e. with both the *part-of-speech* and *semantic* tags,  extracted from a variety of existing Welsh corpora, including:
* `Kynulliad314` (Welsh Assembly proceedings),
* `Meddalwedd15` (translations of software instructions),
* `Kwici16`(Welsh Wikipedia articles),
* `LERBIML17` (multi-domain spoken corpora) and
* some short abstracts of some Welsh Wikipedia articles.

### Embedding model
A key contribution of this work to Welsh NLP research is the application of pre-trained embeddings to build the model. 

To that effect, we used the Welsh pre-trained embedding models built by the [FastText Project](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cy.300.vec.gz) (Grave et al 2018). 

### Usage example: Training tagger
To train the model, execute the python file `train_tagger.py`:

```
$ python path/to/train_tagger.py
```
The expected screen outputs at the `training` stage will be similar to the one shown below:

```Running in Eager mode.
----------------------------------------
data file: path/to/cy_both_tagged.data
vector file: path/to/welsh_fasttext_filtered_300.vec
Training configuration:
        -'nvecs'=5; 'mini_batch_size'=8, 
        -'dropout'=0.3,'batchnorm'=False,
----------------------------------------
Loading filtered embedding models... Done!
Loading and processing the training data... Done!
DataLoader() objected returned!
Preparing training data... Done!
Preparing evaluation data ...Done!
Training and evaluation data returned.
Tagger configuration:
        - input shape=(4, 5)
        - dropout=0.3
        - batchnorm=False
Model successfully configured!
Training in progress...
Testing before training:
        -acc  = 0.07%
        -loss = 5.970
Epoch 01: Loss = 4.244, Accuracy = 15.378%
Epoch 02: Loss = 3.969, Accuracy = 20.181%
--[more results shown here...]
Epoch 10: Loss = 3.302, Accuracy = 33.760%
------------------------------------------
-Eval 02: Loss = 3.628, Accuracy = 33.907%
==========================================

Training details successfully dumped in 'result_dump.pkl'!
Model training checkpoints stored in the 'checkpoint' folder
Testing after training:
        -acc  = 35.29%
        -loss = 3.628
```


When you are satisfied with the training and evaluation results, then run `demo_tagger.py`, input a Welsh sentence to the model see it tagged with the [CyTag](https://github.com/IgnatiusEzeani/CyTag) POS tags and [CySemTag](http://eprints.lancs.ac.uk/123588/1/lrec2018_cysemtagger.pdf) semantic tags




```
$ python demo_tagger.py
Enter a sentence to be annotated:
A fydd rhywfaint o 'r arian hwn yn cael ei ddefnyddio
i sicrhau bod modd defnyddio tocynnau rhatach yn Lloegr
yn ogystal ag yng Nghymru?

Loading saved vocabulary...
Generating tensors...
Your sentence, annotated:
A|Rha|Z5 fydd|B|A3+ rhywfaint|E|N5/N5.1- o|Ar|Z5 'r|YFB|Z5 arian|E|I1 
hwn|Rha|A3+ yn|U|Z5 cael|B|A9 ei|Rha|Z8 ddefnyddio|B|A1.5.1 i|Ar|Z5
sicrhau|B|A7+ bod|B|A3+ modd|E|X4.2 defnyddio|B|A1.5.1 tocynnau|E|Q1.2 
rhatach|Ans|I1.3- yn|Ar|Z5 Lloegr|E|Z2 yn|Ar|Z5 ogystal|Ans|Z99 ag|Ar|Z5 
yng|Ar|Z5 Nghymru|E|Z2 ?|Atd|PUNCT
```
**Acknowledgement:** This code was originally adapted (but heavily modified) from [mrahtz's](https://github.com/mrahtz/tensorflow-pos-tagger) work on *Tensorflow POS tagger*
