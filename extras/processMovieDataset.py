import os, sys
import numpy as np
import random
random.seed(0)

input_file = 'data/raw/summaries/MovieSummaries/plot_summaries.txt'
dir_out = 'data/raw/summaries'
dest_out = ['train', 'val', 'test']

summaries_dict = {}
with open(input_file) as f:
    for line in f:
        line = line.rstrip()
        summary_id = line.split('\t')[0]
        text = line.split('\t')[1]
        summaries_dict[summary_id] = text

splits = {'train': 0.5, 'val': 0.2, 'test': 0.3}

keys  = list(summaries_dict.keys())

rnd = random.Random(0)
for key in keys:
    dest = None
    p = rnd.random()
    if p < splits['train']:
        dest = 'train'
    elif p < splits['train'] + splits['val']:
        dest = 'val'
    else:
        dest = 'test'
    dir_out_ = os.path.join(dir_out, dest)
    with open(os.path.join(dir_out_, str(key) + '.txt'), 'w') as f:
        f.write(summaries_dict[key])
