import os, sys
import numpy as np
import random
import shutil
random.seed(0)

input_dir = 'data/raw/novels/files/'
dir_out = 'data/raw/novels'
dest_out = ['train', 'val', 'test']

filenames = [filename for filename in os.listdir(input_dir) if '.txt' in filename]

print("Total number of files: ", len(filenames))

splits = {'train': 0.5, 'val': 0.2, 'test': 0.3}
rnd = random.Random(0)

for filename in filenames:
    dest = None
    p = rnd.random()
    if p < splits['train']:
        dest = 'train'
    elif p < splits['train'] + splits['val']:
        dest = 'val'
    else:
        dest = 'test'
    dir_out_ = os.path.join(dir_out, dest)

    srcFile = str(os.path.join(input_dir, filename))
    shutil.copy2(srcFile, str(dir_out_))

    # Use this to move instead of copying files
    # shutil.move(srcFile, str(os.path.join(dir_out_, filename)))
