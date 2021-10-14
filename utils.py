import os
import numpy as np
import json

def json_from_dict(dict, path):
    with open(os.path.join(path,'cfg.json'), 'w') as fp:
        json.dump(dict, fp)

def create_dir(dir, subdirs=None):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if subdirs:
        for s in subdirs:
            if not os.path.exists(os.path.join(dir, s)):
                os.makedirs(os.path.join(dir, s))
    
    return dir

def make_unique_filename(files, dir=None):
    '''
    Given a list of files, generate a new - unique - filename
    '''
    while True:
        f = str(np.random.randint(low=100, high=100000))
        if not f in files: break
    # Optionally create dir
    if dir:
        create_dir(os.path.join(dir,f))
    return f, os.path.join(dir,f)