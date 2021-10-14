import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    dst_config =   {"partition": "train", # train, val, eval

                    # Directories
                    "source_root": "./_dummy/dest/", # Where the output of the standardization process resides
                    "output_dir": "./_dummy/dest/DNR", # Where the new dataset will be compiled

                    # Processing parameters
                    "lufs_music": -24,
                    "lufs_speech": -17,
                    "lufs_sfx": -21,
                    "lufs_background": -29,
                    "lufs_master": -27,
                    "lufs_noise": None, # set to none for no noise

                    "lufs_range_class": 2,
                    "lufs_range_clip": 1,

                    "peak_norm_db": -0.5,

                    "sr": 44100,
                    "channel": 1, 
                    "seq_dur": 60.0,

                    # Dataset spec
                    "num_files": 10000, # dummy var. to keep the progress bar going

                    # Seed
                    "seed": 42,

                    "alt_dir": None, # Experimental: set this if you want to build dnr from metadata as opposed to from scratch
                    }

@config_ingredient.named_config
def eval():
    print("Computing dataset for eval-set")
    dst_config = {
        "partition": "eval",
        "num_files": 10000 # Allows to iterate until dataset exhaustion
    }

@config_ingredient.named_config
def train():
    print("Computing dataset for train-set")
    dst_config = {
        "partition": "train",
        "num_files": 3295
    }

@config_ingredient.named_config
def val():
    print("Computing dataset for val-set")
    dst_config = {
        "partition": "val",
        "num_files": 440
    }