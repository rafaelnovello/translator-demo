# -*- coding: utf-8 -*-

import pickle


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    with open('checkpoint/preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


def load_params():
    """
    Load parameters from file
    """
    with open('checkpoint/params.p', mode='rb') as in_file:
        return pickle.load(in_file)
