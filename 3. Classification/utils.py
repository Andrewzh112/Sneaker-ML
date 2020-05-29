from fuzzywuzzy import fuzz
import re

from PIL import Image, ImageOps
import cv2

import pandas as pd
import numpy as np


def unique_shoelists(shoenames):
    shoe2shoes = []
    for i, shoe in enumerate(shoenames):
        new_shoes = [shoe]
        for j, other in enumerate(shoenames):
            if i == j:
                continue
            if fuzz.token_sort_ratio(shoe,other)>=95:
                num_shoe=re.findall(r'\d+',shoe)
                num_other=re.findall(r'\d+',other)
                if num_shoe!=[] and num_other!=[] and num_shoe[0]!=num_other[0]:
                    continue
                new_shoes.append(other)
        new_shoes.sort()
        shoe2shoes.append(new_shoes)
    shoes_fin = dedupe(shoe2shoes)
    return shoes_fin


def dedupe(shoe2shoes):
    seen = set()
    shoes_fin = []
    for shoelist in shoe2shoes:
        if tuple(shoelist) in seen:
            continue
        shoes_fin.append(shoelist)
        seen.add(tuple(shoelist))
    return shoes_fin


def train_test_names(list_names, test_size=0.1, seed=0):
    np.random.seed(seed)
    train,test = [],[]

    n_tests = int(len(list_names)*test_size)
    packed_list = unique_shoelists(list_names)
    
    for shoe_list in packed_list:
        if len(shoe_list) == 1:
            continue
        test_or_train = np.random.uniform(0,1)
        if test_or_train >= test_size:
            train.extend(shoe_list)
        else:
            test.extend(shoe_list)
    
    hyped_left = [shoe_list[0] for shoe_list in packed_list if len(shoe_list)==1]
    np.random.shuffle(hyped_left)
    test.extend(hyped_left[:n_tests-len(test)])
    train.extend(hyped_left[n_tests-len(test):])
    np.random.shuffle(test)
    np.random.shuffle(train)
    
    return train, test

