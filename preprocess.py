import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tqdm import tqdm

def get_data(train,test):
    test_dataset=tf.keras.preprocessing.image_dataset_from_directory(test, image_size=(256, 256), label_mode=None, shuffle=True,class_names=None, batch_size=1)
    train_dataset=tf.keras.preprocessing.image_dataset_from_directory(train, image_size=(256, 256), label_mode=None, shuffle=True,class_names=None, batch_size=1)

    return train_dataset,test_dataset