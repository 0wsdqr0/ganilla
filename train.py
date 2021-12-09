import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from skimage.metrics import structural_similarity
import argparse
from preprocess import get_data
from model import *

checkpoint="checkpoints/"
epoch_choose="epoch_125"
logs="logs/"

def train(model,photo,illus,checkpoint,logs,init_ep):
    print("Training model...")
    # print(photo,il)
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs,
            update_freq='batch',
            profile_batch=0),
        # checkpoint_callback
        Checkpoints(checkpoint, 5)
    ]
    # print(callback_list)
    model.fit(
        tf.data.Dataset.zip((photo, illus)),
        epochs=100,
        initial_epoch=init_ep,
        callbacks=callback_list
    )

def test(generator,photo,epoch):

    count = 0
    structure_similarity = 0

    for img in photo.take(10):
        count += 1

        prediction = generator(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

        original = (img[0] ).numpy().astype(np.uint8)  # converts img to [0, 255]

        # structure_similarity += structural_similarity(original, prediction, channel_axis=True)

        file_path = "generated"  + os.sep + "epoch_" + str(epoch)


        if not os.path.exists(file_path):
            os.makedirs(file_path)

        prediction = tf.keras.preprocessing.image.array_to_img(prediction)
        prediction.save(file_path + os.sep + "{i}_generated.png".format(i=count))

        original = tf.keras.preprocessing.image.array_to_img(original)
        original.save(file_path + os.sep + "{i}_original.png".format(i=count))
    # print("SSIM", structure_similarity / count)



def main():
    ##parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path',default=None,help='path to the checkpoint dir')
    parser.add_argument('--test',action='store_true')

    args=parser.parse_args()
    # print(args.checkpoint_path)



    illustration_train,illustration_test=get_data(os.getcwd()+"/monet2photo/train/illustration",os.getcwd()+"/monet2photo/test/illustration")
    photo_train,photo_test = get_data(os.getcwd() + "/monet2photo/train/photo", os.getcwd() + "/monet2photo/test/photo")


    # print(illustration_train,photo_train)
    gen_G = Generator()
    gen_F = Generator()
    dis_F = Discriminator()
    dis_G = Discriminator()

    gen_G(tf.keras.Input(shape=(256, 256, 3)))
    gen_F(tf.keras.Input(shape=(256, 256, 3)))
    dis_F(tf.keras.Input(shape=(256, 256, 3)))
    dis_G(tf.keras.Input(shape=(256, 256, 3)))

    gen_G.summary()
    dis_F.summary()

    init_ep = 0
    model = GANILLA(g_G=gen_G, g_F=gen_F, d_G=dis_G, d_F=dis_F)
    model.compile()
    if(os.listdir(checkpoint)):
        cp_g_G = os.getcwd() + "/" + checkpoint + epoch_choose + "/g_Gweights.h5"
        cp_g_F = os.getcwd() + "/" + checkpoint + epoch_choose + "/g_Fweights.h5"
        cp_d_G = os.getcwd() + "/" + checkpoint + epoch_choose + "/d_Gweights.h5"
        cp_d_F = os.getcwd() + "/" + checkpoint + epoch_choose + "/d_Fweights.h5"

        gen_G.load_weights(cp_g_G, by_name=False)
        gen_F.load_weights(cp_g_F, by_name=False)
        dis_F.load_weights(cp_d_F, by_name=False)
        dis_G.load_weights(cp_d_G, by_name=False)

        init_ep=int(epoch_choose[-3:])


    if(args.test):
        test(gen_G,photo_test,init_ep)
        # print(1)
    else:
        train(model,photo_train,illustration_train,checkpoint,logs,init_ep)


if __name__=="__main__":
    main()