import os
import glob
import tensorflow as tf
import keras
import numpy as np
import random

from options import Options


class Loader(Options):
    def __init__(self):
        super(Loader, self).__init__()

        A_path = f'{self.root}/monet_jpg'
        B_path = f'{self.root}/photo_jpg'

        A_list = glob.glob(f'{A_path}/*.jpg')
        B_list = glob.glob(f'{B_path}/*.jpg')

        # print(A_list)
        random.shuffle(A_list)
        random.shuffle(B_list)

        B_list = B_list[:300]
        # print(A_path)

        self.A = tf.data.Dataset.from_tensor_slices(A_list)
        self.B = tf.data.Dataset.from_tensor_slices(B_list)

    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, 3)
        img = tf.image.resize(img, [224, 224]) / 255.

        return img

    def load(self):
        A_ds = self.A.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        B_ds = self.B.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = tf.data.Dataset.zip((A_ds, B_ds))

        return ds


def configure_for_performance(ds, cnt, batchsz, shuffle=False):
    if shuffle==True:
        ds = ds.shuffle(buffer_size=cnt)
        ds = ds.batch(batchsz)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    elif shuffle==False:
        ds = ds.batch(batchsz)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds