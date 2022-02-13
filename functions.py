import os
import tensorflow as tf
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

from options import Options


class func(Options):
    def __init__(self):
        super(func, self).__init__()

    def mkdir(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

        if not os.path.exists(self.OUTPUT_DIR_CKP):
            os.makedirs(self.OUTPUT_DIR_CKP)

        if not os.path.exists(self.OUTPUT_DIR_SAMPLE):
            os.makedirs(self.OUTPUT_DIR_SAMPLE)

        if not os.path.exists(self.OUTPUT_DIR_LOSS):
            os.makedirs(self.OUTPUT_DIR_LOSS)

        if not os.path.exists(self.OUTPUT_DIR_TEST):
            os.makedirs(self.OUTPUT_DIR_TEST)

    def save_sampling(self, img, ep, step, tag):
        # output = self.generator(img)

        # output = output * 255.0
        output = tf.clip_by_value(img * 255.0, 0, 255).numpy()
        # img_Name = re.split(r'[\\]', names)[-1]
        # img_Name = re.compile('.png').sub('', img_Name)
        for i in range(len(output)):
            cv2.imwrite(f'{self.OUTPUT_DIR_SAMPLE}/{ep}_{step}_{tag}.png', cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))

    def save_loss(self, ep, disc_loss, gen_loss):
        print(f'||Epoch {ep} / {self.epochs}|| '
              f'\n || GenLoss {np.mean(gen_loss)}|| '
              f'\n || Disc Loss {np.mean(disc_loss)}')

        fig, ax = plt.subplots()
        plt.plot(disc_loss, label='Discriminator', alpha=0.6)
        plt.plot(gen_loss, label='Generator', alpha=0.6)
        plt.title('Losses')
        plt.legend()
        plt.savefig(f'{self.OUTPUT_DIR_LOSS}/losses_{str(ep)}.png')
