import tensorflow.keras as keras

import GANs.losses as losses
from options import Options
from GANs.CycleGAN.CycleGAN import Cycle

# 각 GAN 모델의 dis와 gen을 받아서 load해준다.
class GAN(Options):
    def __init__(self):
        super(GAN, self).__init__()
        Module = Cycle()

        if self.GAN_type == 'CycleGAN':
            self.gen_G = Module.gen(name='gen_G')
            self.gen_F = Module.gen(name='gen_F')

            self.disc_X = Module.disc(name='disc_X')
            self.disc_Y = Module.disc(name='disc_Y')

            self.gen_G_optim = keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)
            self.gen_F_optim = keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)
            self.disc_X_optim = keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)
            self.disc_Y_optim = keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)
            self.gen_loss_fn = losses.Cycle_gen_loss_fn
            self.disc_loss_fn = losses.Cycle_disc_loss_fn