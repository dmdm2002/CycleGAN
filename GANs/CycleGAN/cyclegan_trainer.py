import tensorflow as tf
import tensorflow.keras as keras
import os
import time

from DataLoader import Loader, configure_for_performance
from functions import func
from GANs.Model import GAN


class Cyclegan_Trainer(GAN):
    def __init__(self):
        super(Cyclegan_Trainer, self).__init__()
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5
        self.func = func()
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as grads_G_tape, tf.GradientTape() as grads_F_tape, tf.GradientTape() as disc_X_grads_tape, tf.GradientTape() as disc_Y_grads_tape:
            # x -> y
            fake_y = self.gen_G(x, training=True)
            # y - > x
            fake_x = self.gen_F(y, training=True)
            # x -> y -> x
            cycle_x = self.gen_F(fake_y, training=True)
            # y -> x -> y
            cycle_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(x, training=True)
            same_y = self.gen_G(y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.gen_loss_fn(disc_fake_y)
            gen_F_loss = self.gen_loss_fn(disc_fake_x)

            # Generator Cycle loss
            cycle_loss_G = self.cycle_loss_fn(y, cycle_x) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(x, cycle_y) * self.lambda_cycle

            id_loss_G = (
                self.cycle_loss_fn(y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.cycle_loss_fn(x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.disc_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.disc_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = grads_G_tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = grads_F_tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = disc_X_grads_tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = disc_Y_grads_tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        self.gen_G_optim.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optim.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        self.disc_X_optim.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optim.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return total_loss_G, total_loss_F, disc_X_loss, disc_Y_loss, fake_y, cycle_x

    def training(self):
        self.func.mkdir()

        ckpt_prefix = os.path.join(self.OUTPUT_DIR_CKP, "ckpt")
        ckpt = tf.train.Checkpoint(gen_G_optim=self.gen_G_optim,
                                   gen_F_optim=self.gen_F_optim,
                                   disc_X_optim=self.disc_X_optim,
                                   disc_Y_optim=self.disc_Y_optim,
                                   gen_G=self.gen_G,
                                   gen_F=self.gen_F,
                                   disc_X=self.disc_X,
                                   disc_Y=self.disc_Y)

        gen_G_losses = []
        gen_F_losses = []
        disc_X_losses = []
        disc_Y_losses = []

        for ep in range(self.epochs):
            print(f'||  Epoch : {ep}/{self.epochs}  ||')
            ds_loader = Loader()
            full_ds = ds_loader.load()

            full_ds = configure_for_performance(full_ds, self.cnt, self.batchsz, shuffle=False)
            full_ds_iter = iter(full_ds)

            start = time.time()

            for step in range(self.cnt//self.batchsz):
                monet, photo = next(full_ds_iter)

                total_loss_G, total_loss_F, disc_X_loss, disc_Y_loss, fake_y, cycle_x = self.train_step(photo, monet)

                gen_G_losses.append(total_loss_G)
                gen_F_losses.append(total_loss_F)
                disc_X_losses.append(disc_X_loss)
                disc_Y_losses.append(disc_Y_loss)

                if step % 10 == 0:
                    print(f'||  step : {step}/{self.cnt//self.batchsz}    |   GenLoss : {total_loss_G}  |   DiscLoss : {disc_X_loss}||')
                    # naming을 바꿀 필요가 있다.
                    self.func.save_sampling(fake_y, ep, step, 'X2Y')
                    self.func.save_sampling(cycle_x, ep, step, 'X2Y2X')

            ckpt.save(file_prefix=ckpt_prefix)
