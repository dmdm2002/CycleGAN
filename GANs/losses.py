import tensorflow as tf
import tensorflow.keras as keras

adv_loss_fn = keras.losses.MeanSquaredError()

def Cycle_gen_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

def Cycle_disc_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)

    return (real_loss + fake_loss) * 0.5