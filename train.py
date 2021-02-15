import time
import tensorflow as tf
import numpy as np

from models.SRGAN import RRDBNet
from models.discriminator import Discriminator_VGG_128, VGGFeatureExtractor


from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


def cri_gan(x, target_is_real):
    if target_is_real:
        target_label = tf.fill(x.shape, 1.0)
    else:
        target_label = tf.fill(x.shape, 0.0)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(x, target_label)
    return loss


class SrganTrainer:
    def __init__(self,
                 generator,
                 discriminator,
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        self.generator = generator
        self.discriminator = discriminator
        self.netF = VGGFeatureExtractor(feature_layer=34)

        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_dataset, steps=200):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        for lr, hr in train_dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 10 == 0:
                print(
                    f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f},'
                    f' discriminator loss = {dls_metric.result():.4f}'
                )
                pls_metric.reset_states()
                dls_metric.reset_states()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr)

            l_g_total = 0

            l_g_pix = 1e-2 * tf.reduce_mean(tf.abs(sr, hr))
            l_g_total += l_g_pix

            real_fea = self.netF(hr)
            fake_fea = self.netF(sr)
            l_g_fea = tf.reduce_mean(tf.abs(real_fea - fake_fea))
            l_g_total += l_g_fea

            pred_g_fake = self.discriminator(sr)
            pred_d_real = self.discriminator(hr)
            l_g_gan = 5e-3 * (
                cri_gan(pred_d_real - pred_g_fake, False) +
                cri_gan(pred_g_fake - pred_d_real, True)) / 2
            l_g_total += l_g_gan

            pred_d_real = self.discriminator(hr)
            pred_d_fake = self.discriminator(sr)
            l_d_real = cri_gan(pred_d_real - pred_d_fake, True)
            l_d_fake = cri_gan(pred_d_fake - pred_d_real, False)
            l_d_total = (l_d_real + l_d_fake) / 2

        gradients_of_generator = gen_tape.gradient(l_g_total, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(l_d_total, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return l_g_total, l_d_total


def main():
    generator = RRDBNet(nf=64, nb=23)
    discriminator = Discriminator_VGG_128()
    trainer = SrganTrainer(
        generator=generator,
        discriminator=discriminator
    )
    trainer.train(None, 12)
    return 1


if __name__ == '__main__':
    main()
