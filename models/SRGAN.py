from abc import ABC
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, UpSampling2D


class ResidualDenseBlock_5C(tf.keras.Model, ABC):
    def __init__(self, gc):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = Conv2D(gc, kernel_size=3, padding='same')
        self.conv2 = Conv2D(gc, kernel_size=3, padding='same')
        self.conv3 = Conv2D(gc, kernel_size=3, padding='same')
        self.conv4 = Conv2D(gc, kernel_size=3, padding='same')
        self.conv5 = Conv2D(gc, kernel_size=3, padding='same')
        self.lrelu = LeakyReLU(0.2)

    def call(self, x, training=None, mask=None):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(tf.concat([x, x1], -1)))
        x3 = self.lrelu(self.conv3(tf.concat([x, x1, x2], -1)))
        x4 = self.lrelu(self.conv4(tf.concat([x, x1, x2, x3], -1)))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], -1))
        return x5 * 0.2 + x


class RRDB(tf.keras.Model, ABC):
    def __init__(self, gc):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(gc)
        self.RDB2 = ResidualDenseBlock_5C(gc)
        self.RDB3 = ResidualDenseBlock_5C(gc)

    def call(self, x, training=None, mask=None):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(tf.keras.Model, ABC):
    def __init__(self, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        layers = []
        for _ in range(nb):
            layers.append(RRDB(gc))

        self.conv_first = Conv2D(32, kernel_size=3, padding='same')
        self.RRDB_trunk = Sequential(layers)
        self.trunk_conv = Conv2D(gc, kernel_size=3, padding='same')
        self.upsamp1 = UpSampling2D(size=(2, 2), interpolation='nearest')
        self.upconv1 = Conv2D(nf, kernel_size=3, padding='same')
        self.upsamp2 = UpSampling2D(size=(2, 2), interpolation='nearest')
        self.upconv2 = Conv2D(nf, kernel_size=3, padding='same')
        self.HRconv = Conv2D(nf, kernel_size=3, padding='same')
        self.conv_last = Conv2D(3, kernel_size=3, padding='same')

        self.lrelu = LeakyReLU(0.2)

    def call(self, x, training=None, mask=None):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(self.upsamp1(fea)))
        fea = self.lrelu(self.upconv2(self.upsamp2(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
