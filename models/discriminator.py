from abc import ABC
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense


class NLayerDiscriminator(tf.keras.Model, ABC):
    def __init__(self):
        super(NLayerDiscriminator, self).__init__()
        

class Discriminator_VGG_128(tf.keras.Model, ABC):
    def __init__(self, nf):
        super(Discriminator_VGG_128, self).__init__()

        self.conv0_0 = Conv2D(nf, 3, 1, 1, bias=True)
        self.conv0_1 = Conv2D(nf, 4, 2, 1, bias=False)
        self.bn0_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [64, 64, 64]
        self.conv1_0 = Conv2D(nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv1_1 = Conv2D(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [128, 32, 32]
        self.conv2_0 = Conv2D(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv2_1 = Conv2D(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [256, 16, 16]
        self.conv3_0 = Conv2D(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv3_1 = Conv2D(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [512, 8, 8]
        self.conv4_0 = Conv2D(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv4_1 = Conv2D(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)

        self.linear1 = Dense(512 * 4 * 4, 100)
        self.linear2 = Dense(100, 1)

        # activation function
        self.lrelu = LeakyReLU(negative_slope=0.2, inplace=True)

    def call(self, x, training=None, mask=None):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out
