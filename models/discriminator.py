from abc import ABC
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dense, Flatten, BatchNormalization


class NLayerDiscriminator(tf.keras.Model, ABC):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [Conv2D(input_nc, ndf, kernel_size=kw, stride=2, padding=1), LeakyReLU(0.2)]

        for n in range(1, n_layers):
            nf_mult = min(2 ** n, 8)
            sequence += [
                Conv2D(ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                BatchNormalization(ndf * nf_mult),
                LeakyReLU(0.2)
            ]

        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Conv2D(ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            BatchNormalization(ndf * nf_mult),
            LeakyReLU(0.2)
        ]

        sequence += [Conv2D(1, kernel_size=kw, stride=1, padding=padw)]
        self.model = Sequential(sequence)

    def forward(self, x):
        return self.model(x)


class Discriminator_VGG_128(tf.keras.Model, ABC):
    def __init__(self, nf=64):
        super(Discriminator_VGG_128, self).__init__()

        self.conv0_0 = Conv2D(nf, kernel_size=3, padding='same')
        self.conv0_1 = Conv2D(nf, kernel_size=4, padding='same')
        self.bn0_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [64, 64, 64]
        self.conv1_0 = Conv2D(nf * 2, kernel_size=3, padding='same')
        self.bn1_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv1_1 = Conv2D(nf * 2, kernel_size=3, padding='same')
        self.bn1_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [128, 32, 32]
        self.conv2_0 = Conv2D(nf * 4, kernel_size=3, padding='same')
        self.bn2_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv2_1 = Conv2D(nf * 4, kernel_size=4, padding='same')
        self.bn2_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [256, 16, 16]
        self.conv3_0 = Conv2D(nf * 8, kernel_size=3, padding='same')
        self.bn3_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv3_1 = Conv2D(nf * 8, kernel_size=4, padding='same')
        self.bn3_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        # [512, 8, 8]
        self.conv4_0 = Conv2D(nf * 8, kernel_size=3, padding='same')
        self.bn4_0 = BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv4_1 = Conv2D(nf * 8, kernel_size=4, padding='same')
        self.bn4_1 = BatchNormalization(momentum=0.1, epsilon=1e-05)

        self.flatten = Flatten()
        self.linear1 = Dense(100)
        self.linear2 = Dense(1)

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

        fea = self.flatten(fea)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class VGGFeatureExtractor(tf.keras.Model, ABC):
    def __init__(self, feature_layer=34, use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        model = VGG19()

        mean = tf.constant([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = tf.constant([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.features = Sequential(list(model.features.children())[:(feature_layer + 1)])

        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output
