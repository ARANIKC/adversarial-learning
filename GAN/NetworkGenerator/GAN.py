""" Module building the GAN object
"""

from keras import Model
from keras import Sequential
from keras.layers import Conv2DTranspose, Dense, \
    Reshape, Conv2D, Flatten, BatchNormalization, LeakyReLU
from keras.optimizers import Adam


def _generator(inp_size):
    """
    construct the generator model
    :param inp_size: size of the input noise vector
    :return: generator model
    """

    # define the generator model
    generator_model = Sequential()

    # dense block for bringing spatial integrity
    # following is a simple linear projection
    generator_model.add(Dense(4096, input_shape=(inp_size,)))

    # reshape it properly
    generator_model.add(Reshape(target_shape=(8, 8, 64)))
    generator_model.add(BatchNormalization())

    # de-convolutional block 1: inp: 8 x 8 x 64
    generator_model.add(Conv2DTranspose(filters=32, kernel_size=(7, 7),
                                        strides=(2, 2), padding='same',
                                        activation='relu'))
    generator_model.add(BatchNormalization())
    generator_model.add(Conv2DTranspose(filters=32, kernel_size=(7, 7),
                                        padding='same', activation='relu'))
    generator_model.add(BatchNormalization())
    generator_model.add(Conv2DTranspose(filters=32, kernel_size=(7, 7),
                                        padding='same', activation='relu'))
    generator_model.add(BatchNormalization())

    # de-convolutional block 2: inp 16 x 16 x 32
    generator_model.add(Conv2DTranspose(filters=16, kernel_size=(5, 5),
                                        strides=(2, 2), padding='same',
                                        activation='relu'))
    generator_model.add(BatchNormalization())
    generator_model.add(Conv2DTranspose(filters=16, kernel_size=(5, 5),
                                        padding='same', activation='relu'))
    generator_model.add(BatchNormalization())
    generator_model.add(Conv2DTranspose(filters=16, kernel_size=(5, 5),
                                        padding='same', activation='relu'))
    generator_model.add(BatchNormalization())

    # de-convolutional block 3: inp 32 x 32 x 16
    generator_model.add(Conv2DTranspose(filters=8, kernel_size=(3, 3),
                                        padding='same', activation='relu'))
    generator_model.add(BatchNormalization())
    generator_model.add(Conv2DTranspose(filters=8, kernel_size=(3, 3),
                                        padding='same', activation='relu'))

    generator_model.add(Conv2DTranspose(filters=3, kernel_size=(3, 3),
                                        padding='same', activation='tanh'))

    # final output of the model is 32 x 32 x 3 (cifar-10 image size)
    return generator_model


def _discriminator(inp_shape):
    """
    construct the discriminator model
    :param inp_shape: shape of the input to the discriminator
            should be a tuple of the form (height, width, channels)
    :return: discriminator model
    """

    # define the discriminator model:
    discriminator_model = Sequential()

    # Convolutional block 1: inp 32 x 32 x 3
    discriminator_model.add(Conv2D(filters=16, kernel_size=(3, 3),
                                   padding='same', input_shape=inp_shape))
    discriminator_model.add(LeakyReLU(alpha=0.2))

    discriminator_model.add(Conv2D(filters=16, kernel_size=(3, 3),
                                   padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    discriminator_model.add(Conv2D(filters=16, kernel_size=(3, 3),
                                   padding='same',
                                   strides=(2, 2)))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    # Convolutional block 2: inp 16 x 16 x 16
    discriminator_model.add(Conv2D(filters=32, kernel_size=(5, 5),
                                   padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    discriminator_model.add(Conv2D(filters=32, kernel_size=(5, 5),
                                   padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    discriminator_model.add(Conv2D(filters=32, kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2)))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    # Convolutional block 3: inp 8 x 8 x 32
    discriminator_model.add(Conv2D(filters=64, kernel_size=(5, 5),
                                   padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    discriminator_model.add(Conv2D(filters=64, kernel_size=(5, 5),
                                   padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    discriminator_model.add(Conv2D(filters=64, kernel_size=(5, 5),
                                   padding='same'))

    # flatten the activations: inp 8 x 8 x 64 := 4096
    discriminator_model.add(Flatten())
    discriminator_model.add(LeakyReLU(alpha=0.2))
    discriminator_model.add(BatchNormalization())

    # add final few dense layers:
    discriminator_model.add(Dense(units=1, activation='sigmoid'))

    # return the created model
    return discriminator_model


def get_models(noise_size, dis_lr, comb_lr):
    """
    generate the GAN models
    :param noise_size: size of the input noise to the model
    :param dis_lr: learning rate for the discriminator
    :param comb_lr: learning rate for the combined model
    :return: Generator, Discriminator and Combined models
    """
    # obtain the generator model
    gen = _generator(noise_size)

    # obtain the discriminator model:
    # input shape is of Cifar-10 images: i.e. 32 x 32 x 3
    dis = _discriminator(inp_shape=(32, 32, 3))
    dis.compile(optimizer=Adam(dis_lr, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    dis.trainable = False

    # create the combined model:
    comb = Model(inputs=gen.inputs, outputs=dis(gen.outputs))
    comb.compile(optimizer=Adam(comb_lr, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # return the three created models:
    return gen, dis, comb


# print the shape information of the three models
if __name__ == '__main__':
    inp = 16
    generator, discriminator, combined = get_models(inp, 3e-3, 3e-3)

    # print the description about the three models
    print("Generator:", generator.inputs, "->", generator.outputs)
    print("Discriminator:", discriminator.inputs, "->", discriminator.outputs)
    print("Combined:", combined.inputs, "->", discriminator.outputs)

    # show the discriminator summary
    discriminator.summary()

    # also show the combined summary
    combined.summary()
