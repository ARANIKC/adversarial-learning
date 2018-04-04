""" Module building the GAN object
"""

from keras import Sequential
from keras.layers import Conv2DTranspose, Dense, \
    Reshape, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras import Model


def _generator(inp_size):
    """
    construct the generator model
    :param inp_size: size of the input noise vector
    :return: generator model
    """

    # define the generator model
    generator_model = Sequential()

    # dense block for bringing spatial integrity
    generator_model.add(Dense(128, activation='relu', input_shape=(inp_size,)))
    generator_model.add(Dense(256, activation='relu'))
    generator_model.add(Dense(512, activation='relu'))
    generator_model.add(Dense(1024, activation='relu'))
    generator_model.add(Dense(2048, activation='relu'))
    generator_model.add(Dense(4096, activation='relu'))

    # reshape it properly
    generator_model.add(Reshape(target_shape=(8, 8, 64)))

    # de-convolutional block 1: inp: 8 x 8 x 64
    generator_model.add(Conv2DTranspose(filters=32, kernel_size=(7, 7),
                                        strides=(2, 2), padding='same',
                                        activation='relu'))
    generator_model.add(Conv2DTranspose(filters=32, kernel_size=(7, 7),
                                        padding='same', activation='relu'))
    generator_model.add(Conv2DTranspose(filters=32, kernel_size=(7, 7),
                                        padding='same', activation='relu'))

    # de-convolutional block 2: inp 16 x 16 x 32
    generator_model.add(Conv2DTranspose(filters=16, kernel_size=(5, 5),
                                        strides=(2, 2), padding='same',
                                        activation='relu'))
    generator_model.add(Conv2DTranspose(filters=16, kernel_size=(5, 5),
                                        padding='same', activation='relu'))
    generator_model.add(Conv2DTranspose(filters=16, kernel_size=(5, 5),
                                        padding='same', activation='relu'))

    # de-convolutional block 3: inp 32 x 32 x 16
    generator_model.add(Conv2DTranspose(filters=8, kernel_size=(3, 3),
                                        padding='same', activation='relu'))
    generator_model.add(Conv2DTranspose(filters=8, kernel_size=(3, 3),
                                        padding='same', activation='relu'))
    generator_model.add(Conv2DTranspose(filters=3, kernel_size=(3, 3),
                                        padding='same', activation='sigmoid'))

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
                                   padding='same', activation='relu',
                                   input_shape=inp_shape))
    discriminator_model.add(Conv2D(filters=16, kernel_size=(3, 3),
                                   padding='same', activation='relu'))
    discriminator_model.add(Conv2D(filters=16, kernel_size=(3, 3),
                                   padding='same', activation='relu'))
    # entailing pooling layer
    discriminator_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    # Convolutional block 2: inp 16 x 16 x 16
    discriminator_model.add(Conv2D(filters=32, kernel_size=(5, 5),
                                   padding='same', activation='relu'))
    discriminator_model.add(Conv2D(filters=32, kernel_size=(5, 5),
                                   padding='same', activation='relu'))
    discriminator_model.add(Conv2D(filters=32, kernel_size=(5, 5),
                                   padding='same', activation='relu'))
    # entailing pooling layer
    discriminator_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    # Convolutional block 3: inp 8 x 8 x 32
    discriminator_model.add(Conv2D(filters=64, kernel_size=(5, 5),
                                   padding='same', activation='relu'))
    discriminator_model.add(Conv2D(filters=64, kernel_size=(5, 5),
                                   padding='same', activation='relu'))
    discriminator_model.add(Conv2D(filters=64, kernel_size=(5, 5),
                                   padding='same', activation='relu'))

    # flatten the activations: inp 8 x 8 x 64 := 4096
    discriminator_model.add(Flatten())

    # add final few dense layers:
    discriminator_model.add(Dense(units=1024, activation='relu'))
    discriminator_model.add(Dense(units=256, activation='relu'))
    discriminator_model.add(Dense(units=32, activation='relu'))
    discriminator_model.add(Dense(units=1, activation='sigmoid'))

    # return the created model
    return discriminator_model


def get_models(noise_size, dis_lr, comb_lr):
    """
    generate the GAN models
    :param noise_size: size of the input noise to the model
    :return: Generator, Discriminator and Combined models
    """
    # obtain the generator model
    gen = _generator(noise_size)

    # obtain the discriminator model:
    # input shape is of Cifar-10 images: i.e. 32 x 32 x 3
    dis = _discriminator(inp_shape=(32, 32, 3))
    dis.compile(optimizer=Adam(dis_lr, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # create the combined model:
    comb = Model(inputs=gen.inputs, outputs=dis(gen.outputs))
    comb.compile(optimizer=Adam(comb_lr, 0.5), loss='binary_crossentropy')

    # return the three created models:
    return gen, dis, comb


# print the shape information of the three models
if __name__ == '__main__':
    inp = 16
    generator, discriminator, combined = get_models(inp)

    # print the description about the three models
    print("Generator:", generator.inputs, "->", generator.outputs)
    print("Discriminator:", discriminator.inputs, "->", discriminator.outputs)
    print("Combined:", combined.inputs, "->", discriminator.outputs)
