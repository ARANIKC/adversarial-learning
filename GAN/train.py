""" Script for training the GAN on CIFAR-10 data
"""

import tensorflow as tf
import numpy as np
import keras
import NetworkGenerator.GAN as GAN
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS


def train(models, data, epochs, batch_size=128, checkpoint_sample=1,
          highest_pixel_value=255):
    """
    train the network using the provided data
    :param models: GAN network components (gen, dis, comb)
        # make sure that the models are compiled before passing here.
    :param data: original data samples
    :param epochs: number of epochs to train for
    :param batch_size: batch_size for training
    :param checkpoint_sample: generate samples after these many epochs
    :param highest_pixel_value: highest pixel value for RGB images
    :return: Nothing
    """
    # setup the network models:
    generator, discriminator, combined = models

    # extract the random noise shape from the model
    noise_size = int(generator.inputs[0].shape[-1])

    # Rescale to range [0, 1)
    data = data.astype(np.float32) / highest_pixel_value

    # for the discriminator training, we create a batch of
    # half positive examples and half negative examples
    half_batch = int(batch_size / 2)

    # define the helper method for saving snapshots of generated
    # samples during the training process
    def save_imgs(epch, rows=5, columns=5):
        r, c = rows, columns
        rnd_noise = np.random.normal(0, 1, (r * c, noise_size))
        gen_samples = generator.predict(rnd_noise)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_samples[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("generated_samples/cifar-10_epoch_%d.png" % (epch + 1))
        plt.close()

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        d_accs = []
        num_batches = int(np.ceil(epochs / batch_size))
        for _ in range(num_batches):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, data.shape[0], half_batch)
            imgs = data[idx]

            noise = np.random.normal(0, 1, (half_batch, noise_size))

            # Generate a half batch of new images
            gen_imgs = generator.predict(noise)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss, d_acc = 0.5 * np.add(d_loss_real, d_loss_fake)

            d_losses.append(d_loss)
            d_accs.append(d_acc)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, noise_size))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            # make sure that during the training of the combined model, the discriminator
            # is turned off. This way, only generator is trained
            discriminator.trainable = False
            g_loss = combined.train_on_batch(noise, valid_y)
            discriminator.trainable = True

            g_losses.append(g_loss)

        # Plot the progress
        avg_d_loss = np.mean(d_losses)
        avg_d_acc = 100 * np.mean(d_accs)
        avg_g_loss = np.mean(g_losses)
        print("%d [D_loss: %f, acc.: %.2f%%] [G_loss: %f]" % (epoch + 1, avg_d_loss, avg_d_acc, avg_g_loss))

        # If at save interval => save generated image samples
        if epoch == 0 or (epoch + 1) % checkpoint_sample == 0:
            save_imgs(epoch)


def get_data():
    """
    obtain the cifar-10 dataset from the keras_datasets
    :return: data_x, data_y
    """
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    # combine all the data since this is a gan
    data_x = np.concatenate((train_x, test_x))
    data_y = np.concatenate((train_y, test_y))

    # return the combined_data
    return data_x, data_y


def main(_):
    """
    main function for the script ...
    :param _: nothing (ignore this parameter)
    :return: None
    """
    # setup the data
    x, _ = get_data()  # ignore the labels

    # print the information about the data:
    print("Input_Images:", x.shape)

    # obtain the network models
    gen, dis, combined = GAN.get_models(FLAGS.input_noise_size)

    # compile the models:
    combined.compile(optimizer=keras.optimizers.Nadam(lr=FLAGS.combined_lr),
                     loss='binary_crossentropy')

    dis.trainable = True
    dis.compile(optimizer=keras.optimizers.Adam(lr=FLAGS.discriminator_lr),
                loss='binary_crossentropy', metrics=['accuracy'])

    train(
        models=(gen, dis, combined),
        data=x,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        checkpoint_sample=FLAGS.sample_interval
    )

    # save the trained models:
    gen.save("Generator.dnn")
    dis.save("Discriminator.dnn")
    combined.save("Combined.dnn")


if __name__ == '__main__':
    # define the optional flags for the app to run
    flags.DEFINE_integer("input_noise_size", 16,
                         "Number of dimensions of the input noise samples")
    flags.DEFINE_float("generator_lr", 3e-3,
                       "Learning rate for training the Generator")
    flags.DEFINE_float("discriminator_lr", 3e-3,
                       "Learning rate for training the Discriminator")
    flags.DEFINE_float("combined_lr", 3e-3,
                       "Learning rate for training the Combined Model")
    flags.DEFINE_integer("epochs", 12,
                         "Number of epochs for training the GAN")
    flags.DEFINE_integer("batch_size", 128,
                         "Batch size for the SGD")
    flags.DEFINE_integer("sample_interval", 1,
                         "Number of intervals after which samples are to be saved")

    # start the main function
    tf.app.run(main)
