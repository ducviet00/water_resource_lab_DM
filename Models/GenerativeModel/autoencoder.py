import tensorflow as tf
import tensorflow.keras.layers as tfl
print(tf.__version__)

import numpy as np
import pandas as pd
#print(Dense)

# class Conv1DTranspose(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, strides=1, padding='valid'):
#         super().__init__()
#         self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
#             filters, (kernel_size, 1), (strides, 1), padding)

#     def call(self, x):
#         x = tf.expand_dims(x, axis=2)
#         x = self.conv2dtranspose(x)
#         x = tf.squeeze(x, axis=2)
#         return x


class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_model = tf.keras.Sequential
        ([
            tfl.InputLayer(input_shape=(self.input_dim)),
            tfl.Dense(units=64, activation='relu'),
            tfl.Dense(units=32, activation='relu'),
            tfl.Dense(self.latent_dim + self.latent_dim)
        ])

        self.decoder_model = tf.keras.Sequential  #NOTE: decoder is converted from encoder, but inverse in order and computation
        ([
            tfl.InputLayer(input_shape=(self.latent_dim)),
            tfl.Dense(units=32, activation='relu'),
            tfl.Dense(units=64, activation='relu'),
            tfl.Dense(units=self.input_dim)
        ])

    @tf.function
    def sampling(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(value=self.encoder_model(x),
                                num_or_size_splits=2,
                                axis=1)
        print(mean.shape)
        return mean, logvar
        #return 0, 0

    def reparameterize(self, mean, logvar):
        ''' Sampling from latent space distribution p(z|x) '''
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, sigmoid=False):
        logits = self.decoder_model(z)
        if sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


opt = tf.keras.optimizers.Adam(lr=1e-3)


def log_n_pdf(sample, mean, logvar, axis=1):
    '''function to compute log n of probability distribution'''
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        .5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=axis)


@tf.function
def compute_loss(model, x):
    ''' compute loss value ( - ELBO on the marginal log-likelihood) '''
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_entr = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                         labels=x)
    log_px_z = -tf.reduce_sum(cross_entr, axis=[
        1,
    ])
    log_pz = log_n_pdf(z, 0, 0)
    log_qz_x = log_n_pdf(z, mean, logvar)

    return -tf.reduce_mean(log_px_z + log_pz - log_qz_x)


@tf.function
def grad_compute_and_apply(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradient = tape.gradient(loss, model.trainabel_variables)
    optimizer.apply_gradients(zip(gradient, model.trainabel_variables))


#TODO: OOP wrapper

epochs = 100
input_dim = 8
latent_dim = 5
num_example = 10

rand_vect_for_generation = tf.random.normal(shape=[num_example, latent_dim])
model = VAE(input_dim, latent_dim)


def generate_and_save_data(model, input):
    predicts = model.sample(input)

    dat = pd.DataFrame(data=predicts)
    dat.to_csv('./ProcessedData/generated_data.csv')


if __name__ == "__main__":
    #generate_and_save_data(model,rand_vect_for_generation)
    import time

    #data fetching
    x = pd.read_csv('./RawData/Kontum-daily.csv', index_col=0, header=0)
    x = x.drop('time', axis=1)
    x = x.to_numpy()

    train_size = 10000
    test_size = x.shape[0] - train_size
    batch_size = 32

    train_x = x[:train_size]
    test_x = x[test_size:]

    train_set = tf.data.Dataset.from_tensor_slices(train_x).shuffle(
        train_size).batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices(test_x).shuffle(
        test_size).batch(batch_size)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for tr_x in train_set:
            print(tr_x.shape)
            model.sampling(tr_x)
        #     grad_compute_and_apply(model, tr_x, opt)
        # end_time = time.time()

        # loss = tf.keras.metrics.Mean()
        # for t_x in test_set:
        #     loss(compute_loss(model, x))
        #     elbo_metric = -loss.result()
        #     print('Epoch: {}, Test set ELBO: {}, '
        #           'time elapse for current epoch {}'.format(
        #               epoch, elbo_metric, end_time - start_time))
