import pandas as pd
import numpy as np
import keras.backend as K


class VAE:
    def __init__(self,
                 input_dim,
                 intermediate_dim,
                 latent_dim,
                 batchsize=64,
                 epochs=100,
                 lr=1e-4,
                 lr_decay=1e-6,
                 logdir='./Log/VAE/'):
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.batch_size = batchsize
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.log_dir = logdir

        self.vae_model, self.encoder_model, self.decoder_model = self.build_model(
        )

    def build_model(self):
        from keras.layers import Input, Dense, Lambda
        from keras.models import Model
        #ENCODER
        en_x = Input(shape=(self.input_dim, ))
        en_h = Dense(units=self.intermediate_dim,
                     activation='relu')(en_x)  #(32,8)

        z_mean = Dense(units=self.latent_dim, name='z_mean')(en_h)  # (32,6)
        z_logvar = Dense(units=self.latent_dim, name='z_logvar')(en_h)  #(32,3)

        cus_shape = (self.batch_size, self.latent_dim)

        def sampling(args):
            '''sample a z-value from probability distribution p(z|x)'''
            mean, logvar = args
            #print(mean.shape)
            #batch = K.shape(mean)[0]
            epsilon = K.random_normal(
                shape=cus_shape)  #random from gaussian's distribution

            return mean + epsilon * K.exp(0.5 * logvar)

        z = Lambda(sampling, name='z')([z_mean, z_logvar])
        #DECODER
        encoder = Model(inputs=en_x,
                        outputs=[z_mean, z_logvar, z],
                        name='encoder')
        encoder.summary()

        de_x = Input(shape=(self.latent_dim, ), name='z_sampling')
        decoder_h = Dense(units=self.intermediate_dim, activation='relu')
        decoder_mean = Dense(units=self.input_dim, activation='sigmoid')
        h_decoded = decoder_h(de_x)
        x_decoded_mean = decoder_mean(h_decoded)

        #de_x = Input(batch_shape=(self.batch_size, self.latent_dim))
        #print(de_x.shape)
        # _h_decoded = decoder_h(de_x)
        # _x_decoded_mean = decoder_mean(_h_decoded)

        decoder = Model(inputs=de_x, outputs=x_decoded_mean, name='decoder')
        decoder.summary()
        from keras.utils.vis_utils import plot_model
        plot_model(model=decoder,
                   to_file=self.log_dir + '/model_decoder.png',
                   show_shapes=True)

        vae_out = decoder(encoder(en_x)[2])
        vae = Model(en_x, vae_out, name='vae')

        #define loss and compile model
        from keras.objectives import mse, kullback_leibler_divergence, binary_crossentropy

        def vae_loss(x, x_decoded_mean):
            xent_loss = binary_crossentropy(
                x, x_decoded_mean) * self.input_dim  #reconstruction loss
            kl_loss = -0.5 * K.sum(
                1 + z_logvar - K.square(z_mean) - K.exp(z_logvar),
                axis=-1)  #KL diversion loss
            #kl_loss = kullback_leibler_divergence(x, x_decoded_mean)
            return K.mean(xent_loss + kl_loss)

        #vae.add_loss(vae_loss(en_x, vae_out))
        from keras.optimizers import Adam
        opt = Adam(clipnorm=1.,
                   clipvalue=0.5)  #gradient clipping due to gardient explode
        vae.compile(optimizer=opt, loss=vae_loss, metrics=['mae', 'mse'])
        vae.summary()

        # import os
        # os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
        plot_model(model=vae,
                   to_file=self.log_dir + '/model.png',
                   show_shapes=True)

        return vae, encoder, decoder

    def train_model(self, x_train, x_val, mode):
        if mode == 'train':
            from keras.callbacks import EarlyStopping, ModelCheckpoint

            callbacks = []
            es = EarlyStopping(monitor='val_loss',
                               patience=50,
                               restore_best_weights=True)
            checkpoint = ModelCheckpoint(self.log_dir + 'vae.hdf5',
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)
            callbacks.append(es)
            callbacks.append(checkpoint)

            self.vae_model.fit(x_train,
                               x_train,
                               shuffle=True,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               validation_data=(x_val, x_val),
                               callbacks=callbacks)
            #self.vae_model.save_weights(self.log_dir + 'vae.hdf5')
        else:
            self.vae_model.load_weights(self.log_dir + 'vae.hdf5')

        loss, mae, mape = self.vae_model.evaluate(x_val,
                                                  x_val,
                                                  batch_size=self.batch_size,
                                                  verbose=2)
        print(f'loss-mse: {loss} mae: {mae} mape: {mape}')

    def generate_data_and_save(self, scaler, shape, num=10):
        mask = np.zeros(shape=shape)
        dat = pd.DataFrame(np.zeros(shape=(num**3, self.input_dim)))
        dat.columns = [
            'wind', 'max_temp', 'min_temp', 'solar_r', 'humidity', 'discharge',
            'rain', 'water_level'
        ]

        d1 = np.linspace(-5, 5, num)
        d2 = np.linspace(-5, 5, num)
        d3 = np.linspace(-5, 5, num)
        index = 0
        for i in d1:
            for j in d2:
                for k in d3:
                    # z_sample = np.random.rand(1, self.latent_dim) * 10
                    z_sample = np.array([[i, j, k]])
                    gen_out = self.decoder_model.predict(z_sample)
                    # print('Before rescale:')
                    # print(gen_out)

                    mask[0] = gen_out
                    gen_out = scaler.inverse_transform(mask)[0]
                    dat.iloc[index] = gen_out
                    index += 1
                    # print('After rescale:')
                    # print(gen_out)
        dat = dat.round(5)
        print(dat.head())
        dat.to_csv('./ProcessedData/GeneratedData/generated_dat.csv',
                   index=None)

    def gen_data_seasonal(self, scaler, shape, xs, n=1, variance=0.1):
        mask = np.zeros(shape=shape)
        dat = []
        for i in range(0, shape[0] - self.batch_size, self.batch_size):
            z = self.encoder_model.predict(xs[i:i + self.batch_size],
                                           batch_size=self.batch_size)[2]
            for j in range(n):
                z_noisy = z + np.random.randn(self.batch_size,
                                              self.latent_dim) * variance
                gen_out = self.decoder_model.predict(z)
                mask[0:self.batch_size] = gen_out
                gen_out = scaler.inverse_transform(mask)[0:self.batch_size]

                dat.append(gen_out)

        dat = np.array(dat)
        dat = dat.reshape(dat.shape[0] * dat.shape[1], -1)
        print(dat.shape)
        dat = pd.DataFrame(dat).round(3)
        dat.columns = [
            'wind', 'max_temp', 'min_temp', 'solar_r', 'humidity', 'discharge',
            'rain', 'water_level', 'vapor'
        ]
        print(dat.head())
        dat.to_csv('./ProcessedData/GeneratedData/generated_dat.csv',
                   index=None)


if __name__ == "__main__":
    vae = VAE(input_dim=9,
              intermediate_dim=16,
              latent_dim=16,
              batchsize=64,
              epochs=400)

    x = pd.read_csv('./RawData/Kontum-daily.csv', index_col=0, header=0)
    x = x.dropna(axis=0)
    x = x.to_numpy()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x)
    x_norm = scaler.transform(x)

    train_size = 9984
    test_size = x.shape[0] - train_size - 57
    train_x = x_norm[:train_size]
    test_x = x_norm[-test_size:]

    print(train_x.shape)
    print(test_x.shape)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Train mode')

    args = parser.parse_args()
    vae.train_model(train_x, test_x, args.mode)
    #vae.generate_data_and_save(scaler, shape=x.shape, num=30)
    vae.gen_data_seasonal(scaler, x_norm.shape, x_norm, n=1, variance=0)
    # from keras.objectives import binary_crossentropy

    # y_true = [[0, 1], [0, 0]]
    # y_pred = [[0.6, 0.4], [0.4, 0.6]]
    # loss = binary_crossentropy(y_true, y_pred)
    # #assert loss.shape == (2, )
    # loss.numpy()
    # print(loss)
