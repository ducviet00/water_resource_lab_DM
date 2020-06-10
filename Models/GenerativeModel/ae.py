import pandas as pd
import numpy as np
import keras.backend as K


class AE:
    def __init__(self,
                 input_dim,
                 intermediate_dim,
                 latent_dim,
                 noise_div=0.001,
                 batchsize=64,
                 epochs=100,
                 lr=1e-4,
                 lr_decay=1e-6,
                 logdir='./Log/AE/'):
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.noise_stddiv = noise_div
        self.batch_size = batchsize
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.log_dir = logdir

        self.ae_model, self.encoder_model, self.decoder_model = self.build_model(
        )

    def build_model(self):
        from keras.layers import Input, Dense, Lambda, GaussianNoise
        from keras.models import Model

        input = Input(shape=(self.input_dim, ))
        encoded_h = Dense(self.intermediate_dim, activation='relu')(input)
        encoded_z = Dense(self.latent_dim, activation='relu')(encoded_h)

        decoded_h = Dense(self.intermediate_dim, activation='relu')(encoded_z)
        decoded_out = Dense(self.input_dim, activation='sigmoid')(decoded_h)

        encoder = Model(input, encoded_z)
        ae = Model(input, decoded_out)
        ae.summary()

        decode_in = Input(shape=(self.latent_dim, ))
        decoded_h = ae.layers[-2](decode_in)
        decoded_out = ae.layers[-1](decoded_h)
        decoder = Model(decode_in, decoded_out)
        decoder.summary()
        # import os
        # os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
        from keras.utils.vis_utils import plot_model
        ae.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['mae', 'mape'])
        plot_model(model=ae,
                   to_file=self.log_dir + '/model.png',
                   show_shapes=True)

        return ae, encoder, decoder

    def train_model(self, x_train, x_val, mode):
        if mode == 'train':
            from keras.callbacks import EarlyStopping, ModelCheckpoint

            callbacks = []
            es = EarlyStopping(monitor='val_loss',
                               patience=50,
                               restore_best_weights=True)
            checkpoint = ModelCheckpoint(self.log_dir + 'ae.hdf5',
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)
            callbacks.append(es)
            callbacks.append(checkpoint)

            self.ae_model.fit(x_train,
                              x_train,
                              shuffle=False,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              validation_data=(x_val, x_val),
                              callbacks=callbacks)
        else:
            self.ae_model.load_weights(self.log_dir + 'ae.hdf5')

    def generate_data_and_save(self, scaler, shape, xs):
        loss, mae, mape = self.ae_model.evaluate(xs, xs, verbose=2)
        print(f'loss-mse: {loss} mae: {mae} mape: {mape}')
        mask = np.zeros(shape=shape)
        dat = pd.DataFrame(np.zeros(shape=shape))
        dat.columns = [
            'wind', 'max_temp', 'min_temp', 'solar_r', 'humidity', 'discharge',
            'rain', 'water_level'
        ]

        for i, x in enumerate(xs):
            z = self.encoder_model.predict(x.reshape(1, -1), batch_size=1)
            z_noisy = z + np.random.randn(z.shape[0],
                                          z.shape[1]) * self.noise_stddiv
            if i < 5:
                print('Dat')
                print(z)
                print(z_noisy)

                print('real dat')
                print(x)
                print(self.decoder_model.predict(z_noisy, batch_size=1))
                print(self.decoder_model.predict(z, batch_size=1))

            gen_x_noisy = self.decoder_model.predict(z_noisy, batch_size=1)
            mask[0] = gen_x_noisy
            dat.iloc[i] = scaler.inverse_transform(mask)[0]

        dat = dat.round(3)
        print(dat.head())
        dat.to_csv('./ProcessedData/GeneratedData/generated_dat_ae.csv',
                   index=None)


if __name__ == "__main__":
    ae = AE(input_dim=8,
            intermediate_dim=32,
            latent_dim=64,
            noise_div=1e-5,
            batchsize=128,
            epochs=800)

    x = pd.read_csv('./RawData/Kontum-daily.csv', index_col=0, header=0)
    x = x.drop('time', axis=1)
    x = x.dropna(axis=0)
    x = x.to_numpy()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x)
    x_norm = scaler.transform(x)

    train_size = 10000
    test_size = x.shape[0] - train_size
    train_x = x_norm[:train_size]
    test_x = x_norm[-test_size:]

    print(train_x.shape)
    print(test_x.shape)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Train mode')

    args = parser.parse_args()
    ae.train_model(train_x, test_x, args.mode)
    ae.generate_data_and_save(scaler, shape=x.shape, xs=x_norm)
    # from keras.objectives import binary_crossentropy

    # y_true = [[0, 1], [0, 0]]
    # y_pred = [[0.6, 0.4], [0.4, 0.6]]
    # loss = binary_crossentropy(y_true, y_pred)
    # #assert loss.shape == (2, )
    # loss.numpy()
    # print(loss)
