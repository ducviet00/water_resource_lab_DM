import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ProcessingData.reprocess_daily import extract_data, ed_extract_data


class Ensemble:
    def __init__(self, mode, model_kind, **kwargs):
        self.mode = mode
        self.model_kind = model_kind

        self.log_dir = kwargs.get('log_dir')
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')

        self.data_file = self._data_kwargs.get('data_file')
        self.dt_split_point_outer = self._data_kwargs.get('split_point_outer')
        self.dt_split_point_inner = self._data_kwargs.get('split_point_inner')
        self.cols_x = self._data_kwargs.get('cols_x')
        self.cols_y = self._data_kwargs.get('cols_y')
        self.target_timestep = self._data_kwargs.get('target_timestep')
        self.window_size = self._data_kwargs.get('window_size')
        self.norm_method = self._data_kwargs.get('norm_method')

        self.batch_size = self._model_kwargs.get('batch_size')
        self.epoch_min = self._model_kwargs.get('epoch_min')
        self.epoch_max = self._model_kwargs.get('epoch_max')
        self.epoch_step = self._model_kwargs.get('epoch_step')
        self.epochs_out = self._model_kwargs.get('epochs_out')
        self.input_dim = self._model_kwargs.get('in_dim')
        self.output_dim = self._model_kwargs.get('out_dim')
        self.patience = self._model_kwargs.get('patience')
        self.dropout = self._model_kwargs.get('dropout')

        self.data = self.generate_data()
        self.inner_model = self.build_model_inner()
        self.outer_model = self.build_model_outer()

    def generate_data(self):
        dat = pd.read_csv(self.data_file, header=0, index_col=0)
        dat_q = pd.read_csv('./RawData/Kontum-daily.csv', header=0, index_col=0)
        #gen_dat = gen_dat.to_numpy()
        dat = dat.to_numpy()

        data = {}
        data['shape'] = dat.shape

        test_outer = int(dat.shape[0] * self.dt_split_point_outer)
        train_inner = int((dat.shape[0] - test_outer) * (1 - self.dt_split_point_inner))
        #train_outer = dat.shape[0] - train_inner - test_outer

        if self.model_kind == 'rnn_cnn':
            x, y, scaler = extract_data(dataframe=dat,
                                        window_size=self.window_size,
                                        target_timstep=self.target_timestep,
                                        cols_x=self.cols_x,
                                        cols_y=self.cols_y,
                                        mode=self.norm_method)

            x_train_in, y_train_in = x[:train_inner, :], y[:train_inner, :]
            x_test_in, y_test_in = x[train_inner:-test_outer, :], y[train_inner:-test_outer, :]
            x_test_out, y_test_out = x[-test_outer:, :], y[-test_outer:, :]

            for cat in ["train_in", "test_in", "test_out"]:
                x, y = locals()["x_" + cat], locals()["y_" + cat]
                print(cat, "x: ", x.shape, "y: ", y.shape)
                data["x_" + cat] = x
                data["y_" + cat] = y

        elif self.model_kind == 'en_de':
            en_x, de_x, de_y, scaler = ed_extract_data(dataframe=dat,
                                                       window_size=self.window_size,
                                                       target_timstep=self.target_timestep,
                                                       cols_x=self.cols_x,
                                                       cols_y=self.cols_y,
                                                       mode=self.norm_method)

            en_x_train_in, de_x_train_in, y_train_in = en_x[:train_inner, :], de_x[:
                                                                                   train_inner, :], de_y[:
                                                                                                         train_inner, :]
            en_x_test_in, de_x_test_in, y_test_in = en_x[train_inner:-test_outer, :], de_x[
                train_inner:-test_outer, :], de_y[train_inner:-test_outer, :]
            en_x_test_out, de_x_test_out, y_test_out = en_x[-test_outer:, :], de_x[-test_outer:, :], de_y[
                -test_outer:, :]

            for cat in ["train_in", "test_in", "test_out"]:
                en_x, de_x, de_y = locals()["en_x_" + cat], locals()["de_x_" + cat], locals()["y_" + cat]
                print(cat, "en_x: ", en_x.shape, "de_x: ", de_x.shape, "de_y: ", de_y.shape)
                data["en_x_" + cat] = en_x
                data["de_x_" + cat] = de_x
                data["y_" + cat] = de_y
        #data['y_train_out'] = data['y_test_in']

        data['scaler'] = scaler
        return data

    def build_model_inner(self):
        if self.model_kind == 'rnn_cnn':
            from Models.multi_rnn_cnn import model_builder
            model = model_builder(self.input_dim, self.output_dim, self.target_timestep, self.dropout)
            model.save_weights(self.log_dir + 'ModelPool/init_model.hdf5')
        elif self.model_kind == 'en_de':
            from Models.en_de import model_builder
            model = model_builder(self.input_dim, self.output_dim, self.target_timestep, self.dropout)
            model.save_weights(self.log_dir + 'ModelPool/init_model.hdf5')
        return model

    def train_model_inner(self):
        train_shape = self.data['y_test_in'].shape
        test_shape = self.data['y_test_out'].shape
        #print(train_shape)
        #print(test_shape)
        step = int((self.epoch_max - self.epoch_min) / self.epoch_step) + 1
        self.data['sub_model'] = step

        x_train_out = np.zeros(shape=(train_shape[0], self.target_timestep, step, train_shape[2]))
        x_test_out = np.zeros(shape=(test_shape[0], self.target_timestep, step, test_shape[2]))
        j = 0  #submodel index

        if (self.mode == 'train' or self.mode == 'train-inner'):
            from Models.en_de import train_model as ed_train

            for epoch in range(self.epoch_min, self.epoch_max + 1, self.epoch_step):
                self.inner_model.load_weights(self.log_dir + 'ModelPool/init_model.hdf5')

                if self.model_kind == 'rnn_cnn':
                    from Models.multi_rnn_cnn import train_model
                    self.inner_model, _ = train_model(self.inner_model,
                                                      self.data['x_train_in'],
                                                      self.data['y_train_in'],
                                                      self.batch_size,
                                                      epoch,
                                                      save_dir=self.log_dir + 'ModelPool/')
                elif self.model_kind == 'en_de':
                    #TODO: finish all things related encoder decoder model
                    from Models.en_de import train_model
                    self.inner_model, _ = train_model(self.inner_model,
                                                      self.data['en_x_train_in'],
                                                      self.data['de_x_train_in'],
                                                      self.data['y_train_in'],
                                                      self.batch_size,
                                                      epoch,
                                                      save_dir=self.log_dir + 'ModelPool/')
                train, test = self.predict_in()
                # x_train_out = pd.concat([x_train_out,train],axis=1)
                # x_test_out = pd.concat([x_test_out,test],axis=1)
                for i in range(self.target_timestep):
                    x_train_out[:, i, j, :] = train[:, i, :]
                    x_test_out[:, i, j, :] = test[:, i, :]
                j += 1
        else:
            for epoch in range(self.epoch_min, self.epoch_max + 1, self.epoch_step):
                if self.model_kind == 'rnn_cnn':
                    self.inner_model.load_weights(self.log_dir + f'ModelPool/best_model_{epoch}.hdf5')
                    train, test = self.predict_in()
                elif self.model_kind == 'en_de':
                    self.inner_model.load_weights(self.log_dir + f'ModelPool/ed_best_model_{epoch}.hdf5')
                    train, test = self.predict_in()
                # x_train_out = pd.concat([x_train_out,train],axis=1)
                # x_test_out = pd.concat([x_test_out,test],axis=1)

                for i in range(self.target_timestep):
                    x_train_out[:, i, j, :] = train[:, i, :]
                    x_test_out[:, i, j, :] = test[:, i, :]
                j += 1

        #data preparation for outer model
        self.data_out_generate(x_train_out, x_test_out)

    def predict_in(self):
        # x_train_out = self.inner_model.predict(self.data['x_test_in']).reshape(-3,self.target_timestep * self.output_dim)
        # x_test_out = self.inner_model.predict(self.data['x_test_out']).reshape(-3,self.target_timestep * self.output_dim)
        if self.model_kind == 'rnn_cnn':
            x_train_out = self.inner_model.predict(self.data['x_test_in'])
            x_test_out = self.inner_model.predict(self.data['x_test_out'])
        elif self.model_kind == 'en_de':
            x_train_out = self.inner_model.predict([self.data['en_x_test_in'], self.data['de_x_test_in']])
            x_test_out = self.inner_model.predict([self.data['en_x_test_out'], self.data['de_x_test_out']])
        #print(x_train_out[:3])
        #return pd.DataFrame(x_train_out), pd.DataFrame(x_test_out)
        return x_train_out, x_test_out

    def data_out_generate(self, x_train_out, x_test_out):
        self.data['x_train_out'] = x_train_out
        self.data['y_train_out'] = self.data['y_test_in']  #.reshape(-3,self.target_timestep * self.output_dim)
        self.data['x_test_out_submodel'] = x_test_out
        self.data['y_test_out'] = self.data['y_test_out']  #.reshape(-3,self.target_timestep * self.output_dim)

    #TODO: change to multiple timestep
    def build_model_outer(self):
        #from sklearn.svm import SVR
        from keras.layers import Dense, Input, Bidirectional, LSTM, Reshape, Concatenate, Conv1D, TimeDistributed
        from keras.models import Model

        #model = SVR(kernel='poly',C=1.0,epsilon=0.1)

        self.train_model_inner()
        in_shape = self.data['x_train_out'].shape
        print(f'Input shape: {in_shape}')
        #print(self.data['x_train_out'][:3])
        #print(self.data['x_train_out'].head())

        input_submodel = Input(shape=(self.data['sub_model'], self.output_dim))
        input_val_x = Input(shape=(self.window_size, self.input_dim))

        # conv = Conv1D(filters=16,kernel_size=2,strides=1,padding='same')
        # conv_out = conv(input_val_x)
        # conv_2 = Conv1D(filters=32,kernel_size=3,padding='same')
        # conv_out_2 = conv_2(conv_out)
        # conv_3 = Conv1D(filters=64,kernel_size=4,padding='same')
        # conv_out_3 = conv_3(conv_out_2)

        rnn_1 = Bidirectional(
            LSTM(units=128,
                 return_sequences=True,
                 return_state=True,
                 dropout=self.dropout,
                 recurrent_dropout=self.dropout))
        rnn_1_out, forward_h, forward_c, backward_h, backward_c = rnn_1(input_val_x)
        state_h = Concatenate(axis=-1)([forward_h, backward_h])
        state_c = Concatenate(axis=-1)([forward_c, backward_c])

        # rnn_2 = LSTM(units=256,return_sequences=True)
        # rnn_2_out = rnn_2(input_submodel,initial_state=[state_h,state_c])

        rnn_2 = LSTM(units=256, return_sequences=False, dropout=self.dropout, recurrent_dropout=self.dropout)
        rnn_2_out = rnn_2(input_submodel, initial_state=[state_h, state_c])

        # rnn_3 = LSTM(units=256,return_sequences=False)
        # rnn_3_out = rnn_3(rnn_2_out)

        reshape = Reshape(target_shape=(self.target_timestep, -1))
        dense_in = reshape(rnn_2_out)

        # dense_1 = Dense(units=32,activation='relu')
        # dense_1_out = dense_1(dense_in)

        dense_4 = TimeDistributed(Dense(units=self.output_dim))
        output = dense_4(dense_in)

        model = Model(inputs=[input_submodel, input_val_x], outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

        return model

    def train_model_outer(self):
        if (self.mode == 'train' or self.mode == 'train-outer'):
            from keras.callbacks import EarlyStopping, ModelCheckpoint

            callbacks = []
            #lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
            early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
            checkpoint = ModelCheckpoint(self.log_dir + 'best_model.hdf5',
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True)

            #callbacks.append(lr_schedule)
            callbacks.append(early_stop)
            callbacks.append(checkpoint)

            if self.model_kind == 'rnn_cnn':
                history = self.outer_model.fit(x=[self.data['x_train_out'][:, 0, :, :], self.data['x_test_in']],
                                               y=self.data['y_train_out'],
                                               batch_size=self.batch_size,
                                               epochs=self.epochs_out,
                                               callbacks=callbacks,
                                               validation_split=0.1)
            elif self.model_kind == 'en_de':
                history = self.outer_model.fit(x=[self.data['x_train_out'][:, 0, :, :], self.data['en_x_test_in']],
                                               y=self.data['y_train_out'],
                                               batch_size=self.batch_size,
                                               epochs=self.epochs_out,
                                               callbacks=callbacks,
                                               validation_split=0.1)

            if history is not None:
                self.plot_training_history(history)

        elif self.mode == 'test':
            self.outer_model.load_weights(self.log_dir + 'best_model.hdf5')
            print('Load weight from ' + self.log_dir)

        from keras.utils.vis_utils import plot_model
        import os
        os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
        plot_model(model=self.outer_model, to_file=self.log_dir + 'model.png', show_shapes=True)

    def plot_training_history(self, history):
        fig = plt.figure(figsize=(10, 6))
        #fig.add_subplot(121)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        #plt.plot(history.history['mae'],label='mae')
        plt.legend()

        #fig.add_subplot(122)
        #plt.semilogx(history.history["lr"], history.history["loss"])

        plt.savefig(self.log_dir + 'training_phase.png')

    def predict_and_plot(self):
        x_extract_by_day = self.data['x_test_out_submodel'][:, 0, :, :]
        if self.model_kind == 'rnn_cnn':
            results = self.outer_model.predict(x=[x_extract_by_day, self.data['x_test_out']])
        elif self.model_kind == 'en_de':
            results = self.outer_model.predict(x=[x_extract_by_day, self.data['en_x_test_out']])
        print(f'The output shape: {results.shape}')

        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(self.data['y_test_out'][:, 0, 0], label='ground_truth_Q')
        plt.plot(results[:, 0, 0], label='predict_Q')
        plt.legend()

        fig.add_subplot(122)
        plt.plot(self.data['y_test_out'][:, 0, 1], label='ground_truth_H')
        plt.plot(results[:, 0, 1], label='predict_H')
        plt.legend()

        plt.savefig(self.log_dir + 'predict.png')
        #plt.show()
        #print(results[:,1,0])
        return results

    def retransform_prediction(self):
        result = self.predict_and_plot()

        mask = np.zeros(self.data['shape'])
        test_shape = self.data['y_test_out'].shape[0]

        mask[-test_shape:, self.cols_y] = self.data['y_test_out'][:, 0, :]
        actual_data = self.data['scaler'].inverse_transform(mask)[-test_shape:, self.cols_y]

        mask[-test_shape:, self.cols_y] = result[:, 0, :]
        actual_predict = self.data['scaler'].inverse_transform(mask)[-test_shape:, self.cols_y]

        return actual_data, actual_predict

    def evaluate_model(self):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # actual_dat = self.data['y_test_out']
        # actual_pre = self.predict_and_plot()
        actual_dat, actual_pre = self.retransform_prediction()

        variance_score_q = r2_score(actual_dat[:, 0], actual_pre[:, 0])
        mse_q = mean_squared_error(actual_dat[:, 0], actual_pre[:, 0])
        mae_q = mean_absolute_error(actual_dat[:, 0], actual_pre[:, 0])

        variance_score_h = r2_score(actual_dat[:, 1], actual_pre[:, 1])
        mse_h = mean_squared_error(actual_dat[:, 1], actual_pre[:, 1])
        mae_h = mean_absolute_error(actual_dat[:, 1], actual_pre[:, 1])

        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(actual_dat[:, 0], label='actual_ground_truth_Q')
        plt.plot(actual_pre[:, 0], label='actual_predict_Q')
        plt.legend()

        fig.add_subplot(122)
        plt.plot(actual_dat[:, 1], label='ground_truth_H')
        plt.plot(actual_pre[:, 1], label='predict_H')
        plt.legend()

        plt.savefig(self.log_dir + 'predict_actual.png')

        with open(self.log_dir + 'evaluate_score.txt', 'a') as f:
            f.write(
                f'Model: H: R2: {variance_score_h} MSE: {mse_h} MAE: {mae_h} \nQ: R2: {variance_score_q} MSE: {mse_q} MAE: {mae_q} \n\n'
            )
            # Model: H: R2: {variance_score_h} MSE: {mse_h} MAE: {mae_h} \n


if __name__ == '__main__':
    import sys
    import os
    import argparse
    import yaml

    import keras.backend as K

    K.clear_session()

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help='Run mode.')
    parser.add_argument('--model', default='rnn_cnn', type=str, help='Model used.')
    args = parser.parse_args()

    np.random.seed(69)

    with open('./Config/Ensemble/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.mode == 'train' or args.mode == 'train-inner' or args.mode == 'train-outer':
        model = Ensemble(args.mode, args.model, **config)
        model.train_model_outer()
        model.evaluate_model()
        #simple_rnn.retransform_prediction()
    elif args.mode == "test":
        model = Ensemble(args.mode, args.model, **config)
        model.train_model_outer()
        model.evaluate_model()
        #simple_rnn.retransform_prediction()
    else:
        raise RuntimeError('Mode must be train or test!')