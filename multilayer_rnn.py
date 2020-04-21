from keras.layers import Dense,LSTM,Bidirectional,Input,Concatenate,Conv1D,TimeDistributed, Reshape
from keras import Model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint 
from ProcessingData.reprocess_daily import extract_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import yaml


class MultiRNN:
    def __init__(self,mode, **kwargs):
        self.mode = mode

        self.log_dir = kwargs.get('log_dir')
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')

        self.data_file = self._data_kwargs.get('data_file')
        self.dt_split_point = self._data_kwargs.get('split_point')
        self.cols_x = self._data_kwargs.get('cols_x')
        self.cols_y = self._data_kwargs.get('cols_y')
        self.target_timestep = self._data_kwargs.get('target_timestep')
        self.norm_method = self._data_kwargs.get('norm_method')

        self.window_size = self._model_kwargs.get('window_size')
        self.batch_size = self._model_kwargs.get('batch_size')
        self.epochs = self._model_kwargs.get('epochs')
        self.input_dim = self._model_kwargs.get('input_dim')
        self.output_dim = self._model_kwargs.get('output_dim')
        self.patience = self._model_kwargs.get('patience')
        self.dropout = self._model_kwargs.get('dropout')
        self.lr_decay = self._model_kwargs.get('lr_decay')
        
        self.data = self.generate_data()
        self.model = self.build_model()

        


    def generate_data(self):
        dat = pd.read_csv(self.data_file,header=0,index_col=0)
        dat = dat.drop('time',axis=1)
        dat = dat.to_numpy()

        data = {}
        data['shape'] = dat.shape

        train_size = int(dat.shape[0] * (1 - self.dt_split_point))
        test_size = int(train_size * self.dt_split_point / 2)

        x, y, scaler = extract_data(dataframe=dat,window_size=self.window_size,
                                                    target_timstep=self.target_timestep,
                                                    cols_x=self.cols_x,cols_y=self.cols_y,
                                                    mode=self.norm_method)

        x_train, y_train = x[:train_size,:], y[:train_size,:]
        x_val, y_val = x[train_size:-test_size,:], y[train_size:-test_size,:]
        x_test, y_test = x[-test_size:,:], y[-test_size:,:]
        
        for cat in ["train", "val", "test"]:
            x, y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", x.shape, "y: ", y.shape)
            data["x_" + cat] = x
            data["y_" + cat] = y
        
        data['scaler'] = scaler
        return data

    def build_model(self):
        input = Input(shape=(None,self.input_dim))

        conv = Conv1D(filters=16,kernel_size=2,strides=1,padding='same')
        conv_out = conv(input)
        conv_2 = Conv1D(filters=32,kernel_size=3,padding='same')
        conv_out_2 = conv_2(conv_out)
        conv_3 = Conv1D(filters=64,kernel_size=4,padding='same')
        conv_out_3 = conv_3(conv_out_2)

        rnn_1 = Bidirectional(LSTM(units=128,return_sequences=True,return_state=True))
        rnn_out_1, forward_h, forward_c, backward_h, backward_c = rnn_1(conv_out_3)
        state_h = Concatenate(axis=-1)([forward_h,backward_h])
        state_c = Concatenate(axis=-1)([forward_c,backward_c])


        rnn_2 = Bidirectional(LSTM(units=128,return_sequences=False))
        rnn_out_2 = rnn_2(rnn_out_1,initial_state=[forward_h,forward_c,backward_h,backward_c])

        # rnn_3 = LSTM(units=256,return_sequences=True,return_state=True)
        # rnn_out_3, state_h_3, state_c_3 = rnn_3(rnn_out_2)

        # rnn_4_in = Concatenate(axis=-1)([rnn_out_2,rnn_out_3])
        # rnn_4 = LSTM(units=256,return_sequences=False)
        # rnn_out_4 = rnn_4(rnn_4_in)
        
        reshape_l = Reshape(target_shape=(self.target_timestep,-1))
        rnn_out = reshape_l(rnn_out_2)


        # dense_1 = TimeDistributed(Dense(units=64,activation='relu'))
        # dense_1_out = dense_1(rnn_4_out)

        # dense_2 = TimeDistributed(Dense(units=32,activation='relu'))
        # dense_2_out = dense_2(dense_1_out)

        dense_3 = TimeDistributed(Dense(units=self.output_dim))
        output = dense_3(rnn_out)

        model = Model(inputs=input,outputs=output)

        #optimizer = SGD(lr=1e-6, momentum=0.9,decay=self.lr_decay,nesterov=True)
        #optimizer = RMSprop(learning_rate=5e-3)
        #optimizer = Adadelta(rho=0.95)
        #optimizer = Adam(learning_rate=5e-2,amsgrad=False)
        model.compile(loss= 'mse',
                    optimizer='adam',
                    metrics=['mae','mape'])
        
        return model

    def train_model(self):
        if self.mode == 'train':
            callbacks = []
            #lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
            early_stop = EarlyStopping(monitor='val_loss',patience=self.patience,restore_best_weights=True)
            checkpoint = ModelCheckpoint(self.log_dir + 'best_model.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
            
            #callbacks.append(lr_schedule)
            callbacks.append(early_stop)
            callbacks.append(checkpoint)
            
            history = self.model.fit(x=self.data['x_train'],
                                    y=self.data['y_train'],
                                    batch_size=self.batch_size,epochs=self.epochs,
                                    callbacks=callbacks,
                                    validation_data=(self.data['x_val'], self.data['y_val']))
            
            if history is not None:
                self.plot_training_history(history)
        elif self.mode == 'test':
            self.model.load_weights(self.log_dir + 'best_model.hdf5')
            print('Load weight from ' + self.log_dir)

        from keras.utils.vis_utils import plot_model
        import os
        os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
        plot_model(model=self.model, to_file=self.log_dir + '/model_dup.png', show_shapes=True)
    
    def plot_training_history(self,history):
        fig = plt.figure(figsize=(10, 6))
        #fig.add_subplot(121)
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_loss'],label='val_loss')
        plt.plot(history.history['mae'],label='mae')
        plt.legend()
        
        #fig.add_subplot(122)
        #plt.semilogx(history.history["lr"], history.history["loss"])

        plt.savefig(self.log_dir + 'training_phase.png')
        #plt.show()

    def predict_and_plot(self):
        results = self.model.predict(x=self.data['x_test'],batch_size=self.batch_size)
        print(f'The output shape: {results.shape}')

        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(self.data['y_test'][:,0,0],label='ground_truth_Q')
        plt.plot(results[:,0,0],label='predict_Q')
        plt.legend()

        fig.add_subplot(122)
        plt.plot(self.data['y_test'][:,0,1],label='ground_truth_H')
        plt.plot(results[:,0,1],label='predict_H')
        plt.legend()

        plt.savefig(self.log_dir + 'predict.png')
        #plt.show()
        #print(results[:,1,0])
        return results

    def retransform_prediction(self):
        result = self.predict_and_plot()

        mask = np.zeros(self.data['shape'])
        test_shape = self.data['y_test'].shape[0]
        
        mask[-test_shape:,self.cols_y] = self.data['y_test'][:,0,:]
        actual_data = self.data['scaler'].inverse_transform(mask)[-test_shape:,self.cols_y]

        mask[-test_shape:,self.cols_y] = result[:,0,:]
        actual_predict = self.data['scaler'].inverse_transform(mask)[-test_shape:,self.cols_y]

        return actual_data, actual_predict
        
    def evaluate_model(self):
        #score = self.model.evaluate(x=self.data[4], y=self.data[5],verbose=1)
        from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
        actual_dat,actual_pre = self.retransform_prediction()
        
        variance_score_q = r2_score(actual_dat[:,0],actual_pre[:,0])
        mse_q = mean_squared_error(actual_dat[:,0],actual_pre[:,0])
        mae_q = mean_absolute_error(actual_dat[:,0],actual_pre[:,0])

        variance_score_h = r2_score(actual_dat[:,1],actual_pre[:,1])
        mse_h = mean_squared_error(actual_dat[:,1],actual_pre[:,1])
        mae_h = mean_absolute_error(actual_dat[:,1],actual_pre[:,1])

        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(actual_dat[:,0],label='actual_ground_truth_Q')
        plt.plot(actual_pre[:,0],label='actual_predict_Q')
        plt.legend()

        fig.add_subplot(122)
        plt.plot(actual_dat[:,1],label='ground_truth_H')
        plt.plot(actual_pre[:,1],label='predict_H')
        plt.legend()

        plt.savefig(self.log_dir + 'predict_actual.png')
        #plt.show()

        with open(self.log_dir + 'evaluate_score.txt', 'a') as f:
            f.write(f'Model: H: R2: {variance_score_h} MSE: {mse_h} MAE: {mae_h} \nQ: R2: {variance_score_q} MSE: {mse_q} MAE: {mae_q} \n\n')

if __name__ == '__main__':
    import sys
    import os
    import argparse

    import keras.backend as K 

    K.clear_session()

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    np.random.seed(69)

    with open('./Config/MultilayerRNN/config.yaml','r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    if args.mode == 'train':
        model = MultiRNN(args.mode,**config)
        model.train_model()
        model.evaluate_model()
        #simple_rnn.retransform_prediction()
    elif args.mode == "test":
        model = MultiRNN(args.mode,**config)
        model.train_model()
        model.evaluate_model()
        #simple_rnn.retransform_prediction()
    else:
        raise RuntimeError('Mode must be train or test!')