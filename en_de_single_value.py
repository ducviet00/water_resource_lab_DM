from keras.layers import Dense,LSTM,Bidirectional,Input,Concatenate,Conv1D, TimeDistributed
from keras import Model
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint 
from ProcessingData.reprocess_daily import ed_extract_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import yaml


class EncoderDecoder:
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
        self.target_name = self._data_kwargs.get('target_name')

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

        en_x, de_x, de_y, scaler = ed_extract_data(dataframe=dat,window_size=self.window_size,
                                                    target_timstep=self.target_timestep,
                                                    cols_x=self.cols_x,cols_y=self.cols_y,
                                                    mode=self.norm_method)

        en_x_train, de_x_train, de_y_train = en_x[:train_size,:], de_x[:train_size,:],de_y[:train_size,:]
        en_x_val, de_x_val, de_y_val = en_x[train_size:-test_size,:], de_x[train_size:-test_size,:],de_y[train_size:-test_size,:]
        en_x_test, de_x_test, de_y_test = en_x[-test_size:,:], de_x[-test_size:,:],de_y[-test_size:,:]
        
        for cat in ["train", "val", "test"]:
            e_x, d_x, d_y = locals()["en_x_" + cat], locals()[
                "de_x_" + cat], locals()["de_y_" + cat]
            print(cat, "e_x: ", e_x.shape, "d_x: ", d_x.shape, "d_y: ", d_y.shape)
            data["en_x_" + cat] = e_x
            data["de_x_" + cat] = d_x
            data["de_y_" + cat] = d_y
        
        data['scaler'] = scaler
        return data

    def build_model(self):
        encoder_inputs = Input(shape=(None, self.input_dim))
        conv1d = Conv1D(filters=16,kernel_size=2,strides=1,padding='same')
        conv_out = conv1d(encoder_inputs)
        #conv1d_2 = Conv1D(filters=8,kernel_size=2,strides=1,padding='valid',activation='sigmoid')(conv1d)
        encoder = Bidirectional(LSTM(256, return_state=True, dropout=self.dropout))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(conv_out)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # decoder
        decoder_inputs = Input(shape=(None, self.output_dim))    
        #de_conv1d = Conv1D(filters=8,kernel_size=2,strides=1,padding='valid',activation='sigmoid')(decoder_inputs)
        #de_conv1d_2 = Conv1D(filters=8,kernel_size=2,strides=1,padding='valid',activation='sigmoid')(de_conv1d)
        decoder_lstm_1 = LSTM(512, return_sequences=True, return_state=False)
        decoder_outputs_1 = decoder_lstm_1(decoder_inputs, initial_state=encoder_states)
        #decoder_lstm_2 = LSTM(256, return_sequences=True, return_state=False)
        #decoder_outputs_2 = decoder_lstm_2(decoder_outputs_1)
        decoder_dense_1 = Dense(units=32,activation='relu')(decoder_outputs_1)
        #decoder_dense_2 = Dense(units=32,activation='relu')(decoder_dense_1)
        decoder_dense = TimeDistributed(Dense(units=self.output_dim))
        decoder_outputs = decoder_dense(decoder_dense_1)

        model = Model(inputs=[encoder_inputs,decoder_inputs],outputs=decoder_outputs)

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
            
            history = self.model.fit(x=[self.data['en_x_train'],self.data['de_x_train']],
                                    y=self.data['de_y_train'],
                                    batch_size=self.batch_size,epochs=self.epochs,
                                    callbacks=callbacks,
                                    validation_data=([self.data['en_x_val'],self.data['de_x_val']],self.data['de_y_val']))
            
            if history is not None:
                self.plot_training_history(history)
        elif self.mode == 'test':
            self.model.load_weights(self.log_dir + 'best_model.hdf5')
            print('Load weight from ' + self.log_dir)

        from keras.utils.vis_utils import plot_model
        import os
        os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
        plot_model(model=self.model, to_file=self.log_dir + '/model.png', show_shapes=True)
    
    def plot_training_history(self,history):
        fig = plt.figure(figsize=(10, 6))
        #fig.add_subplot(121)
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_loss'],label='val_loss')
        plt.legend()
        
        #fig.add_subplot(122)
        #plt.semilogx(history.history["lr"], history.history["loss"])

        plt.savefig(self.log_dir + 'training_phase.png')
        #plt.show()

    def predict_and_plot(self):
        results = self.model.predict(x=[self.data['en_x_test'],self.data['de_x_test']],batch_size=self.batch_size)
        print(f'The output shape: {results.shape}')

        # for i in range(self.target_timestep):
        #     fig = plt.figure(figsize=(10, 6))
        #     plt.plot(self.data['de_y_test'][:,i,0],label=f'ground_truth_{self.target_name}')
        #     plt.plot(results[:,i,0],label=f'predict_{self.target_name}')
        #     plt.legend()

        #     plt.savefig(self.log_dir + f'predict_{self.target_name}_{i+1}.png')
        #plt.show()
        #print(results[:,1,0])
        return results

    def retransform_prediction(self,result,index):

        mask = np.zeros(self.data['shape'])
        test_shape = self.data['de_y_test'].shape[0]
        
        mask[-test_shape:,self.cols_y] = self.data['de_y_test'][:,index,0].reshape(-1,1)
        actual_data = self.data['scaler'].inverse_transform(mask)[-test_shape:,self.cols_y]

        mask[-test_shape:,self.cols_y] = result[:,index,0].reshape(-1,1)
        actual_predict = self.data['scaler'].inverse_transform(mask)[-test_shape:,self.cols_y]

        return actual_data, actual_predict
        
    def evaluate_model(self):
        #score = self.model.evaluate(x=self.data[4], y=self.data[5],verbose=1)
        from sklearn.metrics import mean_squared_error,mean_absolute_error, explained_variance_score, r2_score
        
        result = self.predict_and_plot()
        for i in range(self.target_timestep):
            actual_dat,actual_pre = self.retransform_prediction(result,i)
            #print(actual_dat.shape)
            
            r2 = r2_score(actual_dat[:,0],actual_pre[:,0])
            variance_score = explained_variance_score(actual_dat[:,0],actual_pre[:,0])
            mse = mean_squared_error(actual_dat[:,0],actual_pre[:,0])
            mae = mean_absolute_error(actual_dat[:,0],actual_pre[:,0])

            fig = plt.figure(figsize=(10, 6))
            plt.plot(actual_dat[:,0],label='actual_ground_truth')
            plt.plot(actual_pre[:,0],label='actual_predict')
            plt.legend()

            plt.savefig(self.log_dir + f'predict_actual_{self.target_name}_{i+1}.png')
            #plt.show()

            with open(self.log_dir + 'evaluate_score.txt', 'a') as f:
                f.write(f'Model: {i+1} R2: {r2} Variance: {variance_score} MSE: {mse} MAE: {mae} \n\n')

if __name__ == '__main__':
    import sys
    import os
    import argparse

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    np.random.seed(911)

    with open('./Config/EncoderDecoder/SingleValue/config.yaml','r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    if args.mode == 'train':
        ed_model = EncoderDecoder(args.mode,**config)
        ed_model.train_model()
        ed_model.evaluate_model()
        #simple_rnn.retransform_prediction()
    elif args.mode == "test":
        ed_model = EncoderDecoder(args.mode,**config)
        ed_model.train_model()
        ed_model.evaluate_model()
        #simple_rnn.retransform_prediction()
    else:
        raise RuntimeError('Mode must be train or test!')