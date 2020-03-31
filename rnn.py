from keras.layers import Dense,LSTM,Bidirectional,Input,Conv1D
from keras import Model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint 
from ProcessingData.reprocess_daily import extract_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import yaml


class simpleRNN:
    def __init__(self,mode, **kwargs):
        self.mode = mode

        self.log_dir = kwargs.get('log_dir')
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')

        self.data_file = self._data_kwargs.get('data_file')
        self.dt_split_point = self._data_kwargs.get('split_point')
        self.cols_x = self._data_kwargs.get('cols_x')
        self.norm_method = self._data_kwargs.get('norm_method')

        self.window_size = self._model_kwargs.get('window_size')
        self.batch_size = self._model_kwargs.get('batch_size')
        self.epochs = self._model_kwargs.get('epochs')
        self.input_dim = self._model_kwargs.get('input_dim')
        self.patience = self._model_kwargs.get('patience')
        
        self.data = self.generate_data()
        self.model = self.build_model()

        


    def generate_data(self):
        dat = pd.read_csv(self.data_file)
        xs, ys = extract_data(dataframe=dat,window_size=self.window_size,cols=self.cols_x,mode=self.norm_method)
        x_train, x_val, y_train, y_val = train_test_split(xs, ys, test_size=self.dt_split_point, random_state=1)
        x_train, x_test, y_train, y_test  = train_test_split(x_train, y_train, test_size=self.dt_split_point, random_state=1)
        return x_train, y_train,x_val,y_val,x_test,y_test

    def build_model(self):
        input = Input(shape=(None,self.input_dim))
        conv1d = Conv1D(filters=64,kernel_size=6,strides=1,padding='valid')(input)
        rnn1 = Bidirectional(LSTM(units=64,return_sequences=True,activation='relu'))(input)
        rnn2 = LSTM(units=64,return_sequences=True,activation='relu')(rnn1)
        rnn3 = LSTM(units=64,return_sequences=True,activation='relu')(rnn2)
        dense1 = Dense(units=64,activation='relu')(rnn2)
        dense2 = Dense(units=32,activation='relu')(dense1)
        #dense3 = Dense(units=16,activation='relu')(dense2)
        output = Dense(units=2)(dense2)

        model = Model(inputs=input,outputs=output)

        #optimizer = SGD(lr=1e-6, momentum=0.9)
        optimizer = RMSprop(learning_rate=1e-5)
        model.compile(loss= 'mse',
                    optimizer='adam',
                    metrics=['accuracy'])
        
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
            
            history = self.model.fit(x=self.data[0],y=self.data[1],
                                    batch_size=self.batch_size,epochs=self.epochs,
                                    callbacks=callbacks,validation_data=(self.data[2],self.data[3]))
            
            if history is not None:
                self.plot_training_history(history)
        elif self.mode == 'test':
            self.model.load_weights(self.log_dir + 'best_model.hdf5')
            print('Load weight from ' + self.log_dir)
    
    def plot_training_history(self,history):
        fig = plt.figure(figsize=(10, 6))
        #fig.add_subplot(121)
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_loss'],label='val_loss')
        plt.legend()
        
        #fig.add_subplot(122)
        #plt.semilogx(history.history["lr"], history.history["loss"])

        plt.savefig(self.log_dir + 'training_phase.png')
        plt.show()

    def predict_and_plot(self):
        results = self.model.predict(x=self.data[4])

        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.data[5][:,0,0],self.data[5][:,0,1],'or',label='ground_truth')
        plt.plot(results[:,0,0],results[:,0,1],'ob',label='predict')
        plt.legend()
        plt.savefig(self.log_dir + 'predict.png')
        plt.show()
        return results

    def evaluate_model(self):
        #score = self.model.evaluate(x=self.data[4], y=self.data[5],verbose=1)
        from sklearn.metrics import mean_squared_error,mean_absolute_error, explained_variance_score
        result = self.predict_and_plot()
        
        variance_score_h = explained_variance_score(self.data[5][:,0,0],result[:,0,0])
        mse_h = mean_squared_error(self.data[5][:,0,0],result[:,0,0])
        mae_h = mean_absolute_error(self.data[5][:,0,0],result[:,0,0])

        variance_score_q = explained_variance_score(self.data[5][:,0,1],result[:,0,1])
        mse_q = mean_squared_error(self.data[5][:,0,1],result[:,0,1])
        mae_q = mean_absolute_error(self.data[5][:,0,1],result[:,0,1])
        with open(self.log_dir + 'evaluate_score.txt', 'a') as f:
            f.write(f'Model: H: R2: {variance_score_h} MSE: {mse_h} MAE: {mae_h} \nQ: R2: {variance_score_q} MSE: {mse_q} MAE: {mae_q} \n\n')

        

if __name__ == '__main__':
    import sys
    import os
    import argparse

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    args = parser.parse_args()


    with open('./Config/SimpleRNN/params.yaml','r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    if args.mode == 'train':
        simple_rnn = simpleRNN(args.mode,**config)
        simple_rnn.train_model()
        simple_rnn.evaluate_model()
    elif args.mode == "test":
        simple_rnn = simpleRNN(args.mode,**config)
        simple_rnn.train_model()
        simple_rnn.evaluate_model()
    else:
        raise RuntimeError('Mode must be train or test!')