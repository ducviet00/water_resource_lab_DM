from keras.layers import Dense,SimpleRNN,Bidirectional,Input
from keras import Model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint 
from ProcessingData.process import split_window_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import yaml


class simpleRNN:
    def __init__(self, **kwargs):
        self.log_dir = kwargs.get('log_dir')
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')

        self.data_file = self._data_kwargs.get('data_file')
        self.dt_split_point = self._data_kwargs.get('split_point')

        self.window_size = self._model_kwargs.get('window_size')
        self.batch_size = self._model_kwargs.get('batch_size')
        self.epochs = self._model_kwargs.get('epochs')
        self.input_dim = self._model_kwargs.get('input_dim')
        self.patience = self._model_kwargs.get('patience')
        
        self.data = self.generate_data()
        self.model = self.build_model()

        


    def generate_data(self):
        dat = pd.read_csv(self.data_file)
        dat = dat.drop('Time',axis=1)
        xs, ys = split_window_data(dtframe=dat,window_size=self.window_size,cols=['H','Q'])
        x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=self.dt_split_point, random_state=1)
        x_train, x_val, y_train, y_val  = train_test_split(x_train, y_train, test_size=self.dt_split_point, random_state=1)
        return x_train, y_train,x_val,y_val,x_test,y_test

    def build_model(self):
        input = Input(shape=(None,self.input_dim))
        rnn1 = SimpleRNN(units=32,return_sequences=True,activation='relu')(input)
        rnn2 = SimpleRNN(units=32,activation='relu')(rnn1)
        output = Dense(units=2,)(rnn2)

        model = Model(inputs=input,outputs=output)

        #optimizer = SGD(lr=1e-7, momentum=0.9)
        optimizer = RMSprop(learning_rate=1e-7)
        model.compile(loss= 'mse',
                    optimizer=optimizer,
                    metrics=['accuracy'])
        
        return model

    def train_model(self):
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
    
    def plot_training_history(self,history):
        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(121)
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_loss'],label='val_loss')
        plt.legend()
        
        fig.add_subplot(122)
        plt.semilogx(history.history["lr"], history.history["loss"])

        plt.savefig(self.log_dir + 'training_phase.png')
        plt.show()

    def predict_and_plot(self):
        results = self.model.predict(x=self.data[4])
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.data[5][:,0],self.data[5][:,1],'or',label='ground_truth')
        plt.plot(results[:,0],results[:,1],'ob',label='predict')
        plt.legend()
        plt.savefig(self.log_dir + 'predict.png')
        plt.show()


if __name__ == '__main__':
    with open('./Config/SimpleRNN/params.yaml','r') as f:
        config = yaml.load(f)

    simple_rnn = simpleRNN(**config)
    simple_rnn.train_model()
    simple_rnn.predict_and_plot()