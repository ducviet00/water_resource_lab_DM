from keras.layers import Conv1D, Input,Bidirectional,LSTM,Concatenate, Reshape,TimeDistributed,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint

def model_builder(input_dim=5,output_dim=2,target_timestep=1):
    input = Input(shape=(None,input_dim))

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


    # rnn_2 = Bidirectional(LSTM(units=128,return_sequences=False))
    # rnn_out_2 = rnn_2(rnn_out_1,initial_state=[forward_h,forward_c,backward_h,backward_c])

    rnn_3 = LSTM(units=256,return_sequences=False,return_state=False)
    rnn_out_3= rnn_3(rnn_out_1,initial_state=[state_h,state_c])

    # rnn_4_in = Concatenate(axis=-1)([rnn_out_2,rnn_out_3])
    # rnn_4 = LSTM(units=256,return_sequences=False)
    # rnn_out_4 = rnn_4(rnn_4_in)
    
    reshape_l = Reshape(target_shape=(target_timestep,-1))
    rnn_out = reshape_l(rnn_out_3)


    # dense_1 = TimeDistributed(Dense(units=64,activation='relu'))
    # dense_1_out = dense_1(rnn_4_out)

    # dense_2 = TimeDistributed(Dense(units=32,activation='relu'))
    # dense_2_out = dense_2(dense_1_out)

    dense_3 = TimeDistributed(Dense(units=output_dim))
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

def train_model(model,x_train,y_train,batch_size,epochs,fraction=0.1,patience=0,early_stop=False, save_dir=''):
    callbacks = []

    checkpoint = ModelCheckpoint(save_dir + f'best_model_{epochs}.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
    callbacks.append(checkpoint)
    
    #early_stop = epochs == 250
    if (early_stop):
        early_stop = EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True)
        callbacks.append(early_stop)
    
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,epochs=epochs,
                        callbacks=callbacks,
                        validation_split=fraction)

    return model, history