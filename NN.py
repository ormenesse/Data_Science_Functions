from time import time
import keras.backend as K
from keras import metrics
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Concatenate, Merge, Flatten, Reshape


##https://research.ijcaonline.org/volume32/number10/pxc3875530.pdf


input_dim = nn_base[vars_modelo_nn].shape[1]
#####################################################################
# NN normal Model
first = Sequential()
#
first.add(Dense(512,input_shape=(input_dim,), init='glorot_normal'))
first.add(BatchNormalization())
first.add(Activation('elu'))
first.add(Dropout(0.3))
#
first.add(Dense(256,name='firstCamada2',init='glorot_normal'))
first.add(BatchNormalization())
first.add(Activation('elu'))
first.add(Dropout(0.3))
#
first.add(Dense(128,name='firstCamada3',init='glorot_normal'))
first.add(BatchNormalization())
first.add(Activation('elu'))
first.add(Dropout(0.3))
#
first.add(Dense(64,name='firstCamada4',init='glorot_normal'))
first.add(BatchNormalization())
first.add(Activation('relu'))
first.add(Dropout(0.3))
#


#Convolution Model
second = Sequential()
second.add(Reshape((input_dim,1), input_shape=(input_dim,)))
second.add(Conv1D(4, kernel_size = 4, padding = 'valid',
          dilation_rate = 3, name='secondconv1'))
second.add(MaxPooling1D(name='Max1'))
second.add(Conv1D(4, kernel_size = 4, padding = 'valid',
            dilation_rate = 3, name='secondconv2'))
second.add(MaxPooling1D(name='Max2'))
#
second.add(Dense(128,name='secondCamada3',init='glorot_normal'))
second.add(BatchNormalization())
second.add(Activation('elu'))
second.add(Dropout(0.3))
#
second.add(Dense(64,name='secondCamada4',init='glorot_normal'))
second.add(BatchNormalization())
second.add(Activation('relu'))
second.add(Dropout(0.3))
#
second.add(Flatten())


junta = Merge([first,second],mode='concat')


merged = Sequential()
merged.add(junta)
#merged.add(Concatenate([first,second]))
merged.add(Dense(128,name='mergedCamada1',init='glorot_normal'))
merged.add(BatchNormalization())
merged.add(Activation('relu'))
merged.add(Dropout(0.3))
#
merged.add(Dense(64,name='mergedCamada2',init='glorot_normal'))
merged.add(BatchNormalization())
merged.add(Activation('relu'))
merged.add(Dropout(0.3))
#
merged.add(Dense(16,name='mergedCamada3',init='glorot_normal'))
merged.add(BatchNormalization())
merged.add(Activation('relu'))
merged.add(Dropout(0.3))
#Binary Model
merged.add(Dense(2, activation='softmax'))
#####################################################################
#Summary
merged.summary()


checkpoint = callbacks.ModelCheckpoint('data/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 
merged.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy',metrics.binary_crossentropy])
#tamanho do meu teste e validação
index_test = df[(df['num_mes_ref'] >= 201806) & (df['contratacao'] != 'sintetico')].index
index_train = df.index.difference(index_test)
len(index_test),len(index_train),len(index_test)+len(index_train),df.shape
#fit
merged.fit([nn_base[vars_modelo_nn].loc[index_train].values,nn_base[vars_modelo_nn].loc[index_train].values], 
           nn_base[['target_0','target']].loc[index_train].values, 
              epochs=40,
              batch_size=2048,
              shuffle=True,
              callbacks=[checkpoint],
              validation_data=([nn_base[vars_modelo_nn].loc[index_test].values,nn_base[vars_modelo_nn].loc[index_test].values], 
                               nn_base[['target_0','target']].loc[index_test].values)
             )
nn_predictions = merged.predict_proba([nn_base[vars_modelo_nn].values,nn_base[vars_modelo_nn].values])


#
#
#
#fazendo callbacks
from keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0] 
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]




    def on_train_begin(self, logs={}):
        return


    def on_train_end(self, logs={}):
        return


    def on_epoch_begin(self, epoch, logs={}):
        return


    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return


    def on_batch_begin(self, batch, logs={}):
        return


    def on_batch_end(self, batch, logs={}):
        return


riskmodel.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy',metrics.binary_crossentropy])
riskmodel.fit(nn_base[vars_modelo_nn].loc[index_train].values, nn_base[['target_0','target']].loc[index_train].values, 
              epochs=100,
              batch_size=2048,
              shuffle=True,
              validation_data=(nn_base[vars_modelo_nn].loc[index_test].values, nn_base[['target_0','target']].loc[index_test].values)
              #,callbacks=[roc_callback(training_data=(nn_base[vars_modelo_nn].loc[index_train].values, nn_base[['target_0','target']].loc[index_train].values),
              #           validation_data=(nn_base[vars_modelo_nn].loc[index_test].values, nn_base[['target_0','target']].loc[index_test].values))]
             )
