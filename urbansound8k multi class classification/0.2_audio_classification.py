import librosa

file_name = "dataset\\fold1\\7383-3-1-0.wav"

librosa_audio_data,librosa_sample_rate=librosa.load(file_name)

librosa_audio_data
"""
array([ 1.4551915e-10, -8.7311491e-11, -1.1641532e-10, ...,
        3.6435030e-04,  2.6052771e-04, -2.1291785e-04], dtype=float32) """

librosa_sample_rate # 22050

## feature extraction for 1 audio file
### using Mel-Frequency Cepstral Coefficients (MFCC), https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)

print(mfccs.shape) # (40, 173)

mfccs
"""
array([[-5.8003693e+02, -4.9177695e+02, -3.6617474e+02, ...,
        -5.0730090e+02, -5.1229175e+02, -5.2267572e+02],
       [ 3.0879446e+01,  1.1654912e+02,  1.7551926e+02, ...,
         9.0842987e+01,  9.2580032e+01,  8.8494919e+01],
       [ 1.7225260e+01,  3.9759499e+01, -5.0101824e+00, ...,
         2.7333740e+01,  2.7949635e+01,  3.1582390e+01],
       ...,
       [-3.7395456e+00, -4.9923801e+00,  4.4441934e+00, ...,
         2.7369065e+00,  1.6080571e+00,  2.7038860e+00],
       [-1.9384034e+00, -4.6505088e-01,  6.2187800e+00, ...,
         2.7966838e+00,  2.2690997e+00,  9.9648261e-01],
       [ 1.7400644e+00,  2.0404806e+00,  4.3179607e+00, ...,
         1.5787597e+00,  1.0261321e+00, -3.2630148e+00]], dtype=float32) """

import pandas as pd

## feature extraction for entire dataset

audio_dataset_path= "C:/Users/akshcp/OneDrive - Capgemini/data science/GDSC/tut/multi class classification/dataset/audio/"
metadata = pd.read_csv("dataset/UrbanSound8K.csv")

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

import numpy as np
from tqdm import tqdm
import os

### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),str(row["slice_file_name"]))
    # file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

# open file in write mode
with open(r'inpendent_dependent_features.txt', 'w') as fp:
    for item in extracted_features:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()
"""
                                             feature             class
0  [-217.35526, 70.22338, -130.38527, -53.282898,...          dog_bark
1  [-424.09818, 109.34077, -52.919525, 60.86475, ...  children_playing
2  [-458.79114, 121.38419, -46.52066, 52.00812, -...  children_playing
3  [-413.89984, 101.66373, -35.42945, 53.036354, ...  children_playing
4  [-446.60352, 113.68541, -52.402206, 60.302044,...  children_playing """

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

X.shape # (8732, 40)

### Label Encoding
y=np.array(pd.get_dummies(y))

"""
OR

 from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y)) """

y.shape # (8732, 10)

### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

X_train
"""
array([[-1.31104706e+02,  1.12505905e+02, -2.25746956e+01, ...,
         3.24665213e+00, -1.36902380e+00,  2.75575471e+00],
       [-1.36703424e+01,  9.10850830e+01, -7.79273319e+00, ...,
        -3.25305033e+00, -5.27745247e+00, -1.55697155e+00],
       [-4.98715439e+01,  2.65352815e-01, -2.05009365e+01, ...,
         2.85459447e+00, -1.60920465e+00,  3.52480578e+00],
       ...,
       [-4.27012360e+02,  9.26230469e+01,  3.12939739e+00, ...,
         7.42641270e-01,  7.33490884e-01,  7.11009026e-01],
       [-1.45754608e+02,  1.36265778e+02, -3.35155182e+01, ...,
         1.46811938e+00, -2.00916982e+00, -8.82181883e-01],
       [-4.21031342e+02,  2.10654541e+02,  3.49066067e+00, ...,
        -5.38886738e+00, -3.37136054e+00, -1.56651151e+00]], dtype=float32) """

y
"""
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 1, ..., 0, 0, 0],
       [0, 0, 1, ..., 0, 0, 0],
       ...,
       [0, 1, 0, ..., 0, 0, 0],
       [0, 1, 0, ..., 0, 0, 0],
       [0, 1, 0, ..., 0, 0, 0]], dtype=uint8) """


X_train.shape # (6985, 40)

X_test.shape # (1747, 40)

y_train.shape # (6985, 10)

y_test.shape # (1747, 10)


# Model Creation
import tensorflow as tf
print(tf.__version__) # 2.12.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
### No of classes
num_labels=y.shape[1]

# ANN implementation

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint

from datetime import datetime

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
"""
Epoch 1/100
212/219 [============================>.] - ETA: 0s - loss: 11.5990 - accuracy: 0.1193
Epoch 1: val_loss improved from inf to 2.29058, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 5s 10ms/step - loss: 11.3513 - accuracy: 0.1194 - val_loss: 2.2906 - val_accuracy: 0.1048
Epoch 2/100
213/219 [============================>.] - ETA: 0s - loss: 2.6432 - accuracy: 0.1269
Epoch 2: val_loss improved from 2.29058 to 2.27908, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 2.6415 - accuracy: 0.1268 - val_loss: 2.2791 - val_accuracy: 0.1168
Epoch 3/100
219/219 [==============================] - ETA: 0s - loss: 2.3489 - accuracy: 0.1350
Epoch 3: val_loss improved from 2.27908 to 2.22859, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 2.3489 - accuracy: 0.1350 - val_loss: 2.2286 - val_accuracy: 0.1357
Epoch 4/100
214/219 [============================>.] - ETA: 0s - loss: 2.2615 - accuracy: 0.1357
Epoch 4: val_loss improved from 2.22859 to 2.14534, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 7ms/step - loss: 2.2606 - accuracy: 0.1359 - val_loss: 2.1453 - val_accuracy: 0.1208
Epoch 5/100
211/219 [===========================>..] - ETA: 0s - loss: 2.2014 - accuracy: 0.1715
Epoch 5: val_loss improved from 2.14534 to 2.13198, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 2.1995 - accuracy: 0.1721 - val_loss: 2.1320 - val_accuracy: 0.1849
Epoch 6/100
217/219 [============================>.] - ETA: 0s - loss: 2.1697 - accuracy: 0.1815
Epoch 6: val_loss improved from 2.13198 to 2.10449, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 2.1704 - accuracy: 0.1810 - val_loss: 2.1045 - val_accuracy: 0.1935
Epoch 7/100
217/219 [============================>.] - ETA: 0s - loss: 2.1410 - accuracy: 0.1901
Epoch 7: val_loss improved from 2.10449 to 2.06623, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 2.1410 - accuracy: 0.1897 - val_loss: 2.0662 - val_accuracy: 0.2175
Epoch 8/100
211/219 [===========================>..] - ETA: 0s - loss: 2.0939 - accuracy: 0.2146
Epoch 8: val_loss improved from 2.06623 to 2.02914, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 7ms/step - loss: 2.0938 - accuracy: 0.2159 - val_loss: 2.0291 - val_accuracy: 0.2576
Epoch 9/100
215/219 [============================>.] - ETA: 0s - loss: 2.0298 - accuracy: 0.2475
Epoch 9: val_loss improved from 2.02914 to 1.88447, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 2.0284 - accuracy: 0.2485 - val_loss: 1.8845 - val_accuracy: 0.3537
Epoch 10/100
217/219 [============================>.] - ETA: 0s - loss: 1.9836 - accuracy: 0.2624
Epoch 10: val_loss improved from 1.88447 to 1.78481, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.9835 - accuracy: 0.2626 - val_loss: 1.7848 - val_accuracy: 0.3652
Epoch 11/100
213/219 [============================>.] - ETA: 0s - loss: 1.9211 - accuracy: 0.2861
Epoch 11: val_loss improved from 1.78481 to 1.77019, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 7ms/step - loss: 1.9188 - accuracy: 0.2872 - val_loss: 1.7702 - val_accuracy: 0.3904
Epoch 12/100
214/219 [============================>.] - ETA: 0s - loss: 1.8553 - accuracy: 0.3143
Epoch 12: val_loss improved from 1.77019 to 1.66901, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.8516 - accuracy: 0.3151 - val_loss: 1.6690 - val_accuracy: 0.4299
Epoch 13/100
213/219 [============================>.] - ETA: 0s - loss: 1.8208 - accuracy: 0.3433
Epoch 13: val_loss improved from 1.66901 to 1.63908, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.8215 - accuracy: 0.3424 - val_loss: 1.6391 - val_accuracy: 0.4465
Epoch 14/100
212/219 [============================>.] - ETA: 0s - loss: 1.7669 - accuracy: 0.3545
Epoch 14: val_loss improved from 1.63908 to 1.57363, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.7630 - accuracy: 0.3550 - val_loss: 1.5736 - val_accuracy: 0.4614
Epoch 15/100
217/219 [============================>.] - ETA: 0s - loss: 1.7076 - accuracy: 0.3822
Epoch 15: val_loss improved from 1.57363 to 1.53214, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 3s 13ms/step - loss: 1.7081 - accuracy: 0.3820 - val_loss: 1.5321 - val_accuracy: 0.4843
Epoch 16/100
219/219 [==============================] - ETA: 0s - loss: 1.6891 - accuracy: 0.3933
Epoch 16: val_loss improved from 1.53214 to 1.52187, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.6891 - accuracy: 0.3933 - val_loss: 1.5219 - val_accuracy: 0.4957
Epoch 17/100
217/219 [============================>.] - ETA: 0s - loss: 1.6388 - accuracy: 0.4109
Epoch 17: val_loss improved from 1.52187 to 1.45965, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 11ms/step - loss: 1.6385 - accuracy: 0.4109 - val_loss: 1.4597 - val_accuracy: 0.5163
Epoch 18/100
217/219 [============================>.] - ETA: 0s - loss: 1.6250 - accuracy: 0.4227
Epoch 18: val_loss improved from 1.45965 to 1.41948, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.6244 - accuracy: 0.4226 - val_loss: 1.4195 - val_accuracy: 0.5358
Epoch 19/100
210/219 [===========================>..] - ETA: 0s - loss: 1.5799 - accuracy: 0.4339
Epoch 19: val_loss improved from 1.41948 to 1.38856, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.5798 - accuracy: 0.4338 - val_loss: 1.3886 - val_accuracy: 0.5381
Epoch 20/100
216/219 [============================>.] - ETA: 0s - loss: 1.5517 - accuracy: 0.4661
Epoch 20: val_loss improved from 1.38856 to 1.38067, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 3s 14ms/step - loss: 1.5510 - accuracy: 0.4657 - val_loss: 1.3807 - val_accuracy: 0.5467
Epoch 21/100
215/219 [============================>.] - ETA: 0s - loss: 1.5328 - accuracy: 0.4568
Epoch 21: val_loss improved from 1.38067 to 1.36847, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 3s 14ms/step - loss: 1.5298 - accuracy: 0.4573 - val_loss: 1.3685 - val_accuracy: 0.5415
Epoch 22/100
215/219 [============================>.] - ETA: 0s - loss: 1.5194 - accuracy: 0.4722
Epoch 22: val_loss improved from 1.36847 to 1.33816, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.5188 - accuracy: 0.4724 - val_loss: 1.3382 - val_accuracy: 0.5650
Epoch 23/100
218/219 [============================>.] - ETA: 0s - loss: 1.4789 - accuracy: 0.4789
Epoch 23: val_loss improved from 1.33816 to 1.28234, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.4796 - accuracy: 0.4789 - val_loss: 1.2823 - val_accuracy: 0.5844
Epoch 24/100
213/219 [============================>.] - ETA: 0s - loss: 1.4545 - accuracy: 0.4900
Epoch 24: val_loss improved from 1.28234 to 1.24893, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 10ms/step - loss: 1.4560 - accuracy: 0.4898 - val_loss: 1.2489 - val_accuracy: 0.5902
Epoch 25/100
218/219 [============================>.] - ETA: 0s - loss: 1.4269 - accuracy: 0.5103
Epoch 25: val_loss did not improve from 1.24893
219/219 [==============================] - 2s 11ms/step - loss: 1.4270 - accuracy: 0.5104 - val_loss: 1.2592 - val_accuracy: 0.5930
Epoch 26/100
213/219 [============================>.] - ETA: 0s - loss: 1.4075 - accuracy: 0.5112
Epoch 26: val_loss improved from 1.24893 to 1.22649, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.4056 - accuracy: 0.5117 - val_loss: 1.2265 - val_accuracy: 0.5947
Epoch 27/100
217/219 [============================>.] - ETA: 0s - loss: 1.3983 - accuracy: 0.5181
Epoch 27: val_loss improved from 1.22649 to 1.21439, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 10ms/step - loss: 1.3990 - accuracy: 0.5185 - val_loss: 1.2144 - val_accuracy: 0.5993
Epoch 28/100
212/219 [============================>.] - ETA: 0s - loss: 1.3664 - accuracy: 0.5230
Epoch 28: val_loss improved from 1.21439 to 1.16763, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.3682 - accuracy: 0.5227 - val_loss: 1.1676 - val_accuracy: 0.6382
Epoch 29/100
217/219 [============================>.] - ETA: 0s - loss: 1.3713 - accuracy: 0.5372
Epoch 29: val_loss improved from 1.16763 to 1.16508, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 10ms/step - loss: 1.3732 - accuracy: 0.5369 - val_loss: 1.1651 - val_accuracy: 0.6193
Epoch 30/100
213/219 [============================>.] - ETA: 0s - loss: 1.3579 - accuracy: 0.5374
Epoch 30: val_loss did not improve from 1.16508
219/219 [==============================] - 2s 10ms/step - loss: 1.3579 - accuracy: 0.5372 - val_loss: 1.1789 - val_accuracy: 0.6211
Epoch 31/100
216/219 [============================>.] - ETA: 0s - loss: 1.3384 - accuracy: 0.5376
Epoch 31: val_loss did not improve from 1.16508
219/219 [==============================] - 2s 8ms/step - loss: 1.3398 - accuracy: 0.5369 - val_loss: 1.1738 - val_accuracy: 0.6199
Epoch 32/100
218/219 [============================>.] - ETA: 0s - loss: 1.3062 - accuracy: 0.5497
Epoch 32: val_loss improved from 1.16508 to 1.10832, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.3057 - accuracy: 0.5499 - val_loss: 1.1083 - val_accuracy: 0.6434
Epoch 33/100
219/219 [==============================] - ETA: 0s - loss: 1.2879 - accuracy: 0.5721
Epoch 33: val_loss did not improve from 1.10832
219/219 [==============================] - 2s 11ms/step - loss: 1.2879 - accuracy: 0.5721 - val_loss: 1.1191 - val_accuracy: 0.6371
Epoch 34/100
212/219 [============================>.] - ETA: 0s - loss: 1.2906 - accuracy: 0.5710
Epoch 34: val_loss did not improve from 1.10832
219/219 [==============================] - 2s 11ms/step - loss: 1.2903 - accuracy: 0.5704 - val_loss: 1.1443 - val_accuracy: 0.6279
Epoch 35/100
218/219 [============================>.] - ETA: 0s - loss: 1.2879 - accuracy: 0.5592
Epoch 35: val_loss did not improve from 1.10832
219/219 [==============================] - 2s 8ms/step - loss: 1.2877 - accuracy: 0.5592 - val_loss: 1.1413 - val_accuracy: 0.6188
Epoch 36/100
217/219 [============================>.] - ETA: 0s - loss: 1.2889 - accuracy: 0.5658
Epoch 36: val_loss did not improve from 1.10832
219/219 [==============================] - 2s 8ms/step - loss: 1.2898 - accuracy: 0.5652 - val_loss: 1.1145 - val_accuracy: 0.6451
Epoch 37/100
219/219 [==============================] - ETA: 0s - loss: 1.2400 - accuracy: 0.5798
Epoch 37: val_loss improved from 1.10832 to 1.07740, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.2400 - accuracy: 0.5798 - val_loss: 1.0774 - val_accuracy: 0.6474
Epoch 38/100
218/219 [============================>.] - ETA: 0s - loss: 1.2348 - accuracy: 0.5819
Epoch 38: val_loss improved from 1.07740 to 1.06601, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.2350 - accuracy: 0.5817 - val_loss: 1.0660 - val_accuracy: 0.6434
Epoch 39/100
213/219 [============================>.] - ETA: 0s - loss: 1.2385 - accuracy: 0.5781
Epoch 39: val_loss improved from 1.06601 to 1.03752, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.2384 - accuracy: 0.5774 - val_loss: 1.0375 - val_accuracy: 0.6588
Epoch 40/100
218/219 [============================>.] - ETA: 0s - loss: 1.2182 - accuracy: 0.5884
Epoch 40: val_loss improved from 1.03752 to 1.02943, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 10ms/step - loss: 1.2181 - accuracy: 0.5884 - val_loss: 1.0294 - val_accuracy: 0.6646
Epoch 41/100
218/219 [============================>.] - ETA: 0s - loss: 1.2139 - accuracy: 0.5866
Epoch 41: val_loss improved from 1.02943 to 1.01844, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 11ms/step - loss: 1.2143 - accuracy: 0.5864 - val_loss: 1.0184 - val_accuracy: 0.6726
Epoch 42/100
211/219 [===========================>..] - ETA: 0s - loss: 1.2324 - accuracy: 0.5847
Epoch 42: val_loss did not improve from 1.01844
219/219 [==============================] - 2s 7ms/step - loss: 1.2331 - accuracy: 0.5865 - val_loss: 1.0382 - val_accuracy: 0.6732
Epoch 43/100
211/219 [===========================>..] - ETA: 0s - loss: 1.1897 - accuracy: 0.5976
Epoch 43: val_loss did not improve from 1.01844
219/219 [==============================] - 2s 7ms/step - loss: 1.1919 - accuracy: 0.5969 - val_loss: 1.0210 - val_accuracy: 0.6726
Epoch 44/100
214/219 [============================>.] - ETA: 0s - loss: 1.1696 - accuracy: 0.6032
Epoch 44: val_loss did not improve from 1.01844
219/219 [==============================] - 2s 7ms/step - loss: 1.1674 - accuracy: 0.6040 - val_loss: 1.0330 - val_accuracy: 0.6720
Epoch 45/100
214/219 [============================>.] - ETA: 0s - loss: 1.1594 - accuracy: 0.6085
Epoch 45: val_loss improved from 1.01844 to 1.01521, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.1599 - accuracy: 0.6079 - val_loss: 1.0152 - val_accuracy: 0.6852
Epoch 46/100
217/219 [============================>.] - ETA: 0s - loss: 1.1637 - accuracy: 0.6142
Epoch 46: val_loss improved from 1.01521 to 0.98202, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.1633 - accuracy: 0.6140 - val_loss: 0.9820 - val_accuracy: 0.6915
Epoch 47/100
211/219 [===========================>..] - ETA: 0s - loss: 1.1625 - accuracy: 0.6173
Epoch 47: val_loss did not improve from 0.98202
219/219 [==============================] - 2s 9ms/step - loss: 1.1639 - accuracy: 0.6165 - val_loss: 0.9956 - val_accuracy: 0.6938
Epoch 48/100
217/219 [============================>.] - ETA: 0s - loss: 1.1628 - accuracy: 0.6080
Epoch 48: val_loss did not improve from 0.98202
219/219 [==============================] - 2s 11ms/step - loss: 1.1640 - accuracy: 0.6077 - val_loss: 0.9961 - val_accuracy: 0.6875
Epoch 49/100
213/219 [============================>.] - ETA: 0s - loss: 1.1388 - accuracy: 0.6121
Epoch 49: val_loss did not improve from 0.98202
219/219 [==============================] - 2s 7ms/step - loss: 1.1405 - accuracy: 0.6106 - val_loss: 0.9857 - val_accuracy: 0.6875
Epoch 50/100
213/219 [============================>.] - ETA: 0s - loss: 1.1508 - accuracy: 0.6171
Epoch 50: val_loss improved from 0.98202 to 0.97007, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 7ms/step - loss: 1.1502 - accuracy: 0.6180 - val_loss: 0.9701 - val_accuracy: 0.6932
Epoch 51/100
210/219 [===========================>..] - ETA: 0s - loss: 1.1307 - accuracy: 0.6238
Epoch 51: val_loss improved from 0.97007 to 0.96330, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.1305 - accuracy: 0.6252 - val_loss: 0.9633 - val_accuracy: 0.6920
Epoch 52/100
214/219 [============================>.] - ETA: 0s - loss: 1.1208 - accuracy: 0.6235
Epoch 52: val_loss did not improve from 0.96330
219/219 [==============================] - 3s 12ms/step - loss: 1.1211 - accuracy: 0.6242 - val_loss: 0.9660 - val_accuracy: 0.6869
Epoch 53/100
211/219 [===========================>..] - ETA: 0s - loss: 1.1261 - accuracy: 0.6284
Epoch 53: val_loss improved from 0.96330 to 0.94198, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.1246 - accuracy: 0.6279 - val_loss: 0.9420 - val_accuracy: 0.7109
Epoch 54/100
214/219 [============================>.] - ETA: 0s - loss: 1.1074 - accuracy: 0.6275
Epoch 54: val_loss did not improve from 0.94198
219/219 [==============================] - 2s 9ms/step - loss: 1.1074 - accuracy: 0.6269 - val_loss: 0.9558 - val_accuracy: 0.6938
Epoch 55/100
213/219 [============================>.] - ETA: 0s - loss: 1.1031 - accuracy: 0.6307
Epoch 55: val_loss improved from 0.94198 to 0.92321, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.1039 - accuracy: 0.6304 - val_loss: 0.9232 - val_accuracy: 0.7052
Epoch 56/100
212/219 [============================>.] - ETA: 0s - loss: 1.1053 - accuracy: 0.6321
Epoch 56: val_loss did not improve from 0.92321
219/219 [==============================] - 1s 7ms/step - loss: 1.1087 - accuracy: 0.6309 - val_loss: 0.9496 - val_accuracy: 0.7069
Epoch 57/100
213/219 [============================>.] - ETA: 0s - loss: 1.1087 - accuracy: 0.6244
Epoch 57: val_loss did not improve from 0.92321
219/219 [==============================] - 2s 9ms/step - loss: 1.1079 - accuracy: 0.6251 - val_loss: 0.9480 - val_accuracy: 0.7029
Epoch 58/100
212/219 [============================>.] - ETA: 0s - loss: 1.0948 - accuracy: 0.6322
Epoch 58: val_loss did not improve from 0.92321
219/219 [==============================] - 2s 8ms/step - loss: 1.0930 - accuracy: 0.6324 - val_loss: 0.9482 - val_accuracy: 0.6972
Epoch 59/100
211/219 [===========================>..] - ETA: 0s - loss: 1.1210 - accuracy: 0.6259
Epoch 59: val_loss did not improve from 0.92321
219/219 [==============================] - 1s 7ms/step - loss: 1.1213 - accuracy: 0.6255 - val_loss: 0.9373 - val_accuracy: 0.7167
Epoch 60/100
212/219 [============================>.] - ETA: 0s - loss: 1.1090 - accuracy: 0.6310
Epoch 60: val_loss did not improve from 0.92321
219/219 [==============================] - 1s 7ms/step - loss: 1.1070 - accuracy: 0.6312 - val_loss: 0.9518 - val_accuracy: 0.6920
Epoch 61/100
218/219 [============================>.] - ETA: 0s - loss: 1.0956 - accuracy: 0.6385
Epoch 61: val_loss did not improve from 0.92321
219/219 [==============================] - 1s 7ms/step - loss: 1.0954 - accuracy: 0.6387 - val_loss: 0.9313 - val_accuracy: 0.7184
Epoch 62/100
216/219 [============================>.] - ETA: 0s - loss: 1.0837 - accuracy: 0.6413
Epoch 62: val_loss improved from 0.92321 to 0.91435, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.0846 - accuracy: 0.6408 - val_loss: 0.9144 - val_accuracy: 0.7149
Epoch 63/100
215/219 [============================>.] - ETA: 0s - loss: 1.0745 - accuracy: 0.6490
Epoch 63: val_loss improved from 0.91435 to 0.91235, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.0753 - accuracy: 0.6485 - val_loss: 0.9124 - val_accuracy: 0.7058
Epoch 64/100
219/219 [==============================] - ETA: 0s - loss: 1.0946 - accuracy: 0.6362
Epoch 64: val_loss did not improve from 0.91235
219/219 [==============================] - 2s 7ms/step - loss: 1.0946 - accuracy: 0.6362 - val_loss: 0.9148 - val_accuracy: 0.7104
Epoch 65/100
214/219 [============================>.] - ETA: 0s - loss: 1.0746 - accuracy: 0.6504
Epoch 65: val_loss improved from 0.91235 to 0.88194, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.0709 - accuracy: 0.6520 - val_loss: 0.8819 - val_accuracy: 0.7252
Epoch 66/100
216/219 [============================>.] - ETA: 0s - loss: 1.0728 - accuracy: 0.6427
Epoch 66: val_loss did not improve from 0.88194
219/219 [==============================] - 2s 7ms/step - loss: 1.0736 - accuracy: 0.6427 - val_loss: 0.9367 - val_accuracy: 0.7064
Epoch 67/100
218/219 [============================>.] - ETA: 0s - loss: 1.0698 - accuracy: 0.6464
Epoch 67: val_loss did not improve from 0.88194
219/219 [==============================] - 2s 7ms/step - loss: 1.0694 - accuracy: 0.6464 - val_loss: 0.9046 - val_accuracy: 0.7144
Epoch 68/100
215/219 [============================>.] - ETA: 0s - loss: 1.0504 - accuracy: 0.6578
Epoch 68: val_loss did not improve from 0.88194
219/219 [==============================] - 1s 7ms/step - loss: 1.0514 - accuracy: 0.6578 - val_loss: 0.8934 - val_accuracy: 0.7155
Epoch 69/100
217/219 [============================>.] - ETA: 0s - loss: 1.0758 - accuracy: 0.6437
Epoch 69: val_loss did not improve from 0.88194
219/219 [==============================] - 1s 7ms/step - loss: 1.0767 - accuracy: 0.6435 - val_loss: 0.9087 - val_accuracy: 0.7041
Epoch 70/100
211/219 [===========================>..] - ETA: 0s - loss: 1.0637 - accuracy: 0.6460
Epoch 70: val_loss did not improve from 0.88194
219/219 [==============================] - 2s 7ms/step - loss: 1.0657 - accuracy: 0.6451 - val_loss: 0.9115 - val_accuracy: 0.7086
Epoch 71/100
215/219 [============================>.] - ETA: 0s - loss: 1.0518 - accuracy: 0.6499
Epoch 71: val_loss improved from 0.88194 to 0.88157, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.0512 - accuracy: 0.6495 - val_loss: 0.8816 - val_accuracy: 0.7258
Epoch 72/100
214/219 [============================>.] - ETA: 0s - loss: 1.0494 - accuracy: 0.6508
Epoch 72: val_loss improved from 0.88157 to 0.87599, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.0491 - accuracy: 0.6505 - val_loss: 0.8760 - val_accuracy: 0.7167
Epoch 73/100
215/219 [============================>.] - ETA: 0s - loss: 1.0640 - accuracy: 0.6485
Epoch 73: val_loss did not improve from 0.87599
219/219 [==============================] - 2s 7ms/step - loss: 1.0668 - accuracy: 0.6482 - val_loss: 0.9061 - val_accuracy: 0.7138
Epoch 74/100
212/219 [============================>.] - ETA: 0s - loss: 1.0637 - accuracy: 0.6453
Epoch 74: val_loss did not improve from 0.87599
219/219 [==============================] - 2s 8ms/step - loss: 1.0632 - accuracy: 0.6457 - val_loss: 0.8777 - val_accuracy: 0.7178
Epoch 75/100
209/219 [===========================>..] - ETA: 0s - loss: 1.0508 - accuracy: 0.6479
Epoch 75: val_loss improved from 0.87599 to 0.87122, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.0545 - accuracy: 0.6462 - val_loss: 0.8712 - val_accuracy: 0.7304
Epoch 76/100
215/219 [============================>.] - ETA: 0s - loss: 1.0381 - accuracy: 0.6557
Epoch 76: val_loss improved from 0.87122 to 0.86911, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 9ms/step - loss: 1.0366 - accuracy: 0.6557 - val_loss: 0.8691 - val_accuracy: 0.7247
Epoch 77/100
210/219 [===========================>..] - ETA: 0s - loss: 1.0414 - accuracy: 0.6545
Epoch 77: val_loss did not improve from 0.86911
219/219 [==============================] - 2s 8ms/step - loss: 1.0444 - accuracy: 0.6541 - val_loss: 0.8745 - val_accuracy: 0.7149
Epoch 78/100
211/219 [===========================>..] - ETA: 0s - loss: 1.0322 - accuracy: 0.6600
Epoch 78: val_loss did not improve from 0.86911
219/219 [==============================] - 1s 6ms/step - loss: 1.0367 - accuracy: 0.6581 - val_loss: 0.8887 - val_accuracy: 0.7161
Epoch 79/100
212/219 [============================>.] - ETA: 0s - loss: 1.0382 - accuracy: 0.6543
Epoch 79: val_loss improved from 0.86911 to 0.86301, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.0395 - accuracy: 0.6534 - val_loss: 0.8630 - val_accuracy: 0.7355
Epoch 80/100
214/219 [============================>.] - ETA: 0s - loss: 1.0287 - accuracy: 0.6574
Epoch 80: val_loss improved from 0.86301 to 0.85457, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.0283 - accuracy: 0.6578 - val_loss: 0.8546 - val_accuracy: 0.7396
Epoch 81/100
219/219 [==============================] - ETA: 0s - loss: 1.0327 - accuracy: 0.6618
Epoch 81: val_loss did not improve from 0.85457
219/219 [==============================] - 1s 7ms/step - loss: 1.0327 - accuracy: 0.6618 - val_loss: 0.8625 - val_accuracy: 0.7470
Epoch 82/100
212/219 [============================>.] - ETA: 0s - loss: 1.0629 - accuracy: 0.6536
Epoch 82: val_loss improved from 0.85457 to 0.85456, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.0627 - accuracy: 0.6541 - val_loss: 0.8546 - val_accuracy: 0.7338
Epoch 83/100
211/219 [===========================>..] - ETA: 0s - loss: 1.0139 - accuracy: 0.6617
Epoch 83: val_loss did not improve from 0.85456
219/219 [==============================] - 1s 7ms/step - loss: 1.0166 - accuracy: 0.6611 - val_loss: 0.8622 - val_accuracy: 0.7373
Epoch 84/100
210/219 [===========================>..] - ETA: 0s - loss: 1.0374 - accuracy: 0.6557
Epoch 84: val_loss improved from 0.85456 to 0.83837, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.0373 - accuracy: 0.6555 - val_loss: 0.8384 - val_accuracy: 0.7367
Epoch 85/100
212/219 [============================>.] - ETA: 0s - loss: 1.0192 - accuracy: 0.6679
Epoch 85: val_loss did not improve from 0.83837
219/219 [==============================] - 1s 6ms/step - loss: 1.0256 - accuracy: 0.6656 - val_loss: 0.8445 - val_accuracy: 0.7464
Epoch 86/100
213/219 [============================>.] - ETA: 0s - loss: 1.0247 - accuracy: 0.6605
Epoch 86: val_loss did not improve from 0.83837
219/219 [==============================] - 1s 7ms/step - loss: 1.0256 - accuracy: 0.6593 - val_loss: 0.8485 - val_accuracy: 0.7407
Epoch 87/100
214/219 [============================>.] - ETA: 0s - loss: 1.0137 - accuracy: 0.6660
Epoch 87: val_loss did not improve from 0.83837
219/219 [==============================] - 1s 6ms/step - loss: 1.0172 - accuracy: 0.6647 - val_loss: 0.8607 - val_accuracy: 0.7230
Epoch 88/100
214/219 [============================>.] - ETA: 0s - loss: 1.0189 - accuracy: 0.6640
Epoch 88: val_loss improved from 0.83837 to 0.83393, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 7ms/step - loss: 1.0171 - accuracy: 0.6640 - val_loss: 0.8339 - val_accuracy: 0.7361
Epoch 89/100
214/219 [============================>.] - ETA: 0s - loss: 1.0301 - accuracy: 0.6641
Epoch 89: val_loss did not improve from 0.83393
219/219 [==============================] - 1s 7ms/step - loss: 1.0309 - accuracy: 0.6631 - val_loss: 0.8721 - val_accuracy: 0.7338
Epoch 90/100
215/219 [============================>.] - ETA: 0s - loss: 1.0212 - accuracy: 0.6584
Epoch 90: val_loss improved from 0.83393 to 0.82755, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 7ms/step - loss: 1.0175 - accuracy: 0.6597 - val_loss: 0.8275 - val_accuracy: 0.7367
Epoch 91/100
215/219 [============================>.] - ETA: 0s - loss: 1.0002 - accuracy: 0.6705
Epoch 91: val_loss did not improve from 0.82755
219/219 [==============================] - 1s 6ms/step - loss: 1.0016 - accuracy: 0.6704 - val_loss: 0.8470 - val_accuracy: 0.7424
Epoch 92/100
218/219 [============================>.] - ETA: 0s - loss: 1.0094 - accuracy: 0.6676
Epoch 92: val_loss did not improve from 0.82755
219/219 [==============================] - 1s 7ms/step - loss: 1.0083 - accuracy: 0.6680 - val_loss: 0.8397 - val_accuracy: 0.7396
Epoch 93/100
215/219 [============================>.] - ETA: 0s - loss: 1.0085 - accuracy: 0.6651
Epoch 93: val_loss improved from 0.82755 to 0.82668, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 8ms/step - loss: 1.0081 - accuracy: 0.6654 - val_loss: 0.8267 - val_accuracy: 0.7476
Epoch 94/100
214/219 [============================>.] - ETA: 0s - loss: 1.0236 - accuracy: 0.6659
Epoch 94: val_loss did not improve from 0.82668
219/219 [==============================] - 2s 7ms/step - loss: 1.0253 - accuracy: 0.6651 - val_loss: 0.8386 - val_accuracy: 0.7310
Epoch 95/100
218/219 [============================>.] - ETA: 0s - loss: 0.9983 - accuracy: 0.6686
Epoch 95: val_loss did not improve from 0.82668
219/219 [==============================] - 2s 7ms/step - loss: 0.9984 - accuracy: 0.6684 - val_loss: 0.8423 - val_accuracy: 0.7310
Epoch 96/100
213/219 [============================>.] - ETA: 0s - loss: 1.0006 - accuracy: 0.6646
Epoch 96: val_loss did not improve from 0.82668
219/219 [==============================] - 2s 7ms/step - loss: 1.0002 - accuracy: 0.6643 - val_loss: 0.8418 - val_accuracy: 0.7413
Epoch 97/100
215/219 [============================>.] - ETA: 0s - loss: 1.0173 - accuracy: 0.6695
Epoch 97: val_loss improved from 0.82668 to 0.82628, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 2s 7ms/step - loss: 1.0167 - accuracy: 0.6689 - val_loss: 0.8263 - val_accuracy: 0.7367
Epoch 98/100
218/219 [============================>.] - ETA: 0s - loss: 1.0244 - accuracy: 0.6649
Epoch 98: val_loss improved from 0.82628 to 0.82414, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 6ms/step - loss: 1.0237 - accuracy: 0.6650 - val_loss: 0.8241 - val_accuracy: 0.7470
Epoch 99/100
211/219 [===========================>..] - ETA: 0s - loss: 1.0029 - accuracy: 0.6754
Epoch 99: val_loss improved from 0.82414 to 0.81871, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 7ms/step - loss: 1.0002 - accuracy: 0.6769 - val_loss: 0.8187 - val_accuracy: 0.7533
Epoch 100/100
218/219 [============================>.] - ETA: 0s - loss: 1.0144 - accuracy: 0.6676
Epoch 100: val_loss did not improve from 0.81871
219/219 [==============================] - 1s 6ms/step - loss: 1.0142 - accuracy: 0.6674 - val_loss: 0.8281 - val_accuracy: 0.7396 """

duration = datetime.now() - start
print("Training completed in time: ", duration) # Training completed in time:  0:03:26.147135

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1]) # 0.7395535111427307

filename="dataset\\audio\\2937-1-0-0.wav"
prediction_feature=features_extractor(filename)
"""
array([-315.6028    ,   94.854805  ,  -37.22234   ,   46.778263  ,
         -6.728693  ,   10.012548  ,   -1.607553  ,   18.511341  ,
        -11.9006195 ,    7.5940356 ,   -7.8546596 ,   11.362425  ,
        -15.617316  ,    3.301991  ,  -11.958161  ,    6.353489  ,
         -5.5870256 ,   20.78539   ,   -0.4692282 ,    6.0436325 ,
        -11.619548  ,    2.8686748 ,  -10.176432  ,    8.332485  ,
          1.7765609 ,    2.5638974 ,  -14.761059  ,    1.1465564 ,
          3.7835658 ,    3.1094651 ,  -12.185812  ,   -3.0522912 ,
          3.7284145 ,    8.962753  ,    0.9306449 ,    3.1800797 ,
          2.485049  ,    0.61386466,  -11.449189  ,   -6.0105853 ],
      dtype=float32) """

prediction_feature=prediction_feature.reshape(1,-1)
"""
array([[-315.6028    ,   94.854805  ,  -37.22234   ,   46.778263  ,
          -6.728693  ,   10.012548  ,   -1.607553  ,   18.511341  ,
         -11.9006195 ,    7.5940356 ,   -7.8546596 ,   11.362425  ,
         -15.617316  ,    3.301991  ,  -11.958161  ,    6.353489  ,
          -5.5870256 ,   20.78539   ,   -0.4692282 ,    6.0436325 ,
         -11.619548  ,    2.8686748 ,  -10.176432  ,    8.332485  ,
           1.7765609 ,    2.5638974 ,  -14.761059  ,    1.1465564 ,
           3.7835658 ,    3.1094651 ,  -12.185812  ,   -3.0522912 ,
           3.7284145 ,    8.962753  ,    0.9306449 ,    3.1800797 ,
           2.485049  ,    0.61386466,  -11.449189  ,   -6.0105853 ]],
      dtype=float32) """

model.predict_classes(prediction_feature)
"""
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Sequential' object has no attribute 'predict_classes'. Did you mean: 'predict_step'? """

# https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
predict_x=model.predict(prediction_feature)
"""
array([[1.13554115e-07, 7.69973099e-01, 4.81419760e-04, 1.13897892e-02,
        1.14212409e-02, 3.58158140e-05, 1.61009171e-04, 2.63001311e-06,
        3.44199907e-05, 2.06500456e-01]], dtype=float32) """

classes_x=np.argmax(predict_x,axis=1) # array([1], dtype=int64)
# 1 is car horn

"""
when label encoder is used:

predicted_label=model.predict_classes(mfccs_scaled_features)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label)
prediction_class """