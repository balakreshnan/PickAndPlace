import tensorflow as tf
import keras
import h5py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import pandas as pd

from keras import backend as K
# This line must be executed before loading Keras model.
#K.set_learning_phase(0)

#from keras.models import load_model
from tensorflow.keras.models import load_model
model = load_model('picknplace_model.h5')
#model = tf.keras.models.load_model('picknplace_model.h5')
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)
# [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]

## Define path data
COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'date_time', 'Ch-Hole', 'optel_schedule_wo', 'rejected']
#COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
features = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
PATH = "gc10-classify.csv"
PATH_test = "gc10-classify-test.csv"

df_train = pd.read_csv(PATH, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)
df_test = pd.read_csv(PATH_test, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)


df_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", "rejected","Ch-Hole", "optel_schedule_wo"]]
features = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']


df_test['Noz Rejects Sum%'] = df_test['Noz Rejects Sum%'].astype(float)
df_test['Noz Rejects Sum'] = df_test['Noz Rejects Sum'].astype(float)
df_test['Noz Reject Factor'] = df_test['Noz Reject Factor'].astype(float)
df_test['rejected'] = df_test['rejected'].astype(float)
df_test['chhole'] = df_test['Ch-Hole'].astype('category').cat.codes
df_test['optelno'] = df_test['optel_schedule_wo'].astype('category').cat.codes
features_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", 'chhole', 'optelno']]
labels_test = df_test[["rejected"]]
dataset_test = tf.data.Dataset.from_tensor_slices((features_test.values, labels_test.values))
test_data = dataset_test.shuffle(len(df_test)).batch(1)

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = model.evaluate(test_data)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))