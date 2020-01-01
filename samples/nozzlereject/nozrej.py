import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import pandas as pd

## Define path data
COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'date_time', 'Ch-Hole', 'optel_schedule_wo', 'rejected']
#COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
features = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
PATH = "gc10-classify.csv"
PATH_test = "gc10-classify-test.csv"

df_train = pd.read_csv(PATH, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)
df_test = pd.read_csv(PATH_test, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)

df_train = df_train[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", "rejected","Ch-Hole", "optel_schedule_wo"]]
df_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", "rejected","Ch-Hole", "optel_schedule_wo"]]

features = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']

df_train['Noz Rejects Sum%'] = df_train['Noz Rejects Sum%'].astype(float)
df_train['Noz Rejects Sum'] = df_train['Noz Rejects Sum'].astype(float)
df_train['Noz Reject Factor'] = df_train['Noz Reject Factor'].astype(float)
df_train['rejected'] = df_train['rejected'].astype(float)

df_test['Noz Rejects Sum%'] = df_test['Noz Rejects Sum%'].astype(float)
df_test['Noz Rejects Sum'] = df_test['Noz Rejects Sum'].astype(float)
df_test['Noz Reject Factor'] = df_test['Noz Reject Factor'].astype(float)
df_test['rejected'] = df_test['rejected'].astype(float)

#desc = pd.read_csv(PATH)[features].describe()
#print(desc)

print (df_train.dtypes)

#features = df_train[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor"]]
#chhole = to_categorical(df_train['Ch-Hole'])
#optelno = to_categorical(df_train['optel_schedule_wo'], num_classes=None)

#df_train.insert(3, "chhole", chhole, True)
#df_train['chhole'] = chhole
df_train['chhole'] = df_train['Ch-Hole'].astype('category').cat.codes
df_train['optelno'] = df_train['optel_schedule_wo'].astype('category').cat.codes

#df_train['chhole'] = to_categorical(df_train['Ch-Hole'])
#df_train['optelno'] = to_categorical(df_train['optel_schedule_wo'], num_classes=None)

df_test['chhole'] = df_test['Ch-Hole'].astype('category').cat.codes
df_test['optelno'] = df_test['optel_schedule_wo'].astype('category').cat.codes

#df_test['chhole'] = to_categorical(df_test['Ch-Hole'])
#df_test['optelno'] = to_categorical(df_test['optel_schedule_wo'])

features = df_train[['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'chhole', 'optelno']]
#features = df_train[['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor']]
labels = df_train[["rejected"]]
features_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", 'chhole', 'optelno']]
labels_test = df_test[["rejected"]]

dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
dataset_test = tf.data.Dataset.from_tensor_slices((features_test.values, labels_test.values))

#for feat, targ in dataset.take(5):
#  print ('Features: {}, Target: {}'.format(feat, targ))

#train_data = df_train
#test_data = df_test

train_data = dataset.shuffle(len(df_train)).batch(1)
test_data = dataset_test.shuffle(len(df_test)).batch(1)

model = tf.keras.Sequential([
  #preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


model.fit(train_data, epochs=5)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

#predictions = model.predict(test_data)

# Show some results
#for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
#  print("Predicted survival: {:.2%}".format(prediction[0]),
#        " | Actual outcome: ",
#        ("Working" if bool(survived) else "Failed"))

