import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import pandas as pd

## Define path data
COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'date_time', 'Ch-Hole', 'optel_schedule_wo', 'rejected']
#COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
features = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
PATH = "gc10-classify.csv"
PATH_test = "gc10-classify-test.csv"

df_train = pd.read_csv(PATH, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)
df_test = pd.read_csv(PATH_test, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)

df_train = df_train[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", "rejected"]]
df_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", "rejected"]]

#print(df_train.shape, df_test.shape)
#print(df_train.dtypes)


# Generate tain and test data
#X, Y = make_classification(n_samples=50000, n_features=10, n_informative=8, 
#                           n_redundant=0, n_clusters_per_class=2)
#Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class
#X, X_test, Y, Y_test = train_test_split(X, Y)

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

#print (df_train.dtypes)

#preprocessing_layer = tf.keras.layers.DenseFeatures(df_train)

#target = df_train.pop('target')


features = df_train[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor"]]
labels = df_train[["rejected"]]
features_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor"]]
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

model.save('picknplace_model.h5')

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

#predictions = model.predict(test_data)

# Show some results
#for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
#  print("Predicted survival: {:.2%}".format(prediction[0]),
#        " | Actual outcome: ",
#        ("Working" if bool(survived) else "Failed"))
