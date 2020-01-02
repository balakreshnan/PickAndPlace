# Nozzle prediction for pick and place SMT line

## Architecture
![alt text](https://github.com/balakreshnan/PickAndPlace/blob/master/picknplace.jpg "Architecture Pick and Place")

## pre requistie

took the data set and added a column call rejected. for rejected column the formula is is nozlle reject sum is greater then 1 then 1 or else it is 0. Basically this signifies if the nozzle is going to fail or not. We can adjust the threshold from 1 to what ever we need based on the reality.

## Implementation

## Imports

```
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import pandas as pd
```

## define columns and data path

```
## Define path data
COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'date_time', 'Ch-Hole', 'optel_schedule_wo', 'rejected']
#COLUMNS = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
features = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']
PATH = "gc10-classify.csv"
PATH_test = "gc10-classify-test.csv"
```

## Load data and select only columns needed

```
df_train = pd.read_csv(PATH, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)
df_test = pd.read_csv(PATH_test, skipinitialspace=True, names = COLUMNS, index_col=False, header=1)

df_train = df_train[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", "rejected","Ch-Hole", "optel_schedule_wo"]]
df_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", "rejected","Ch-Hole", "optel_schedule_wo"]]
```

## Convert columns to float.

This is needed since csv file are read as string data type in each column

```
features = ['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'rejected']

df_train['Noz Rejects Sum%'] = df_train['Noz Rejects Sum%'].astype(float)
df_train['Noz Rejects Sum'] = df_train['Noz Rejects Sum'].astype(float)
df_train['Noz Reject Factor'] = df_train['Noz Reject Factor'].astype(float)
df_train['rejected'] = df_train['rejected'].astype(float)

df_test['Noz Rejects Sum%'] = df_test['Noz Rejects Sum%'].astype(float)
df_test['Noz Rejects Sum'] = df_test['Noz Rejects Sum'].astype(float)
df_test['Noz Reject Factor'] = df_test['Noz Reject Factor'].astype(float)
df_test['rejected'] = df_test['rejected'].astype(float)

```

## convert text column to categorical

```
df_train['chhole'] = df_train['Ch-Hole'].astype('category').cat.codes
df_train['optelno'] = df_train['optel_schedule_wo'].astype('category').cat.codes

df_test['chhole'] = df_test['Ch-Hole'].astype('category').cat.codes
df_test['optelno'] = df_test['optel_schedule_wo'].astype('category').cat.codes
```

## Prepare data set for tensorflow model

Features and label are split. features and labels are converted to vector for further processing.

```
features = df_train[['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor', 'chhole', 'optelno']]
#features = df_train[['Noz Rejects Sum%', 'Noz Rejects Sum', 'Noz Reject Factor']]
labels = df_train[["rejected"]]
features_test = df_test[["Noz Rejects Sum%", "Noz Rejects Sum", "Noz Reject Factor", 'chhole', 'optelno']]
labels_test = df_test[["rejected"]]

dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
dataset_test = tf.data.Dataset.from_tensor_slices((features_test.values, labels_test.values))
```

## Set the training and test date set

```
train_data = dataset.shuffle(len(df_train)).batch(1)
test_data = dataset_test.shuffle(len(df_test)).batch(1)
```

## setup the Model configuration and layers

```
model = tf.keras.Sequential([
  #preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

```

## run the model

```
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


model.fit(train_data, epochs=5)
```

## test the model

```
test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
```

## Validate with test data

```
predictions = model.predict(test_data)

# Show some results
for prediction, rejected in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted rejected: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("Working" if bool(rejected) else "Failed"))
```

## Now save the model and build inferencing code to deploy to production.

## Follow the above to retraining process as well.
