from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
import model
import data
import tools

# seed = 0
# np.random.seed(seed)
# tf.random.set_seed(seed)

X_train, X_test, y_train, y_test, le_classes = data.load_data()

model = model.get_float_model()
model.compile(param["fit"]["compile"])

# print model summary
print('#################')
print('# MODEL SUMMARY #')
print('#################')
print(model.summary())
print('#################')

# callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
 callbacks = [ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True),
            EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
            LearningRateScheduler(lr_schedule_func, verbose=verbose),
            ]
# train
history = model.fit(X_train,
    y_train,
    epochs=param["fit"]["epochs"],
    batch_size=param["fit"]["batch_size"],
    shuffle=param["fit"]["shuffle"],
    validation_split=param["fit"]["validation_split"],
    verbose=param["fit"]["verbose"],
    callbacks=callbacks)

#save model
model.save(param["model_directory"])

# restore "best" model
model.load_weights(model_file_path)

# get predictions
y_pred = model.predict(X_test)

# evaluate with test dataset and share same prediction results

evaluation = model.evaluate(X_test, y_test)

auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
print('Model test accuracy = %.3f' % evaluation[1])
csv_lines.append('Acc', '%.3f' % evaluation[1])
print('Model test weighted average AUC = %.3f' % auc)
csv_lines.append('AUC', auc)

#plot and save metrics 
print("Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
# initialize lines in csv for AUC and pAUC

plt.figure(figsize=(9,9))
_ = plotting.makeRoc(y_test, y_keras, le.classes_)
plt.savefig('{}/keras_roc_curve'.format(param['model_directory']))

result_path = "{result}/result.csv".format(result=param["model_directory"])
tools.save_csv(save_file_path=result_path, save_data=csv_lines)