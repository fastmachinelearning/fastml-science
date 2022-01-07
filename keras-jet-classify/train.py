import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import os
import jet_tagger_model
import data
import tools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

def main(args):
    param = tools.yaml_load(args.config)

    # Fetch the jet tagging dataset from Open ML
    X_train, X_test, y_train, y_test, le_classes = data.load_data()
    print("train dataset size: " + str(len(X_train)))
    print("test dataset size: " + str(len(X_test)))

    # construct and compile model
    model = jet_tagger_model.get_model(param['model']['name'], 
                                        param['model']['shape'], 
                                        fc_bits = param['model']['quantization']['fc_bits'],
                                        fc_int_bits = param['model']['quantization']['fc_int_bits'],
                                        relu_bits = param['model']['quantization']['relu_bits'],)

    model.compile(optimizer=param["fit"]["compile"]["optimizer"], loss=param["fit"]["compile"]["loss"], metrics=["accuracy"])

    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    # declare necessary callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    callbacks = [ModelCheckpoint(param["model_directory"], monitor='val_loss', verbose=param["fit"]["verbose"], save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=10, verbose=param["fit"]["verbose"], restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, mode='min', 
                    verbose=1, epsilon=0.001,cooldown=4, min_lr=1e-5)]
    # Train the model
    history = model.fit(X_train,
        y_train,
        epochs=param["fit"]["epochs"],
        batch_size=param["fit"]["batch_size"],
        shuffle=param["fit"]["shuffle"],
        validation_split=param["fit"]["validation_split"],
        verbose=param["fit"]["verbose"],
        callbacks=callbacks)

    # restore "best" model
    model.load_weights(param["model_directory"]).expect_partial()

    # get predictions
    y_pred = model.predict(X_test)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []
    model_save_path = param["model_directory"]
    result_save_path = '{}/results.csv'.format(model_save_path)

    # evaluate with test dataset and share same prediction results
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
    print('Model test accuracy = %.3f' % acc)
    csv_lines.append(['Acc:', '%.3f' % acc])
    print('Model test weighted average AUC = %.3f' % auc)
    csv_lines.append(['AUC:', auc])

    #plot and save metrics 
    tools.makeRoc(y_test, y_pred, le_classes, save_dir='{}/keras_roc_curve'.format(model_save_path))

    result_path = "{}/result.csv".format(model_save_path)

    # save model
    model.save("{}/model.h5".format(model_save_path))
    tools.save_csv(save_file_path=result_save_path, save_data=csv_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-c', '--config', type=str, default = "baseline.yml", help="specify yml config")
    args = parser.parse_args()
    main(args)