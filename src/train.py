import yaml
import os
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import StratifiedKFold
from model import vgg16Model, resnet50Model, xceptionModel, nesnatlargeModel, inceptionV3Model, mobileNetModel, cnnModel
from util import load_df, load_x_data, load_y_data, seed_fix, f1_m, precision_m, recall_m
from inference import plot_loss_graph, plot_confusion_matrix, plot_roc_curve, export_train_result
import warnings
warnings.filterwarnings(action='ignore')


def train(train_path, learning_rate, epoch, batch_size, seed, model_list):
    target_size = [400, 400, 1]
    
    experiment_path = datetime.now().strftime("../result_dir/%Y-%m-%d_%H:%M")
    os.makedirs(experiment_path, exist_ok=True)
    
    for model_name in model_list:
        model_save_path = os.path.join(experiment_path, model_name)
        if os.path.isdir(model_save_path) == False:
            os.makedirs(model_save_path, exist_ok=True)
            os.makedirs(model_save_path + "/log_file", exist_ok=True)
            os.makedirs(model_save_path + "/plot", exist_ok=True)
            os.makedirs(model_save_path + "/weights", exist_ok=True)
            os.makedirs(model_save_path + "/train_result", exist_ok=True)
            os.makedirs(model_save_path + "/test_result", exist_ok=True)
        
        fold_number = 1
        class CustomCallback(Callback):
                def on_train_begin(self, logs = None):
                    raw_data = {'epoch' : [],
                                'train_loss' : [],
                                'train_accuracy' : [],
                                'validation_loss' : [],
                                'validation_accuracy': [],
                                }
                    df = pd.DataFrame(raw_data)
                    df.to_csv(model_save_path + "/log_file/" + "fold_" + str(fold_number) + ".csv", index = False)

                def on_epoch_end(self, epoch, logs=None):
                    df = pd.read_csv(model_save_path + "/log_file/" + "fold_" + str(fold_number) + ".csv")
                    df.loc[-1]=[epoch, logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"]]
                    df.to_csv(model_save_path + "/log_file/" + "fold_" + str(fold_number) + ".csv", index = False)
                    
        
        train_df = load_df(train_path)    
        feature = train_df['filename']
        target = train_df['y_label']
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)

        val_df = pd.DataFrame({'loss': [], 'accuracy': [], 'f1_score': [], 'precision': [], 'recall': [], 'weights_path': []})
        val_df.to_csv(model_save_path + "/train_result" + "/cross_val.csv", index = False)
        
        for train_index, validation_index in skf.split(feature, target):
            os.makedirs(model_save_path + f"/plot/{str(fold_number)}", exist_ok=True)
            tf.keras.backend.clear_session()
            train_fold_df = train_df.iloc[train_index]
            val_fold_df = train_df.iloc[validation_index]
            
            y_train = load_y_data(train_fold_df)
            y_val = load_y_data(val_fold_df)
            
            b1_x_train = load_x_data(train_path, "median_blur", target_size, train_fold_df)
            b2_x_train = load_x_data(train_path, "sobel_masking_y", target_size, train_fold_df)
            b3_x_train = load_x_data(train_path, "original_image", target_size, train_fold_df)
            
            b1_x_val = load_x_data(train_path, "median_blur", target_size, val_fold_df)
            b2_x_val = load_x_data(train_path, "sobel_masking_y", target_size, val_fold_df)
            b3_x_val = load_x_data(train_path, "original_image", target_size, val_fold_df)
            
            if model_name == "VGG16":
                model = vgg16Model(target_size)
            elif model_name == "InceptionV3":
                model = inceptionV3Model(target_size)
            elif model_name == "ResNet50":
                model = resnet50Model(target_size)
            elif model_name == "MobileNet":
                model = mobileNetModel(target_size)
            elif model_name == "CNN":
                model = cnnModel(target_size)
            else:
                print("Invalid model_name")


            filename = (model_save_path + '/weights/' + "fold_" + str(fold_number) + ".h5")
            checkpoint = ModelCheckpoint(filename,
                                monitor = 'val_loss',
                                verbose = 1,
                                save_best_only = True,
                                mode = 'auto')

            earlystopping = EarlyStopping(monitor = 'val_loss', patience = 20)
            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics=['acc', f1_m, precision_m, recall_m])

            history = model.fit(x=(b1_x_train, b2_x_train, b3_x_train), y= y_train, validation_data=((b1_x_val, b2_x_val, b3_x_val), y_val), epochs = epoch, batch_size= batch_size, callbacks = [checkpoint, earlystopping, CustomCallback()])
            plot_loss_graph(history, model_save_path, fold_number)

            loss, accuracy, f1, precision, recall = model.evaluate(x=(b1_x_val, b2_x_val, b3_x_val), y=y_val)
            val_df = pd.read_csv(model_save_path + "/train_result" + "/cross_val.csv")
            val_df.loc[-1]=[round(loss, 4), round(accuracy, 3), round(f1, 3), round(precision, 3), round(recall, 3), filename]
            val_df.to_csv(model_save_path + "/train_result" + "/cross_val.csv", index = False)
            
            model = tf.keras.models.load_model(filename, compile=False)
            y_prob = model.predict(x=(b1_x_val, b2_x_val, b3_x_val))
            y_pred = np.argmax(y_prob, axis = 1)
            
            plot_confusion_matrix(model_save_path, y_val, y_pred, fold_number)
            plot_roc_curve(y_val, y_prob, model_save_path, fold_number)
            
            fold_number += 1
            
        cross_val_df = pd.read_csv(model_save_path + '/train_result/cross_val.csv')
        export_train_result(model_name, cross_val_df, model_save_path)    
    
    
def main():
    with open("./train-config.yaml", "r") as f:
        data = yaml.full_load(f)
        
    train_path = data["train_path"]
    learning_rate = data["learning_rate"]
    epoch = data["epoch"]
    batch_size = data["batch_size"]
    seed = data["seed"]
    model_list = data["model_list"]
    # print(model_name)
    seed_fix(seed)
    train(train_path, learning_rate, epoch, batch_size, seed, model_list)

    return 0    

if __name__ == '__main__':
    # config = parse_opt()
    main()