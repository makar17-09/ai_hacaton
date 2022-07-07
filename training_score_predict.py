import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

import datetime
import re
import os
from tensorflow import keras


from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


class PredictData:


    def __init__(self, train, test):
        data_train = pd.read_csv(train) #открываем файл с данными для обучения
        data_test = pd.read_csv(test) #открываем файл с данными для подготовки файла .csv
        self.data_train = data_train.loc[:, 'Smiles':]
        self.data_test = data_test.loc[:, 'Smiles':]
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.y_predict = []
        self.predict_for_csv = []
        self.model = []
        ##########################################
        #######                            #######
        #######Здесь можно менять параметры#######
        #######                            #######
        ##########################################
        self.trh = 0.1 #не больше 0,4 и не меньше 0,001. Это порог для определения активности
        self.batch_size=2048 #сколько данных будет просчитываться за одну эпоху
        self.epochs=600 #кол-во эпох
        #######№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№

    def file_name(self, prefix, ext='csv'):

        date = datetime.datetime.now()
        postfix = '{:02d}{:02d}{:02d}{:02d}{:02d}'.format(date.month, date.day, date.hour, date.minute, date.second)

        return '{}_{}.{}'.format(prefix, postfix, ext)


    def data_conversion(self, metod, n_components=1):

        name_col = self.data_train.columns.values # получим наименования столбцов, из них удалим первый и второй, т.к. первый в обучение
        data_col = np.delete(name_col, [0, 1]) #не подходит, а второй это правильные метки
        X = self.data_train[data_col]

        r = []
        for a in self.data_train["Active"]:
            r.append(1 if a else 0)
        y = pd.Series(r, copy=False) #правильные метки должны быть 0 и 1, а не True и False


        if metod == "PCA":

            conv = PCA(n_components=n_components, random_state=42)

        elif metod == "NCA":

            conv = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)

        elif metod == "SS":

            conv = StandardScaler()

        else:

            print("metod is not found")
            conv = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)

        conv.fit(X, y)

        data_train_processed = conv.transform(X)
        data_test_processed = conv.transform(self.data_test[data_col])

        self.X_train, self.X_test, self.y_train, self.y_test =\
        train_test_split(data_train_processed, y, test_size=0.3, stratify=y)

        return data_test_processed


    def create_models_ml(self, model, max_depth=2, n_estimators=100, iterations=1):

        if model == "LogisticRegression":

            self.model = LogisticRegression(penalty='l2', random_state=42)

        elif model == "RandomForestClassifier":

            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        elif model == "GradientBoostingClassifier":

            self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        elif model == "CatBoostClassifier":

            self.model = CatBoostClassifier(iterations=iterations, learning_rate=1, depth=max_depth)


        elif model == "AdaBoostClassifier":

            self.model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)

        else:

            print("model is not found")
            self.model = CatBoostClassifier(iterations=iterations, learning_rate=1, depth=max_depth)


        self.model.fit(self.X_train, self.y_train)

        print(f"----------------------------model {model} ---------------------------")



    def create_first_model_nn(self, vr):


        X_train = self.X_train
        y_train = self.y_train

        if vr=="1":

            self.model = keras.Sequential(
                [
                    keras.layers.Dense(
                        X_train.shape[-1], activation="relu", input_shape=(X_train.shape[-1],)
                    ),
                    keras.layers.Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
                    keras.layers.Dense(450, activation="relu"),
                    keras.layers.Dense(200, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-5)),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )

        elif vr=="2":

            self.model = keras.Sequential(
                [
                    keras.layers.Dense(
                        X_train.shape[-1], activation="relu", input_shape=(X_train.shape[-1],)
                    ),
                    keras.layers.Dense(400, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                    keras.layers.Dense(400, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-3, l2=1e-2)),
                    keras.layers.Dense(400, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
    

        elif vr=="3":

            self.model = keras.Sequential(
                [
                    keras.layers.Dense(
                        X_train.shape[-1], activation="relu", input_shape=(X_train.shape[-1],)
                    ),
                    keras.layers.Dense(555, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
                    keras.layers.Dense(444, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-5)),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )


        elif vr=="4":

            self.model = keras.Sequential(
                [
                    keras.layers.Dense(
                        X_train.shape[-1], activation="relu", input_shape=(X_train.shape[-1],)
                    ),
                    keras.layers.Dense(500, activation="relu"),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(444, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )


        elif vr=="5":

            self.model = keras.Sequential(
                [
                    keras.layers.Dense(
                        X_train.shape[-1], activation="relu", input_shape=(X_train.shape[-1],)
                    ),
                    keras.layers.Dense(600, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(800, activation="relu"),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(800, activation="relu"),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )


        elif vr=="6":

            self.model = keras.Sequential(
                [
                    keras.layers.Dense(
                        X_train.shape[-1], activation="relu", input_shape=(X_train.shape[-1],)
                    ),
                    keras.layers.Dense(1000, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                    keras.layers.Dense(3000, activation="relu"),
                    keras.layers.Dense(1500, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-3)),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )

        ##########################################

        metrics = [
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]



        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
        )



        self.model.fit(
            X_train,
            y_train,
            batch_size = self.batch_size,
            epochs = self.epochs,
        )



    def predict_and_preproces(self):

        y_pred = self.model.predict(self.X_test)

        for y in y_pred:
            int_y = 1 if y > self.trh else 0
            self.y_predict.append(int_y)

        return self.y_predict



    def get_accuracy_score(self):
        return accuracy_score(self.y_test, self.y_predict)


    def get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_predict)


    def get_f1_score(self):

        return f1_score(self.y_test, self.y_predict, average='weighted')


    def mae(self):
        return mean_squared_error(self.y_test, self.y_predict)


    def get_predict_on_test(self, new_test_data):

        y_pred = self.model.predict(new_test_data)

        for y in y_pred:
            int_y = 1 if y > self.trh else 0
            self.predict_for_csv.append(int_y)


    def save_result(self):

        file = self.file_name("submission")
        with open(file, 'w') as dst:
            dst.write('Smiles,Active\n')
            for path, score in zip(self.data_test["Smiles"], self.predict_for_csv):
                dst.write(f'{path},{score}\n')
        return file

    def save_model(self):

        path = os.path.join("models", self.file_name("model", "h5"))
        self.model.save(path)
        return path


