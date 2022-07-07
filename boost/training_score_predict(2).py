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
        self.trh = 0.15 #не больше 0,4 и не меньше 0,001. Это порог для определения активности
        self.batch_size=1700 #сколько данных будет просчитываться за одну эпоху
        self.epochs=900 #кол-во эпох
        #######№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№

    def file_name(self, prefix, ext='csv'):

        date = datetime.datetime.now()
        postfix = '{:02d}{:02d}{:02d}{:02d}'.format(date.month, date.day, date.hour, date.minute)

        return '{}_{}.{}'.format(prefix, postfix, ext)


    def preprocessing_train(self):

        name_col = self.data_train.columns.values # получим наименования столбцов, из них удалим первый и второй, т.к. первый в обучение
        name_col = np.delete(name_col, [0, 1]) #не подходит, а второй это правильные метки

        X = self.data_train[name_col]

        r = []
        for a in self.data_train["Active"]:
            r.append(1 if a else 0)
        y = pd.Series(r, copy=False) #правильные метки должны быть 0 и 1, а не True и False

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        #разобьем выборку на обучение и тест, чтобы проверить результат в конце

    '''первая модель в которой 4 слоя и дропаут'''
    def create_first_model(self, X_shape):

        ##########################################
        #######                            #######
        #######Здесь можно менять параметры#######
        #######                            #######
        ##########################################

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    150, activation="relu", input_shape=(X_shape,)
                ),
                keras.layers.Dense(200, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-5)),
                keras.layers.Dense(500, activation="relu"),
                keras.layers.Dense(0.2),
                keras.layers.Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-5)),
                keras.layers.Dense(0.2),
                keras.layers.Dense(150, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-3)),
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



        model.compile(
            optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
        )

        return model 

    '''вторая модель в которой просто три слоя'''
    def create_second_model(self, X_shape):

        ##########################################
        #######                            #######
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!#######
        #######                            #######
        ##########################################

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    150, activation="relu", input_shape=(X_shape,)
                ),
                keras.layers.Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                keras.layers.Dense(450, activation="relu"),
                keras.layers.Dense(200, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
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



        model.compile(
            optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
        )

        return model

    '''третья модель в которой 2 скрытых слоя'''
    def create_third_model(self, X_shape):

        ##########################################
        #######                            #######
        #######Здесь можно менять параметры#######
        #######                            #######
        ##########################################

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    150, activation="relu", input_shape=(X_shape,)
                ),
                keras.layers.Dense(200, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                keras.layers.Dense(200, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
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



        model.compile(
            optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
        )

        return model


    def training(self, mod):

        X_train = self.X_train
        y_train = self.y_train

        if mod == "1":
            self.model = self.create_first_model(X_train.shape[-1])
        elif mod == "2":
            self.model = self.create_second_model(X_train.shape[-1])
        elif mod == "3":
            self.model = self.create_third_model(X_train.shape[-1])

        self.model.fit(
            X_train,
            y_train,
            batch_size = self.batch_size,
            epochs = self.epochs,
        )



    def predict_and_preproces(self):

        y_predict = []

        y_pred = self.model.predict(self.X_test)

        for y in y_pred:
            int_y = 1 if y > self.trh else 0
            y_predict.append(int_y)

        return y_predict



    def get_accuracy_score(self, y_predict):
        return accuracy_score(self.y_test, y_predict)


    def get_confusion_matrix(self, y_predict):
        return confusion_matrix(self.y_test, y_predict)


    def get_f1_score(self, y_predict):

        return f1_score(self.y_test, self.y_predict, average='weighted')


    def mae(self, y_predict):
        return mean_squared_error(self.y_test, self.y_predict)


    def get_predict_on_test(self):


        name_col = self.data_test.columns.values # получим наименования столбцов, из них удалим первый, т.к. первый для предсказаний
        name_col = np.delete(name_col, [0]) #не подходит
        X = self.data_test[name_col]
        y_pred = self.model.predict(X)

        for y in y_pred:
            int_y = 1 if y > self.trh else 0
            self.predict_for_csv.append(int_y)
        self.predict_for_csv = []


    def save_result(self):

        file = self.file_name("submission")
        with open(file, 'w') as dst:
            dst.write('Smiles,Active\n')
            for path, score in zip(self.data_test["Smiles"], self.predict_for_csv):
                dst.write(f'{path},{score}\n')

    def save_model(self):

        path = os.path.join("models", self.file_name("model", "h5"))
        self.model.save(path)
        return path



if __name__ == '__main__':

    sample = PredictData("data_by_boost.csv", "test_by_boost.csv")
    print("Данные были получены")
    sample.preprocessing_train()

    ###############################################################
    sample.training("2")
    sample.predict_and_preproces()

    a_score = sample.get_accuracy_score()
    c_mat = sample.get_confusion_matrix()
    f1 = sample.get_f1_score()
    loss = sample.mae()
    rec = c_mat[1][1] / c_mat[1][0]


    print("--------------------Done--------------------")
    print(f" Accuracy: {a_score}\nConfusion matrix:\n{c_mat}\nF1 score: {f1}\n Loss: {loss}\nRecall: {rec}\n")

    if a_score > 0.95 and f1 > 0.95 and rec > 0.25:
        print("Такое решение можно отправлять")
    else:
        print("Если хочешь, можешь отправить это решение, но результат не очень хороший")


    sample.get_predict_on_test()
    sample.save_result()


    print("-------------submission is saved--------------")


    path = sample.save_model()
    file = "log.txt"
    with open(file, "a") as l:
        l.write(f"---------------model----{path}-----------------")
        l.write(f"\nAccuracy: {a_score}\nConfusion matrix: {c_mat}\nF1 score: {f1}\n Loss: {loss}\nRecall: {rec}\n\n\n")


###############################################################
    sample.training("3")
    y_pred3 = sample.predict_and_preproces()

    a_score = sample.get_accuracy_score(y_pred3)
    c_mat = sample.get_confusion_matrix(y_pred3)
    f1 = sample.get_f1_score(y_pred3)
    loss = sample.mae(y_pred3)
    rec = c_mat[1][1] / c_mat[1][0]


    print("--------------------Done--------------------")
    print(f" Accuracy: {a_score}\nConfusion matrix:\n{c_mat}\nF1 score: {f1}\n Loss: {loss}\nRecall: {rec}\n")

    if a_score > 0.95 and f1 > 0.95 and rec > 0.25:
        print("Такое решение можно отправлять")
    else:
        print("Если хочешь, можешь отправить это решение, но результат не очень хороший")


    sample.get_predict_on_test()
    sample.save_result()


    print("-------------submission is saved--------------")


    path = sample.save_model()
    file = "log.txt"
    with open(file, "a") as l:
        l.write(f"---------------model----{path}-----------------")
        l.write(f"\nAccuracy: {a_score}\nConfusion matrix: {c_mat}\nF1 score: {f1}\n Loss: {loss}\nRecall: {rec}\n\n\n")



