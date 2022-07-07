from training_score_predict import PredictData


def main(train, test, metod, vr="1", ml=True, n_components=1, model="CatBoostClassifier", max_depth=2, n_estimators=100, iterations=1):

    #1. открываем данные
    sample = PredictData(train, test)
    print(f"-----------Данные {train} и {test} были получены---------------")
    if ml:
        print(f"Это метод мо: взяли модель {model}\n на данных {train} {test}\n преобразовали методом {metod} {n_components}\n\
                    с параметрами max_depth={max_depth}\n, n_estimators={n_estimators}, iterations={iterations}\n")

    else:
        print(f"Рассматриваем нейросеть версии {vr} на данных {train} {test}\n преобразовали методом {metod} {n_components}")
    #2. Уберем smiles и целевую переменную. Приведем данные к стандартному распределеную или понизим размерность. Разобъем полученные данные на обучающую и тестовую выборки
    new_test_data = sample.data_conversion(metod, n_components) #new_test_data потом нужны будут для получения предсказания на test.csv

    #3. надо определиться методы ml или нейронка
    if ml:
        sample.create_models_ml(model, max_depth=max_depth, n_estimators=n_estimators, iterations=iterations)
    else:
        sample.create_first_model_nn(vr=vr)

    #4. Построим предсказания
    sample.predict_and_preproces()

    #5. Посчитаем качество получившейся белеберды
    a_score = sample.get_accuracy_score()
    c_mat = sample.get_confusion_matrix()
    f1 = sample.get_f1_score()
    loss = sample.mae()
    rec = c_mat[1][1] / (c_mat[1][0] + c_mat[1][1])

    #6. Посмотрим на получившеся оценки
    if a_score > 0.9 and f1 > 0.9 and rec > 0.4:
        print(f" Accuracy: {a_score}\nConfusion matrix:\n{c_mat}\nF1 score: {f1}\n Loss: {loss}\nRecall: {rec}\n")

        if a_score > 0.95 and f1 > 0.95 and rec > 0.25:
            print("Такое решение можно отправлять")
        else:
            print("Если хочешь, можешь отправить это решение, но результат не очень хороший")


        #7. Получим предсказания на test.csv и сохраним их
        sample.get_predict_on_test(new_test_data)
        name_submitioon = sample.save_result()


        print(f"-------------{name_submitioon} is saved--------------")

        #8. Попытаемся сохранить модель
        try:
            path = sample.save_model()
        except Exception as ex:
            print(f"Не удалось сохранить модель {ex}")
            path = model

        #9. Запишем все что имеем в логи
        file = "log.txt"
        with open(file, "a") as l:
            l.write(f"---------------model--{path}-------------------")
            if ml:
                l.write(f"Это метод мо: взяли модель {model} на данных {train} {test} преобразовали методом {metod} {n_components}\
                с параметрами max_depth={max_depth}, n_estimators={n_estimators}, iterations={iterations}\n")
            else:
                l.write(f"Обучались на нейронке версия-{vr}- на данных {train} {test}\n преобразовали методом {metod} {n_components}\n")
            l.write(f"\nAccuracy: {a_score}\nConfusion matrix: {c_mat}\nF1 score: {f1}\n Loss: {loss}\nRecall: {rec}\n\n\n")

    print("--------------------Done--------------------")




if __name__ == '__main__':


    data_tr = ["Task/train_from_rdkit.csv", "Task/train_from_md_boost.csv", "Task/train_count.csv", "Task/train_rdkit_count.csv",\
    "Task/train_md_count.csv", "Task/train_rdkit_md_boost.csv"]
    data_ts = ["Task/test_from_rdkit.csv", "Task/test_from_md_boost.csv", "Task/test_count.csv", "Task/test_rdkit_count.csv",\
    "Task/test_md_count.csv", "Task/test_rdkit_md_boost.csv"]


    metods = ["NCA", "SS"]
    models = ["LogisticRegression", "GradientBoostingClassifier", "CatBoostClassifier"]

    # for train, test in zip(data_tr, data_ts):
    #     for mtd in metods:
    #         for n_c in range(1,4):
    #             for mdl in models:
    #                 for deep in range(2,9):
    #                     for n_e in range(70, 300, 30):
    #                         try:
    #                             main(train=train, test=test, metod=mtd, n_components=n_c, model=mdl, max_depth=deep, n_estimators=n_e, iterations=1)
    #                         except Exception as ex:
    #                             print(f"Не обучить модель {mdl} на данных {train}. Ошибка: {ex}\n\n")

    version = ["1", "2", "3", "4", "5"]

    for mtd in metods:
        for n_c in range(2,4):
            for vr in version:
                try:
                    main(train="Task/train_all.csv", test="Task/test_all.csv", metod=mtd, n_components=n_c, ml=False, vr=vr)

                except Exception as ex:
                    print(f"Не обучить модель {vr} на данных {mtd} Ошибка: {ex}\n\n")


