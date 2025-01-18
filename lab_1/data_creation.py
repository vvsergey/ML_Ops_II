import numpy as np
import os
import pandas as pd

data_count = 3000

def create_array(base, noise):
    """
    Сгенерируем дата-сет
    из псевдослучайных данных
    :param base: признак без шума
    :param noise: амплитуда
    :return: массив numpy
    """
    return np.array([int(base * x) +
                     int(np.random.choice([-1, 1]) * noise * np.random.rand())
                     for x in np.random.rand(data_count)])


def create_file(file_name, np_data, target_flag, column_names,
                train_dir='./train', test_dir='./test'):
    """
      Запись данных тестовой и тренировочной выборки
      в соответсвующие папки
      :param file_name: наименование файла
      :param np_data: Записоваемая выборка
      :param target_flag:  файл целевой или нет
      :param column_names: список наименований колонок
      :param train_dir: путь до тренировочной директории
      :param test_dir:  путь до тестовой директории
      :return: множество содержащие пути
      """
    if target_flag:
        df_train = pd.DataFrame(np_data[:int(data_count * 0.8)])
        df_test = pd.DataFrame(np_data[int(data_count * 0.8):])
    else:
        df_train = pd.DataFrame(np_data[:int(data_count * 0.8), :])
        df_test = pd.DataFrame(np_data[int(data_count * 0.8):, :])

    df_train.columns = column_names
    df_test.columns = column_names
    train_path = os.path.join(train_dir, file_name + "_train.csv")
    test_path = os.path.join(test_dir, file_name + "_test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    return train_path, test_path


feature_1 = create_array(20, 4)
feature_2 = create_array(1000, 27)
feature_3 = create_array(10, 2)
feature_4 = create_array(90, 7)

feat_1_feat_2 = np.stack((feature_1, feature_2), axis=1)
feat_3_feat_4 = np.stack((feature_3, feature_4), axis=1)
target = create_array(5, 2)

files = [create_file("feat_1_feat_2", feat_1_feat_2, False, ['feat_1', 'feat_2']),
         create_file("feat_3_feat_4", feat_3_feat_4, False, ['feat_3', 'feat_4']),
         create_file("target", target, True, ['target'])]

