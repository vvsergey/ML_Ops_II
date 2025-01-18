from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob
import errno
import os


def get_files(train_path='./train/', test_path='./test/'):
    """
    Получение путей файлов из тестовой и тренировочной директорий
    :param train_path: путь до тренировочной директории
    :param test_path: путь до тестовой директории
    :return: множество из двух массивов путей к файлам
    """
    train_files = glob.glob(train_path + "*.csv")
    test_files = glob.glob(test_path + "*.csv")
    return train_files, test_files


def get_dataframes():
    """
    Получение двух датафреймов (тествого и тренировочного)
    :return: множество из двух pandas датафреймов
    """
    train_files, test_files = get_files()
    train_files.sort()
    test_files.sort()


    if len(train_files) == len(test_files):
        if len(train_files) > 1:
            df_train = pd.read_csv(train_files[0])
            df_test = pd.read_csv(test_files[0])

            for i in range(1, len(train_files)):
                df_train = pd.concat([df_train, pd.read_csv(train_files[i])], axis=1)
                df_test = pd.concat([df_test, pd.read_csv(test_files[i])], axis=1)


        elif len(train_files) == 1:
            return pd.read_csv(train_files[0]), pd.read_csv(test_files[0])
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "*./test or *./train")
    else:
        raise ValueError("train data doesn't equal test_data")
    return df_train, df_test


# получаем датафреймы
train_df, test_df = get_dataframes()



# выделяем признаки и целевое значение
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']


# приводим данные (только признаки)
scaler = StandardScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)


# объединяем в единые датафреймы приведенные признаки и целевые значения
scaled_train_df = pd.concat([scaled_X_train, y_train], names=['10', '20', '30', '40', '50'], axis=1)
scaler_test_df = pd.concat([scaled_X_test, y_test], names=['10', '20', '30', '40', '50'], axis=1)
print("--- Data are reduced to a single type")



# записываем полученные датафреймы в единые файлы
# (напомню) до этого признаки и целевые значения были в отдельных файлах
train_path = os.path.join("./train", "result_train.csv")
test_path = os.path.join("./test", "result_test.csv")

if os.path.exists(train_path):
    os.remove(train_path)
    print("Train file is clean")
if os.path.exists(test_path):
    os.remove(test_path)
    print("Test file is clean")

scaled_train_df.to_csv(train_path, index=False)
scaler_test_df.to_csv(test_path, index=False)

if os.path.exists(train_path) and os.path.exists(test_path):
    print(f"--- The given data is recorded in common files: {train_path} and {test_path}")