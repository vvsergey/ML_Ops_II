import pandas as pd
import os


# Создание переменной с путем корневой директории lab4
dirname = os.path.dirname(os.path.dirname(__file__))

# Загрузка датасета
df = pd.read_csv(os.path.join(dirname, 'datasets/titanic_data.csv'))

# Применение one-hot-encoding для признака "Sex"
sex_one_hot = pd.get_dummies(df['Sex'], prefix='Sex')

# Добавление новых признаков в исходный DataFrame
df = pd.concat([df, sex_one_hot], axis=1)

# Сохранение новой версии датасета
df.to_csv(os.path.join(dirname, 'datasets/titanic_transform_data.csv'), index=False)