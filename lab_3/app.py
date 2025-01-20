import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Определяем входные параметры модели
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()

# Загружаем датасет
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=37)

# Записываем датасет в CSV
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.to_csv('datasets/iris_dataset.csv', index=False)

# Выполняем пред обработку данных
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=37)
model.fit(X_train_scaled, y_train)

# Записываем модель в файл
with open('model/iris_model.pkl', 'wb') as file:
    pickle.dump((model, scaler), file)

# Загружаем модель из файла
with open('model/iris_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

# Сопоставляем индексы с названиями классов
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}


# Создаем эндпоинт для классификации ирисов
@app.post("/predict/")
async def predict(item: IrisInput):
    input_data = [[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]]
    input_data_scaled = scaler.transform(input_data)
    prediction_index = model.predict(input_data_scaled)[0]
    prediction_class = class_names[prediction_index]
    return {"prediction": prediction_class}


# Создаем эндпоинт возвращающий информационное сообщение
@app.get("/")
async def get():
    return {
        "message": "For iris classification, send a POST request to the /predict endpoint.",
        "example_body": {
            "sepal_length": 1.6,
            "sepal_width": 4.4,
            "petal_length": 1.4,
            "petal_width": 3.6
        }
    }