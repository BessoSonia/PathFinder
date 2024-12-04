import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle


# Чтение данных о пройденных элективах
X = pd.read_csv("src\electives_spring_2023_all.csv", delimiter=';')  # Данные
X = X.loc[:, ['Студент ФИО', 'РМУП 1 название', 'РМУП 2 название', 'РМУП 3 название', 'РМУП 4 название']]
X.fillna('', inplace=True)

# Список всех уникальных элективов
electives = X.drop('Студент ФИО', axis=1).values.flatten()
electives = [e for e in electives if e != '']  # Убираем пустые строки
electives = list(set(electives))  # Получаем уникальные элективы

# Создаем бинарную матрицу для коллаборативной фильтрации
student_electives = []

for _, row in X.iterrows():
    student_elective = [1 if elective in row.values else 0 for elective in electives]
    student_electives.append(student_elective)

X_binary = pd.DataFrame(student_electives, columns=electives)
X_binary.insert(0, 'student', X['Студент ФИО'])

# Преобразуем данные для работы с библиотекой Surprise
reader = Reader(rating_scale=(0, 1))
df_melted = X_binary.melt(id_vars='student', var_name='elective', value_name='completed')
data = Dataset.load_from_df(df_melted[['student', 'elective', 'completed']], reader)
trainset, testset = train_test_split(data, test_size=0.1, random_state=42)

# Инициализация и обучение модели коллаборативной фильтрации (SVD)
svd_model = SVD()
svd_model.fit(trainset)

# Сохранение обученной модели SVD
with open("bot/svd_model.pkl", "wb") as f:
    pickle.dump(svd_model, f)