import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from natasha import Segmenter, MorphVocab, Doc
import csv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import os
from datetime import datetime


token = os.getenv("PATH_FINDER_TOKEN")

# Инициализация компонентов Natasha для контентной фильтрации
segmenter = Segmenter()
morph_vocab = MorphVocab()

# Хранилище состояний пользователя
USER_DATA = {}

# Клавиатура для обратной связи
FEEDBACK_KEYBOARD = ReplyKeyboardMarkup(
    [["1", "2", "3", "4", "5"]], one_time_keyboard=True, resize_keyboard=True
)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CSV_FILE = f"feedback/feedback_{current_time}.csv"

# Инициализация CSV-файла
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["student_id", "interests", "recommendations", "satisfaction", "relevance"])


# Инициализация модели для эмбеддингов
def model_content():
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Загрузка данных из Excel
    df = pd.read_excel("src/courses_combined_data.xlsx")  # Замените на правильное имя файла
    df['text'] = df['Название на Отзывусе'] + ' ' + df['Образовательный результат'] + ' ' + df['Полное описание']
    df['text'] = df['text'].apply(preprocess_text_natasha)
    df['embeddings'] = list(model.encode(df['text'], show_progress_bar=True))
    return model, df


# Функция для предварительной обработки текста
def preprocess_text_natasha(text):
    if not isinstance(text, str):
        return ''
    doc = Doc(text)
    doc.segment(segmenter)
    tokens = [token.text.lower() for token in doc.tokens if token.text.isalpha()]
    return ' '.join(tokens)


def model_colab():
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
    return svd_model


# Функция для предсказания
def predict_for_new_student(model_content, df, svd_model, actual_el, input_el, user_query, student_id="new_student"):
    # Обработка запроса пользователя для контентной фильтрации
    user_query_processed = preprocess_text_natasha(user_query)
    query_embedding = model_content.encode(user_query_processed)

    # Вычисление косинусного сходства для контентной фильтрации
    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])

    # Преобразуем строку в список пройденных элективов
    completed_electives = [e.strip() for e in input_el.split(",") if e.strip()]

    # Генерация рекомендаций на основе коллаборативной фильтрации
    recommendations = []
    for elective in df['Название на Отзывусе']:
        # Исключаем уже пройденные элективы и предлагаем только актуальные элективы
        if elective not in completed_electives and elective in actual_el:
        # if elective not in completed_electives:
            prediction_svd = svd_model.predict(student_id, elective).est

            # Косинусное сходство для контентной фильтрации
            elective_row = df[df['Название на Отзывусе'] == elective]
            elective_embedding = elective_row['embeddings'].values[0]
            similarity_to_query = cosine_similarity([query_embedding], [elective_embedding])[0][0]

            # Комбинированная оценка (контентная фильтрация + коллаборативная фильтрация)
            final_score = 0.2 * prediction_svd + 0.8 * similarity_to_query

            recommendations.append((elective, final_score))

    # Сортируем рекомендации по комбинированной оценке
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations


# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    student_id = update.message.chat_id

    # Сброс данных пользователя для нового запроса
    if student_id in USER_DATA:
        del USER_DATA[student_id]
    else:
        await update.message.reply_text(
            "Добро пожаловать!"
        )

    await update.message.reply_text(
        "Напишите о своих интересах, чтобы мы могли подобрать для вас подходящие элективы."
    )


# Обработка сообщений от пользователя
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    student_id = update.message.chat_id
    user_input = update.message.text

    # Проверяем, есть ли данные о моделях
    model_content_data = context.bot_data.get("model_content")
    model_colab_model = context.bot_data.get("svd_model")
    actual_el = context.bot_data.get("actual_el")

    if not model_content_data or not model_colab_model or not actual_el:
        await update.message.reply_text("Ошибка: модели не инициализированы.")
        return

    # Инициализация данных для нового пользователя
    if student_id not in USER_DATA:
        USER_DATA[student_id] = {}

    # Если пользователь ввёл запрос на подбор элективов
    if "interests" not in USER_DATA[student_id]:
        USER_DATA[student_id]["interests"] = user_input
        await update.message.reply_text(
            'Пожалуйста, введите элективы, которые вы уже прошли, через запятую (если у вас ещё не было элективов отправьте "."):'
        )
        USER_DATA[student_id]["awaiting_completed"] = True
        return

    # Если бот ожидает список пройденных элективов
    elif USER_DATA[student_id].get("awaiting_completed"):
        USER_DATA[student_id]["completed_electives"] = user_input
        del USER_DATA[student_id]["awaiting_completed"]

        # Генерация рекомендаций
        completed_electives = USER_DATA[student_id]["completed_electives"]
        interests = USER_DATA[student_id]["interests"]

        recommendations = predict_for_new_student(
            model_content_data["model"],
            model_content_data["df"],
            model_colab_model,
            actual_el,
            completed_electives,
            interests,
            student_id=str(student_id)
        )
        USER_DATA[student_id]["recommendations"] = recommendations

        # Формируем текст ответа с рекомендациями
        response_text = f"На основе ваших интересов: {interests}\nВот рекомендованные элективы:\n"
        for i, (elective, score) in enumerate(recommendations[:5], 1):  # Топ-5 рекомендаций
            response_text += f"{i}. {elective}\n"

        await update.message.reply_text(response_text)
        await update.message.reply_text(
            "Оцените, насколько вы довольны подобранными элективами (1-5).",
            reply_markup=FEEDBACK_KEYBOARD,
        )
        return

    # Если бот ожидает первую оценку
    elif "satisfaction" not in USER_DATA[student_id]:
        if user_input in ["1", "2", "3", "4", "5"]:
            USER_DATA[student_id]["satisfaction"] = int(user_input)
            await update.message.reply_text(
                "Спасибо! Теперь оцените, насколько подбор соответствует вашему запросу (1-5).",
                reply_markup=FEEDBACK_KEYBOARD,
            )
        else:
            await update.message.reply_text(
                "Пожалуйста, введите число от 1 до 5.", reply_markup=FEEDBACK_KEYBOARD
            )
        return

    # Если бот ожидает вторую оценку
    elif "relevance" not in USER_DATA[student_id]:
        if user_input in ["1", "2", "3", "4", "5"]:
            USER_DATA[student_id]["relevance"] = int(user_input)
            await update.message.reply_text(
                "Что вам понравилось или не понравилось в работе бота? Напишите, что можно улучшить:"
            )
            USER_DATA[student_id]["awaiting_feedback"] = True
        else:
            await update.message.reply_text(
                "Пожалуйста, введите число от 1 до 5.", reply_markup=FEEDBACK_KEYBOARD
            )
        return

    # Если бот ожидает отзыв в свободной форме
    elif USER_DATA[student_id].get("awaiting_feedback"):
        USER_DATA[student_id]["feedback"] = user_input
        del USER_DATA[student_id]["awaiting_feedback"]

        # Сохраняем данные в CSV
        save_feedback_to_csv(student_id, USER_DATA[student_id])

        # Отправляем итоговое сообщение
        await update.message.reply_text(
            "Спасибо за обратную связь! Вот ваши оценки:\n"
            f"Удовлетворённость: {USER_DATA[student_id]['satisfaction']}/5\n"
            f"Соответствие запросу: {USER_DATA[student_id]['relevance']}/5\n"
            f"Ваш отзыв: {USER_DATA[student_id]['feedback']}\n"
            "Если хотите начать заново, введите /start."
        )
        return




# Сохранение данных в CSV
# def save_feedback_to_csv(student_id, data):
#     with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             student_id,
#             data["interests"],
#             "; ".join([f"{rec[0]} ({rec[1]:.2f})" for rec in data["recommendations"][:5]]),
#             data["satisfaction"],
#             data["relevance"],
#         ])
def save_feedback_to_csv(student_id, data):
    file_path = "feedback.csv"
    fieldnames = ["student_id", "satisfaction", "relevance", "feedback"]
    write_header = not os.path.exists(file_path)

    with open(file_path, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({
            "student_id": student_id,
            "satisfaction": data["satisfaction"],
            "relevance": data["relevance"],
            "feedback": data.get("feedback", ""),
        })

def main():
    application = ApplicationBuilder().token(token).build()

    # Чтение данных об актуальных элективах
    actual_el = pd.read_csv("src\list_of_actual_electives.csv", delimiter=';', header=None)
    application.bot_data["actual_el"] = list(actual_el.iloc[0])  # Список актуальных элективов

    # Инициализация моделей
    print("Инициализация моделей...")
    model_content_inst, df = model_content()
    application.bot_data["model_content"] = {"model": model_content_inst, "df": df}
    application.bot_data["svd_model"] = model_colab()

    print("Модели успешно инициализированы!")

    # Обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    application.run_polling()



if __name__ == '__main__':
    main()
