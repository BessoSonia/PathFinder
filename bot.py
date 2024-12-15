import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from natasha import Segmenter, Doc
import csv
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import os
from datetime import datetime
from Levenshtein import ratio


'''
Перемнные и методы для работы чат-бота
'''
token = os.getenv("API_KEY")

# Хранилище состояний пользователя
USER_DATA = {}

# Клавиатура для обратной связи
feedback_keyboard = ReplyKeyboardMarkup(
    [["1", "2", "3", "4", "5"]], one_time_keyboard=True, resize_keyboard=True
)


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


# Функция для создания клавиатуры
def get_keyboard(current_page, total_pages):
    # Создаем кнопки "Назад" и "Вперед"
    navigation_buttons = [
        InlineKeyboardButton("⬅️ Назад", callback_data="previous_page"),
        InlineKeyboardButton("➡️ Вперед", callback_data="next_page"),
    ]

    # Кнопка "Оценить бот" вторая строка
    feedback_button = [
        InlineKeyboardButton("📝 Оценить бот", callback_data="feedback")
    ]

    # Формируем клавиатуру (первая строка для навигации, вторая для оценки)
    keyboard = [
        navigation_buttons,  # Кнопки "Назад" и "Вперед"
        feedback_button,  # Кнопка "Оценить бот"
    ]

    return InlineKeyboardMarkup(keyboard)


# Пример функции, которая отправляет или редактирует сообщение с рекомендациями
# Обработчик кнопок
async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Ответ на callback запрос (важно для Telegram API)

    student_id = query.message.chat.id

    if query.data == "feedback":
        # Когда пользователь нажимает на "Оценить бот"
        await query.message.reply_text(
            "Как вы оцениваете подбор элективов? (Оцените от 1 до 5)",
            reply_markup=feedback_keyboard
        )
        USER_DATA[student_id]["awaiting_satisfaction"] = True  # Флаг для ожидания оценки
        return

    # Обработчик для страницы вперед и назад
    current_page = USER_DATA[student_id]["current_page"]
    pages = USER_DATA[student_id]["pages"]

    if query.data == "next_page":
        if current_page < len(pages) - 1:
            USER_DATA[student_id]["current_page"] += 1
            current_page = USER_DATA[student_id]["current_page"]
    elif query.data == "previous_page":
        if current_page > 0:
            USER_DATA[student_id]["current_page"] -= 1
            current_page = USER_DATA[student_id]["current_page"]

    # Формируем текст для текущей страницы
    response_text = create_response(pages, current_page)

    # Обновляем сообщение с рекомендациями и кнопками
    await query.edit_message_text(
        text=response_text,
        reply_markup=get_keyboard(current_page, len(pages))  # Обновляем клавиатуру
    )


def create_response(pages, current_page):
    response_text = (
            "Рекомендованные элективы на основе ваших интересов:\n" +
            "\n".join(
                [
                    f"{5 * current_page + idx}. {el[0]} \n🎓Модеус: {el[2]}"
                    + (f" \n📘Отзывус: {el[3]}" if not pd.isna(el[3]) else "")
                    for idx, el in enumerate(pages[current_page], start=1)
                ]
            )
    )
    return response_text


# Функция поиска схожести
def map_electives(input_electives, db_electives, threshold=0.6):
    results = {}
    for input_elective in input_electives:
        best_match = None
        best_score = 0
        for db_elective in db_electives:
            score = ratio(input_elective.lower(), db_elective.lower())
            if score > best_score:
                best_match = db_elective
                best_score = score
        if best_score >= threshold:
            results[input_elective] = best_match
    return list(results.values())


# Обработка сообщений от пользователя
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    student_id = update.message.chat.id
    user_input = update.message.text

    # Проверяем, есть ли данные о моделях
    model_content_data = context.bot_data.get("model_content")
    model_colab_model = context.bot_data.get("svd_model")
    available_el = context.bot_data.get("available_el")

    if not model_content_data or not model_colab_model or not available_el:
        await update.message.reply_text("Ошибка: Сервис временно недоступен.")
        return

    # Инициализация данных для нового пользователя
    if student_id not in USER_DATA:
        USER_DATA[student_id] = {}

    # Если бот ожидает оценку (1-5)
    if USER_DATA[student_id].get("awaiting_rating"):
        if user_input in ["1", "2", "3", "4", "5"]:
            USER_DATA[student_id]["rating"] = int(user_input)
            del USER_DATA[student_id]["awaiting_rating"]  # Завершаем этап оценки

            # Запрашиваем отзыв
            USER_DATA[student_id]["rating_step"] = "feedback"
            await update.message.reply_text(
                "Спасибо за вашу оценку! Напишите, что вам понравилось или что можно улучшить:",
                reply_markup=None
            )
        else:
            await update.message.reply_text(
                "Пожалуйста, введите число от 1 до 5."
            )
        return

    # Если бот ожидает отзыв
    elif USER_DATA[student_id].get("rating_step") == "feedback":
        USER_DATA[student_id]["feedback"] = user_input
        del USER_DATA[student_id]["rating_step"]

        # Сохраняем данные в CSV
        save_feedback_to_csv(student_id, USER_DATA[student_id])

        # Отправляем итоговое сообщение
        await update.message.reply_text(
            "Спасибо за обратную связь!\nЕсли хотите начать заново, введите /start.",
            reply_markup=None
        )
        return

    # Если пользователь ввел запрос на подбор элективов
    if "interests" not in USER_DATA[student_id]:
        USER_DATA[student_id]["interests"] = user_input
        await update.message.reply_text(
            'Введите элективы, которые вы уже прошли, через запятую \n* если у вас ещё не было элективов, отправьте "."',
            reply_markup=None
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
            model_content=model_content_data["model"],
            df=model_content_data["df"],
            svd_model=model_colab_model,
            available_el=available_el,
            input_el=completed_electives,
            user_query=interests,
            student_id=str(student_id)
        )

        USER_DATA[student_id]["recommendations"] = recommendations

        # Разбиваем рекомендации на страницы (по 5 на странице)
        pages = []
        for i in range(0, len(recommendations), 5):
            page = recommendations[i:i + 5]
            pages.append(page)
        USER_DATA[student_id]["pages"] = pages
        USER_DATA[student_id]["current_page"] = 0

        # Отправляем первую страницу
        current_page = USER_DATA[student_id]["current_page"]
        response_text = create_response(pages, current_page)

        await update.message.reply_text(
            response_text,
            reply_markup=get_keyboard(current_page, len(pages))
        )

        # Запрашиваем оценку
        USER_DATA[student_id]["awaiting_rating"] = True
        await update.message.reply_text(
            "Пожалуйста, не забудьте оценить работу бота.",
            reply_markup=None
        )
        return

    # Если бот завершил основной сценарий и ждет новых команд
    await update.message.reply_text(
        "Неизвестная команда. Попробуйте /start для начала.",
        reply_markup=None
    )


'''
Загрузка моделей и объявление сопутствующих методов
'''
# Инициализация компонентов Natasha для контентной фильтрации
segmenter = Segmenter()

# Функция для предварительной обработки текста
def preprocess_text_natasha(text):
    if not isinstance(text, str):
        return ''
    doc = Doc(text)
    doc.segment(segmenter)
    tokens = [token.text.lower() for token in doc.tokens if token.text.isalpha()]
    return ' '.join(tokens)


# Загрузка модели для эмбеддингов
def load_model_content():
    # Загрузка данных, эмбеддингов и модели
    df = pd.read_csv("bot/courses_data.csv")
    embeddings = np.load("bot/courses_embeddings.npy")
    df['embeddings'] = list(embeddings)
    model = SentenceTransformer("bot/sentence_transformer_model")
    return model, df


# Загрузка сохранённой модели SVD
def load_model_colab():
    with open("bot/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    return svd_model


# Функция для предсказания
def predict_for_new_student(model_content, df, svd_model, available_el, input_el, user_query, student_id="new_student"):
    # Обработка запроса пользователя для контентной фильтрации
    user_query_processed = preprocess_text_natasha(user_query)
    query_embedding = model_content.encode(user_query_processed)

    # Вычисление косинусного сходства для контентной фильтрации
    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])

    # Преобразуем строку в список пройденных элективов
    completed_electives_raw = [e.strip() for e in input_el.split(",") if e.strip()]
    completed_electives = map_electives(completed_electives_raw, available_el)

    # Генерация рекомендаций на основе коллаборативной фильтрации
    recommendations = []
    for elective in df['Название на Отзывусе']:
        # Исключаем уже пройденные элективы и предлагаем только актуальные элективы
        # if elective not in completed_electives and elective in actual_el:
        if elective not in completed_electives:
            prediction_svd = svd_model.predict(student_id, elective).est

            # Косинусное сходство для контентной фильтрации
            elective_row = df[df['Название на Отзывусе'] == elective]
            elective_embedding = elective_row['embeddings'].values[0]
            similarity_to_query = cosine_similarity([query_embedding], [elective_embedding])[0][0]

            # Комбинированная оценка (контентная фильтрация + коллаборативная фильтрация)
            final_score = 0.2 * prediction_svd + 0.8 * similarity_to_query
            link_m = elective_row['Ссылка на Модеус'].values[0]
            link_o = elective_row['Ссылка на Отзывус'].values[0]

            recommendations.append((elective, final_score, link_m, link_o))

    # Сортируем рекомендации по комбинированной оценке
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations


'''
Создание CSV-файла для сохранения обратной связи
'''
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_file = f"feedback/feedback_{current_time}.csv"

# Инициализация CSV-файла
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["student_id", "interests", "recommendations", "satisfaction", "relevance"])

def save_feedback_to_csv(student_id, data):
    fieldnames = ["student_id", "satisfaction", "relevance", "feedback"]
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({
            "student_id": student_id,
            "satisfaction": data["rating"],
            "feedback": data.get("feedback", ""),
        })


'''
Основной метод
'''
def main():
    application = ApplicationBuilder().token(token).build()

    # Чтение файла Excel
    available_el = pd.read_excel("src\courses_combined_data.xlsx")["Название на Отзывусе"].dropna().tolist()
    application.bot_data["available_el"] = list(available_el)

    # Инициализация моделей
    print("Загрузка моделей...")
    model_content_inst, df = load_model_content()
    application.bot_data["model_content"] = {"model": model_content_inst, "df": df}
    application.bot_data["svd_model"] = load_model_colab()

    print("Модели успешно инициализированы!")

    # Обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_button))

    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    application.run_polling()



if __name__ == '__main__':
    main()
