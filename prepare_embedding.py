import pandas as pd
import numpy as np
from natasha import Segmenter, MorphVocab, Doc
from sentence_transformers import SentenceTransformer

# Инициализация компонентов Natasha для контентной фильтрации
segmenter = Segmenter()
morph_vocab = MorphVocab()

# Инициализация языковой модели для создания эмбеддингов
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # Поддерживает русский и английский языки

# Функция для предварительной обработки текста
def preprocess_text_natasha(text):
    if not isinstance(text, str):
        return ''
    doc = Doc(text)
    doc.segment(segmenter)
    tokens = [token.text.lower() for token in doc.tokens if token.text.isalpha()]
    return ' '.join(tokens)

# Загрузка данных из Excel
df = pd.read_excel("src/courses_combined_data_actual_all.xlsx")  # Замените на правильное имя файла
# df = pd.read_excel("src/courses_combined_data_actual.xlsx")  # Замените на правильное имя файла
# df = df.dropna(subset=['Название на Отзывусе'])
# df.reset_index(drop=True, inplace=True)
df['text'] = df['Название на Отзывусе'] + ' ' + df['Образовательный результат'] + ' ' + df['Полное описание']
df['text'] = df['text'].apply(preprocess_text_natasha)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# Генерация эмбеддингов для курсов
df['embeddings'] = list(model.encode(df['text'], show_progress_bar=True))

# Сохранение данных (без эмбеддингов)
df.drop(columns=['embeddings'], inplace=False).to_csv("bot/courses_data.csv", index=False)

# Сохранение эмбеддингов отдельно
np.save("bot/courses_embeddings.npy", np.vstack(df['embeddings']))

# Сохранение модели
model.save("bot/sentence_transformer_model")