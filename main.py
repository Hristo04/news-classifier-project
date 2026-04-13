import re
import queue
import threading
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def preprocess_text(text: str) -> str:
    """
    Почиства текста:
    - прави го с малки букви
    - маха специални символи
    - премахва излишните интервали
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_parallel(texts: list[str], max_workers: int = 4) -> list[str]:
    """
    Паралелна предварителна обработка на текстовете.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        cleaned = list(executor.map(preprocess_text, texts))
    return cleaned


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Зарежда CSV файл с колони:
    category,text

    Поддържа:
    - нормални нови редове
    - буквални '\\n'
    - слепени редове с '$'
    """
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError("CSV файлът е празен.")

    raw = raw.replace("\\n", "\n")
    raw = raw.replace("$", "\n")

    df = pd.read_csv(StringIO(raw), sep=",", engine="python")
    df.columns = [str(col).strip().replace('"', "") for col in df.columns]

    required_columns = {"category", "text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV файлът трябва да съдържа колони: category,text. Намерени: {list(df.columns)}"
        )

    df = df.dropna(subset=["category", "text"]).copy()
    return df


def train_model(csv_path: str):
    """
    Обучава модел за класификация.
    """
    df = load_dataset(csv_path)

    texts = df["text"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()

    print("Паралелна обработка на текстовете...")
    cleaned_texts = preprocess_parallel(texts, max_workers=4)

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=0.33,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vec, y_train)

    predictions = model.predict(x_test_vec)

    print("\n=== РЕЗУЛТАТИ ОТ ОБУЧЕНИЕТО ===")
    print(f"Точност: {accuracy_score(y_test, predictions):.4f}")
    print("\nКласификационен отчет:")
    print(classification_report(y_test, predictions))

    return model, vectorizer


def pipeline_classification(new_articles: list[str], model, vectorizer, output_file: str = "results.txt"):
    """
    Конвейеризация (pipeline) с 4 етапа:
    1. Подаване на статии
    2. Предварителна обработка
    3. Класификация
    4. Запис на резултатите
    """
    q_input = queue.Queue()
    q_preprocessed = queue.Queue()
    q_classified = queue.Queue()

    results = []

    def stage_input():
        for article in new_articles:
            q_input.put(article)
        q_input.put(None)

    def stage_preprocess():
        while True:
            item = q_input.get()
            if item is None:
                q_preprocessed.put(None)
                break
            cleaned = preprocess_text(item)
            q_preprocessed.put((item, cleaned))

    def stage_classify():
        while True:
            item = q_preprocessed.get()
            if item is None:
                q_classified.put(None)
                break

            original_text, cleaned_text = item
            vectorized = vectorizer.transform([cleaned_text])
            predicted_category = model.predict(vectorized)[0]
            q_classified.put((original_text, predicted_category))

    def stage_save():
        with open(output_file, "w", encoding="utf-8") as f:
            while True:
                item = q_classified.get()
                if item is None:
                    break

                original_text, predicted_category = item
                line = f"Категория: {predicted_category}\nТекст: {original_text}\n{'-' * 60}\n"
                f.write(line)
                results.append((original_text, predicted_category))

    t1 = threading.Thread(target=stage_input)
    t2 = threading.Thread(target=stage_preprocess)
    t3 = threading.Thread(target=stage_classify)
    t4 = threading.Thread(target=stage_save)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    return results


def main():
    csv_path = "news_dataset.csv"

    try:
        model, vectorizer = train_model(csv_path)

        print("\n=== PIPELINE КЛАСИФИКАЦИЯ НА НОВИ СТАТИИ ===")
        new_articles = [
            "Отборът спечели важен мач след драматичен обрат през второто полувреме.",
            "Парламентът прие нови промени в държавния бюджет след дълги дебати.",
            "Нова технологична компания представи изкуствен интелект за обработка на текст."
        ]

        results = pipeline_classification(new_articles, model, vectorizer)

        print("\nРезултати от новите статии:")
        for text, category in results:
            print(f"[{category}] {text}")

        print("\nГотово. Резултатите са записани във файла results.txt")

    except FileNotFoundError:
        print("Грешка: Файлът news_dataset.csv не е намерен.")
    except Exception as e:
        print(f"Възникна грешка: {e}")


if __name__ == "__main__":
    main()