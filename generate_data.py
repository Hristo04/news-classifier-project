def generate_initial_dataset(csv_path: str, count: int = 1200):
    """
    Генерира начален файл с данни, ако такъв не съществува или е твърде малък.
    """
    categories = {
        "sport": ["отборът", "футболен мач", "шампионат", "победа", "треньор", "играч", "стадион"],
        "politics": ["парламентът", "избори", "закон", "правителство", "дебати", "политици", "министър"],
        "technology": ["софтуер", "изкуствен интелект", "хардуер", "смартфон", "иновации", "роботи", "данни"]
    }

    data = []
    for _ in range(count):
        cat = random.choice(list(categories.keys()))
        # Генерираме малко по-разнообразни изречения за по-добро обучение
        templates = [
            f"{random.choice(categories[cat])} беше основната тема на днешния ден.",
            f"Вчера обсъждаха новия {random.choice(categories[cat])}.",
            f"Експерти анализират важния {random.choice(categories[cat])} в детайли.",
            f"Очаква се развитие около текущия {random.choice(categories[cat])}."
        ]
        data.append([cat, random.choice(templates)])

    df = pd.DataFrame(data, columns=["category", "text"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f" Успешно генерирани {count} записа в {csv_path}")