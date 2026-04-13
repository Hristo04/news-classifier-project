import random

categories = {
    "sport": [
        "Отборът спечели важен мач",
        "Състезателят постигна нов рекорд",
        "Треньорът подготвя отбора за финала"
    ],
    "politics": [
        "Парламентът прие нов закон",
        "Министърът обяви нови мерки",
        "Правителството обсъжда бюджета"
    ],
    "technology": [
        "Компанията представи нов софтуер",
        "Разработчиците създадоха нова система",
        "Учени разработват изкуствен интелект"
    ]
}

with open("news_dataset.csv", "w", encoding="utf-8") as f:
    f.write("category,text\n")

    for _ in range(1000):
        category = random.choice(list(categories.keys()))
        text = random.choice(categories[category])
        f.write(f'{category},"{text}."\n')

print("Готово! Създадени са 1000 статии.")