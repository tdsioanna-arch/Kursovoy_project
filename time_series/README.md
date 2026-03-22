# Датасеты для курсового проекта

В этой папке хранятся только небольшие файлы с данными. Большие датасеты загружаются отдельно по ссылкам на источники.

---

## 1. Табличные данные (NHL Player Database)

**Файлы:**
- `SKATERS.csv` — данные о полевых игроках NHL
- `GOALIES.csv` — данные о вратарях NHL

**Источник:** [NHL Player Database on Kaggle](https://www.kaggle.com/datasets/flynn28/nhl-player-database)

**Скачивание:**
```bash
# Через Kaggle API
kaggle datasets download -d flynn28/nhl-player-database

## 2. Временные ряды (Human Activity Segmentation)
Название: Human Activity Segmentation Challenge Dataset

Источник: GitHub: patrickzib/human_activity_segmentation_challenge

Скачивание:

bash
# Через Git
git clone https://github.com/patrickzib/human_activity_segmentation_challenge.git
Описание: Многомерные временные ряды с датчиков движения (акселерометр). Частота дискретизации 50 Гц. Подходит для задач распознавания событий (ходьба, бег, ожидание).
