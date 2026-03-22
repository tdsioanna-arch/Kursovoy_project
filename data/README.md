# Датасеты для курсового проекта

В этой папке хранятся только небольшие файлы с данными. Большие датасеты (временные ряды, изображения, текст, аудио) не загружаются в репозиторий из-за ограничений GitHub. Вместо этого приведены ссылки на источники и инструкции по скачиванию.

---

## 1. Табличные данные (кейс №9)

**Тема:** Разработка рекомендательной информационной системы на основе статистических данных хоккейных матчей

**Название датасета:** NHL Player Database

**Файлы в репозитории:**
- `SKATERS.csv` — данные о полевых игроках NHL (нападающие, защитники)
- `GOALIES.csv` — данные о вратарях NHL

**Источник:** [Kaggle: flynn28/nhl-player-database](https://www.kaggle.com/datasets/flynn28/nhl-player-database)

**Скачивание:**
```bash
# Через Kaggle API
kaggle datasets download -d flynn28/nhl-player-database

# Распаковка
unzip nhl-player-database.zip


Описание:

12,568 записей о полевых игроках

867 записей о вратарях

Период: с 1918 года по настоящее время

Признаки: количество игр, голы, передачи, очки, штрафные минуты, позиция, активность и др.


2. Временные ряды (кейс №33)
Тема: Распознавание событий по текстовому и визуальному описанию в видеопотоке с использованием визуально-языковых моделей

Название датасета: Human Activity Segmentation Challenge Dataset

Источник: GitHub: patrickzib/human_activity_segmentation_challenge

Скачивание:

bash
# Через Git
git clone https://github.com/patrickzib/human_activity_segmentation_challenge.git
Описание:

250 многомерных временных рядов

12 измерений (акселерометр, гироскоп)

Частота дискретизации: 50 Гц

Общая длительность записей: 10.7 часов

Активности: ходьба, бег, ожидание и другие

Подходит для задач классификации событий и сегментации временных рядов


3. Изображения (кейс №10)
Тема: Распознавание оружия и поз для стрельбы

Название датасета: Weapon Detection Dataset

Источник: Kaggle: snehilsanyal/weapon-detection-test

Скачивание:

bash
# Через Kaggle API
kaggle datasets download -d snehilsanyal/weapon-detection-test

# Распаковка
unzip weapon-detection-test.zip
Описание:

714 изображений с разметкой

9 классов оружия:

Automatic Rifle (автоматическая винтовка)

Bazooka (базука)

Handgun (пистолет)

Knife (нож)

Grenade Launcher (гранатомет)

Shotgun (дробовик)

SMG (пистолет-пулемет)

Sniper (снайперская винтовка)

Sword (меч)

Формат аннотаций: YOLO (bounding boxes)

Разбиение на train/val/test указано в metadata.csv


4. Текстовые данные (кейс №49)
Тема: Краткое изложение и тестирование загруженных документов (NLP)

Название датасета: CNN / Daily Mail Dataset for Text Summarization

Источник: Hugging Face: cnn_dailymail

Скачивание:

python
# Через библиотеку datasets
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
Описание:

286,817 пар (статья → краткое содержание) для обучения

13,368 пар для валидации

11,487 пар для тестирования

Новостные статьи CNN и Daily Mail

Задача: абстрактивная суммаризация текста

Средняя длина статьи: 766 слов

Средняя длина краткого содержания: 53 слова

Русскоязычная альтернатива (по желанию):

python
dataset = load_dataset("vanya-robot/russian-summarization-dataset")
# 158,661 пар русскоязычных новостей
5. Аудиоданные (кейс №93)
Тема: Голосовой робот-помощник на базе ИИ

Название датасета: Multilingual Speech Command Recognition Dataset

Источник: GitHub: IS2AI/Multilingual-Speech-Command-Recognition

Скачивание:

bash
# Через Git
git clone https://github.com/IS2AI/Multilingual-Speech-Command-Recognition.git
Описание:

1,625 русскоязычных голосовых команд

35 классов команд (цифры, направления, действия)

Формат: WAV, 16 кГц

Задача: распознавание голосовых команд для робота-помощника

Список команд:

Да, нет, стоп, вперед, назад, влево, вправо, вверх, вниз

Ноль, один, два, три, четыре, пять, шесть, семь, восемь, девять

Включи, выключи, тишина, и другие
