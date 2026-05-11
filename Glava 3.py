import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob
from matplotlib.patches import Rectangle


PURPLE = '#9b59b6'
DARK_PURPLE = '#8e44ad'
LIGHT_PURPLE = '#c39bd3'

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)


DATA_PATH = r"Sohas_weapon-Detection"
print(f"Путь к датасету: {DATA_PATH}")



# Ищем XML в папке annotations/xmls
xml_train = glob.glob(os.path.join(DATA_PATH, 'annotations', 'xmls', '*.xml'))
xml_test = glob.glob(os.path.join(DATA_PATH, 'annotations_test', 'xmls', '*.xml'))
all_xml = xml_train + xml_test

print(f"Найдено XML (train): {len(xml_train)}")
print(f"Найдено XML (test): {len(xml_test)}")
print(f"Всего XML: {len(all_xml)}")

# Ищем изображения
images_train = glob.glob(os.path.join(DATA_PATH, 'images', '*.jpg'))
images_train += glob.glob(os.path.join(DATA_PATH, 'images', '*.png'))
images_test = glob.glob(os.path.join(DATA_PATH, 'images_test', '*.jpg'))
images_test += glob.glob(os.path.join(DATA_PATH, 'images_test', '*.png'))
all_images = images_train + images_test

print(f"Найдено изображений (train): {len(images_train)}")
print(f"Найдено изображений (test): {len(images_test)}")
print(f"Всего изображений: {len(all_images)}")

if len(all_xml) == 0:
    print("ОШИБКА: XML файлы не найдены!")
    print("Проверьте путь и структуру папок")
    exit(1)


def parse_voc_xml(xml_path):
    objects = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Размер изображения
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text) if size.find('width') is not None else 1
            height = int(size.find('height').text) if size.find('height') is not None else 1
        else:
            width, height = 1, 1

        # Объекты
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Нормализованные координаты
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            width_norm = (xmax - xmin) / width
            height_norm = (ymax - ymin) / height

            objects.append({
                'class_name': name,
                'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                'x_center': x_center, 'y_center': y_center,
                'width': width_norm, 'height': height_norm,
                'img_width': width, 'img_height': height
            })
    except Exception as e:
        print(f"Ошибка в {xml_path}: {e}")
    return objects



all_objects = []
print("\nПарсинг XML файлов...")
for i, xml_file in enumerate(all_xml):
    objects = parse_voc_xml(xml_file)
    all_objects.extend(objects)
    if (i + 1) % 100 == 0:
        print(f"  Обработано {i + 1} из {len(all_xml)}")

df = pd.DataFrame(all_objects)
print(f"\nВсего объектов (bounding boxes): {len(df)}")
print(f"Среднее объектов на XML: {len(df) / len(all_xml):.2f}")


if len(df) > 0:
    class_counts = df['class_name'].value_counts()
    print(f"\n--- КЛАССЫ (всего {len(class_counts)}) ---")
    for name, count in class_counts.items():
        print(f"  {name}: {count} объектов")

    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_counts.index, class_counts.values,
                  color=PURPLE, edgecolor=DARK_PURPLE)
    ax.set_title('Распределение объектов по классам', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 5,
                f'{int(h)}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'annotation_distribution.png'), dpi=150)
    plt.close()
    print("\nСохранен: annotation_distribution.png")

    # Размеры bbox
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['width'], bins=30, color=PURPLE, edgecolor=DARK_PURPLE, alpha=0.7)
    axes[0].set_title('Распределение ширины bbox')
    axes[1].hist(df['height'], bins=30, color=LIGHT_PURPLE, edgecolor=DARK_PURPLE, alpha=0.7)
    axes[1].set_title('Распределение высоты bbox')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'bbox_sizes.png'), dpi=150)
    plt.close()
    print("Сохранен: bbox_sizes.png")
else:
    print("Нет объектов для анализа")


print("\n")
print("ПРИМЕРЫ ИЗОБРАЖЕНИЙ")


# Создаем словарь: класс -> путь к изображению
sample_by_class = {}
for xml_file in all_xml:
    base_name = os.path.splitext(os.path.basename(xml_file))[0]

    # Ищем изображение в папках images или images_test
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        cand_train = os.path.join(DATA_PATH, 'images', base_name + ext)
        cand_test = os.path.join(DATA_PATH, 'images_test', base_name + ext)
        if os.path.exists(cand_train):
            img_path = cand_train
            break
        if os.path.exists(cand_test):
            img_path = cand_test
            break

    if img_path:
        # Парсим классы из этого XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in sample_by_class:
                sample_by_class[class_name] = img_path
                break

    if len(sample_by_class) >= 9:
        break

# Визуализация
if sample_by_class:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, (class_name, img_path) in enumerate(sample_by_class.items()):
        if i < 9:
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(class_name, fontsize=10)
                axes[i].axis('off')
            except Exception as e:
                axes[i].set_title(f'{class_name}\n(ошибка)', fontsize=10)
                axes[i].axis('off')

    for j in range(len(sample_by_class), 9):
        axes[j].axis('off')

    plt.suptitle('Примеры изображений для каждого класса', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sample_images.png'), dpi=150)
    plt.close()
    print("Сохранен: sample_images.png")


print("\n")
print("Визуализация Bouding boxes")


def draw_bboxes(image_path, xml_path):
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, class_name,
                    fontsize=10, color='red', weight='bold')
    except Exception as e:
        print(f"Ошибка: {e}")

    ax.set_title(f'Детекция: {os.path.basename(image_path)}')
    ax.axis('off')
    return fig


# Собираем все XML и все изображения
all_xml = []
all_xml += glob.glob(os.path.join(DATA_PATH, 'annotations', 'xmls', '*.xml'))
all_xml += glob.glob(os.path.join(DATA_PATH, 'annotations_test', 'xmls', '*.xml'))

all_images = []
all_images += glob.glob(os.path.join(DATA_PATH, 'images', '*.jpg'))
all_images += glob.glob(os.path.join(DATA_PATH, 'images', '*.png'))
all_images += glob.glob(os.path.join(DATA_PATH, 'images_test', '*.jpg'))
all_images += glob.glob(os.path.join(DATA_PATH, 'images_test', '*.png'))

print(f"Найдено XML: {len(all_xml)}")
print(f"Найдено изображений: {len(all_images)}")

# Ищем ПЕРВУЮ ПАРУ (похожие имена, не обязательно точное совпадение)
found = False
for xml_path in all_xml:
    base_xml = os.path.splitext(os.path.basename(xml_path))[0]  # ABbframe00154

    for img_path in all_images:
        base_img = os.path.splitext(os.path.basename(img_path))[0]  # ABbframe00154

        # Если имена начинаются одинаково (хотя бы первые 5 символов)
        if base_xml[:5] == base_img[:5] or base_xml in base_img or base_img in base_xml:
            print(f"Найдена пара: {os.path.basename(xml_path)} ↔ {os.path.basename(img_path)}")
            draw_bboxes(img_path, xml_path)
            plt.savefig(os.path.join(figures_dir, 'bbox_example.png'), dpi=150)
            plt.close()
            print("Сохранен: figures/bbox_example.png")
            found = True
            break
    if found:
        break

if not found:
    print("Не найдено ни одной подходящей пары XML+изображение")
    print("Проверьте структуру файлов:")
    print("  XML:", [os.path.basename(f) for f in all_xml[:5]])
    print("  JPG:", [os.path.basename(f) for f in all_images[:5]])


print("\n")
print("Выводы по третьему разделу")

print(f"""
1. Характеристика набора:
   - XML файлов: {len(all_xml)}
   - Изображений: {len(all_images)}
   - Всего объектов: {len(df)}

2. Пригодность данных:
   - Датасет пригоден для детекции оружия
   - Формат: Pascal VOC (XML)

3. Рекомендации:
   - Аугментация для увеличения выборки
   - Конвертация XML в YOLO формат
""")

print("АНАЛИЗ ЗАВЕРШЕН")


