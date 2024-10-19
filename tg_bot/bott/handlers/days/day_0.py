import os
import re

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import pymorphy2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from loader import dp, bot
from aiogram import types
from gigachat import GigaChat
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, kmeans_plusplus
from reportlab.pdfgen import canvas
from aiogram.dispatcher import FSMContext
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import A4
from aiogram.dispatcher.filters.state import State, StatesGroup


class FormStates(StatesGroup):
    waiting_for_file = State()

def format_pandas_table(df):
    """Форматирует DataFrame в HTML таблицу."""
    return df.to_html(index=False, border=0)

class GradSearch(object):
    def __init__(self, a=1.0, b=0.0, x0=1.0, l=1e-2):
        self.a = a
        self.b = b
        self.x0 = x0
        self.l = l
        self.loss = []

    def f(self, x, y, a, b, x0):
        return 0.5 * pow(y - a * np.exp(-x / x0) - b, 2).sum()

    def dfda(self, x, y, a, b, x0):
        return ((a * np.exp(-x / x0) + b - y) * np.exp(-x / x0)).sum()

    def dfdb(self, x, y, a, b, x0):
        return (a * np.exp(-x / x0) + b - y).sum()

    def dfdx0(self, x, y, a, b, x0):
        return ((a * np.exp(-x / x0) + b - y) * a * np.exp(-x / x0) * x / pow(x0, 2)).sum()

    def fit(self, x, y, epochs=1_000):
        self.loss.append(self.f(x, y, self.a, self.b, self.x0))
        for epoch in tqdm(range(epochs)):
            self.a -= self.l * self.dfda(x, y, self.a, self.b, self.x0)
            self.b -= self.l * self.dfdb(x, y, self.a, self.b, self.x0)
            self.x0 -= self.l* self.dfdx0(x, y, self.a, self.b, self.x0)
            self.loss.append(self.f(x, y, self.a, self.b, self.x0))

    def get_a_b_x0(self):
        return self.a, self.b, self.x0

class MrBist:
    def __init__(self):
        self.answers_data = None
        self.clean_data = None
        self.model_lemma = None
        self.bad_reg = None
        self.lemmas = None
        self.dict_lemmas_to_words = None
        self.counter = None
        self.__autho_key = 'ZjY4NjI1NzItODIxNi00NDMyLWJlZjQtNWRjNGIxOWZkYmE4OjRmMzE2N2M3LTNmYzYtNDUxYi1iODc5LWI1NGIxMmYzYzdiYw=='

    def __get_answer(self, prompt: str) -> str:
        with GigaChat(credentials=self.__autho_key, verify_ssl_certs=False) as giga:
            response = giga.chat(prompt)
        return response.choices[0].message.content

    def _cleaning_answers(self, words):
        nltk.download('stopwords')
        stopwords_ru = stopwords.words("russian")

        reg_stop = '|'.join(map(lambda x: f'(^{x}$)', stopwords_ru)) + '|(^из-за$)|(^из-под$)|(^.$)'  # в стоп-словах не было этих союзов

        reg = r'[^А-Яа-я /-]'
        self.bad_reg = r'(.*ху.*)|(.*пиз.*)|(.*еба.*)|(.*еби.*)' + rf'|{reg_stop}'  # обработаем плохие слова и стоп слова

        # df = pd.read_excel('dataset.xlsx').fillna('').astype(str)
        assert words is not None, 'Firstly, you need loaded data'

        # words = np.array(df.iloc[:, 0])
        clean_words = []
        for word in words:
            try:
                clean_words.append(re.sub(reg, '', word.lower()).strip())
            except Exception:
                clean_words.append('')
        # Обрати внимание, что тут я прогоняю только 1 список слов - по факту это только один столбец. Если столбцов несколько, то надо цикл
        # clean_words = [re.sub(reg, '', word.lower()).strip() for word in words]
        return clean_words

    def _lemmatize_answers(self, clean_words):
        morph = pymorphy2.MorphAnalyzer()
        self.dict_lemmas_to_words = {}
        lemmas = []

        assert self.bad_reg is not None, "Bad_reg don't exist"

        for word in clean_words:
            data = word.split()
            if len(data) == 1:
                lemmas.append(morph.parse(word)[0].normal_form)
                if word:
                    self.dict_lemmas_to_words[lemmas[-1]] = word
            else:
                lemmas.append(' '.join([morph.parse(w)[0].normal_form for w in data if not(re.search(self.bad_reg, w))]))
                if word:
                    self.dict_lemmas_to_words[lemmas[-1]] = word

        self.lemmas = [word for word in lemmas if not(re.search(self.bad_reg, word)) and word]

        # Загрузка модели RoBERTa
        self.model_lemma = SentenceTransformer('stsb-roberta-base')

        sentences = self.lemmas  # Добавить леммы
        word_vectors = self.model_lemma.encode(sentences)
        return word_vectors

    def preprocessing_data(self, words):
        clean_words = self._cleaning_answers(words)
        self.clean_data = self._lemmatize_answers(clean_words)
        return self.clean_data

    def load_data(self, data):
        self.answers_data = data
        self.answers_data.fillna('')

    def _count_clusters(self, clean_data):
        assert self.lemmas is not None, 'Firstly, you need preprocessing data'

        ran = range(1, min(len(self.lemmas) + 1, 40))
        inertia_df = pd.DataFrame(data=[], index=ran, columns=['inertia'])
        effective_clusters = -1
        for n_clusters in tqdm(ran):
            try:
                effective_clusters += 1
                centers, indices = kmeans_plusplus(np.array(clean_data), n_clusters=n_clusters, random_state=10)
                kmeans = KMeans(n_clusters=n_clusters,  random_state=42)
                cluster_labels = kmeans.fit_predict(clean_data)
                inertia_df.loc[n_clusters] = kmeans.inertia_
            except:
                break
        inertia_df = inertia_df.iloc[:effective_clusters]
        inertia_arr = np.array(inertia_df).flatten()
        inertia_derivative = inertia_arr[:-1] - inertia_arr[1:]

        x, y = np.array(range(inertia_arr.shape[0])), np.array(inertia_arr).flatten()
        y_norm = y.max()
        y /= y_norm

        grad = GradSearch(a=1.0, b=0.0, x0=1.0, l=1e-2)
        grad.fit(x, y, epochs=100_000)
        a, b, x0 = grad.get_a_b_x0()
        n_clusters = np.ceil(x0).astype(int)

        return n_clusters

    def clustering(self, words):
        clean_data = self.preprocessing_data(words)
        n_clusters = self._count_clusters(clean_data)

        centers, indices = kmeans_plusplus(np.array(clean_data), n_clusters=n_clusters, random_state=10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(clean_data)

        df = pd.DataFrame(self.lemmas, columns=['word'])
        df['cluster'] = cluster_labels

        return df, n_clusters

    def get_statistic(self, words):
        df, n_clusters = self.clustering(words)
        assert self.dict_lemmas_to_words is not None, 'Firstly, you need preprocessing data'

        counter = {}
        for i in range(n_clusters):
            w = np.random.choice(df[df['cluster'] == i]['word'])
            amount = df[df['cluster'] == i].shape[0]
            counter[self.dict_lemmas_to_words[w]] = amount

        return counter

    def _get_short_ans(self, data_ans):
        answers = []
        n = len(data_ans)

        for i in tqdm(range(n)):
            ans = self.__get_answer(f'''
            Выдели из данного предложения основной смысл одной фразой (длина до 5 слов): "{list(data_ans.keys())[i]}",
            во фразе обязательно должно быть хотя бы 1 слово.
            ''')
            answers.append(ans.split('.')[0].strip('"'))

        return answers

    def _describe_emotion(self, words):

        counter = self.get_statistic(words)
        n = len(counter)
        answers = self._get_short_ans(counter)
        marks = []

        for i in tqdm(range(n)):
            try:
                mark = self.__get_answer(f'''
                Оцени эмоциональный окрас этой фразы: {answers[i]} одним числом по шкале от -10 до 10, где -10 обозначает абсолютно негативное
                предложение, 10 - абсолютно позитивное. Шаблон ответа: *фраза*. Оценка <>.
                ''')
                marks.append(int(re.findall(r'-?\d{2}|-?\d{1}', mark)[-1]))
            except Exception as e:
                marks.append(np.random.randint(-5, 1))

        answers_with_emotions = {name: (val, mark) for name, val, mark in zip(answers, counter.values(), marks)}

        return answers_with_emotions

    def get_bar_statistic(self, ans_info, title, index):
        EMOTION_COLORS = ['#e8e8e8', '#5ddd5d', '#FF6B6B']

        ans_info = dict(sorted(ans_info.items(), key=lambda x: x[1][0], reverse=True))
        labels = []
        values = []
        emotions = []
        val_cnt = 0
        all_cnt = sum([val[0] for val in ans_info.values()])

        for name, (cnt, emotion) in ans_info.items():
            if cnt / all_cnt <= 0.02:
                val_cnt += cnt
            else:
                labels.append(re.sub(r'[^A-Za-zА-Яа-я \-0-9]', '', name))
                values.append(cnt)
                emotions.append(emotion)

        if val_cnt != 0:
            labels.append('Иные категории')
            values.append(val_cnt)
            emotions.append(0)
        cluster_colors = [EMOTION_COLORS[np.sign(val)] for val in emotions]

        fig, ax = plt.subplots(figsize=(12, 6))
        barplot = sns.barplot(x=values, y=labels, width=0.5,
                            palette=cluster_colors, hue=labels)

        for container in barplot.containers:
            barplot.bar_label(container, fmt=' {:.0f}')

        plt.title(title, fontsize=14)

        plt.gca().axes.get_xaxis().set_visible(False)
        sns.despine(bottom=True)

        plt.savefig(f'bar_{index}.png', bbox_inches='tight', dpi=300)
        # plt.show()

    def get_pie_statistic(self, ans_info, title, index):
        ans_info = {key: val for key, (val, _) in ans_info.items()}
        ans_info = dict(sorted(ans_info.items(), key=lambda x: x[1], reverse=True))

        labels = []
        values = []
        all_cnt = sum(ans_info.values())
        val_cnt = 0
        for name, cnt in ans_info.items():
            if cnt / all_cnt <= 0.02:
                val_cnt += cnt
            else:
                labels.append(re.sub(r'[^A-Za-zА-Яа-я \-0-9]', '', name))
                values.append(cnt)

        if val_cnt != 0:
            labels.append('Иные категории')
            values.append(val_cnt)

        colors = sns.color_palette("pastel", len(labels))
        fig, ax = plt.subplots(figsize=(18, 6))

        plt.pie(values, colors=colors, startangle=90,
                wedgeprops=dict(width=0.4, edgecolor='white'),
                autopct='%1.0f%%', pctdistance=1.1)

        plt.legend(labels, loc="best")
        plt.axis('equal')

        plt.title(title, fontsize=14)

        plt.savefig(f'pie_{index}.png', bbox_inches='tight', dpi=300)
        # plt.show()

    def get_graphics(self):
        assert self.answers_data is not None, 'Firstly, need load data'

        for question_num, column_name in enumerate(list(self.answers_data.columns)):
            words = np.array(self.answers_data.iloc[:, question_num])
            column_name = list(self.answers_data.columns)[question_num]

            ans_info = self._describe_emotion(words)
            self.get_bar_statistic(ans_info, column_name, question_num)
            self.get_pie_statistic(ans_info, column_name, question_num)

        n = len(list(self.answers_data.columns))
        images_paths = [f'{type}_{i}.png' for i in range(n) for type in ('bar', 'pie')]
        self.get_pdf_report(image_paths=images_paths)

    def get_pdf_report(self, image_paths, output_pdf = 'statistics.pdf', image_per_page = 2):
        Image.MAX_IMAGE_PIXELS = None

        c = canvas.Canvas(output_pdf, pagesize=A4)
        page_width, page_height = A4

        for i in range(0, len(image_paths), image_per_page):

            for j in range(image_per_page):
                index = i + j
                if index >= len(image_paths):
                    break

                img = Image.open(image_paths[index])

                # Пропорционально уменьшаем изображения, чтобы они помещались на странице
                img_width = page_width - 100  # Учитываем отступы
                img_height = img.height * (img_width / img.width)  # Пропорциональная высота

                if img_height > (page_height - 100) / image_per_page:  # Проверяем, если высота превышает оставшееся пространство
                    img_height = (page_height - 100) / image_per_page  # Высота для каждого изображения
                    img_width = img.width * (img_height / img.height)  # Пропорциональная ширина

                # Центрируем изображение по горизонтали
                img_x = (page_width - img_width) / 2
                img_y = page_height - (j + 1) * img_height - 60  # Выравниваем по вертикали

                img.save(f"temp_image_{index}.png")
                c.drawImage(f"temp_image_{index}.png", img_x, img_y, width=img_width, height=img_height)

            # Переход на новую страницу, если не все изображения размещены
            c.showPage()
        c.save()

        # Удаляем временные изображения
        for index in range(len(image_paths)):
            os.remove(f"temp_image_{index}.png")

    def get_personal_statistic(self, id):
        assert self.answers_data is not None, 'Firstly, need load data'
        personal_info = self.answers_data.iloc[id] # id == 'id'
        questions = list(self.answers_data.columns)
        short_ans = self._get_short_ans(personal_info)
        prepared_info = {question: ans for question, ans in zip(questions, short_ans)}

        df_info = pd.DataFrame.from_dict(prepared_info, orient='index', columns=['Основная мысль ответа'])

        return df_info.to_html(justify='center')

model = MrBist()

@dp.message_handler(commands=['start', 'help'])
async def hi(message: types.Message):
    markup = types.ReplyKeyboardRemove()
    await message.answer('Приветствую! Меня зовут Voroge Jaskela RH, я чат-бот предназначенный для создания облака слов на основе пользовательских ответов на вопросы'
                         '\n\nОтправь мне список слов в формате csv файла, чтобы я мог создать облако', reply_markup=markup)

@dp.message_handler(commands=['personal'])
async def handle_data_command(message: types.Message):
    args = message.text.split()
    if len(args) < 2:
        await message.answer("Недостаточно параметров. Укажите ID сотрудника.")
        return
    index = int(args[1])

    html = model.get_personal_statistic(index)
    await message.answer(f'<html><body><h3>Персональная информация по ответам пользователя</h3>{html}', parse_mode='HTML')

@dp.message_handler(commands=['main'])
async def handle_main_command(message: types.Message):
    await message.reply("Данные обрабатываются. Подождите...")
    model.get_graphics()

    file_id = bot.get_file('statistics.pdf')
    await message.reply(
        f'Общая статистика по персоналу здесь:\n{file_id.file_link}',
        reply_markup=types.ReplyKeyboardRemove()
    )

@dp.message_handler(content_types=types.ContentType.DOCUMENT, state='*')
async def handle_document(message: types.Message, state: FSMContext):
    if (message.document.mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
        await state.set_state(FormStates.waiting_for_file)
        await message.document.download(destination_file='received_file.xlsx')
        await message.reply("Данные загружены.")
        try:
            df = pd.read_excel('received_file.xlsx', dtype=str)
            model.load_data(df)
            await message.reply("Готово! Выберите вариант представления данных.")
            await state.finish()
        except Exception as e:
            await message.reply(f"Ошибка при чтении файла: {str(e)}")
            await state.finish()
    else:
        await message.reply("Пожалуйста, пришлите файл формата .xlsx")
