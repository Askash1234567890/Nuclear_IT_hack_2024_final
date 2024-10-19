import re
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import pymorphy2
import matplotlib.pyplot as plt
from tqdm import tqdm
from gigachat import GigaChat
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, kmeans_plusplus
from sentence_transformers import SentenceTransformer

from scripts.grad_search import GradSearch

AUTHORIZATION_KEY = \
    'ZjY4NjI1NzItODIxNi00NDMyLWJlZjQtNWRjNGIxOWZkYmE4OjRmMzE2N2M3LTNmYzYtNDUxYi1iODc5LWI1NGIxMmYzYzdiYw=='
EMOTION_COLORS = ['#e8e8e8', '#5ddd5d', '#FF6B6B']


class ExitInterviewAnalyzer:
    def __init__(self) -> None:
        """Initialize the ExitInterviewAnalyzer class.
        """
        self.answers_data: pd.DataFrame = pd.DataFrame()
        self.clean_data: np.ndarray = np.array([])
        self.model_lemma: SentenceTransformer = SentenceTransformer('stsb-roberta-base')
        self.bad_reg: str = ""
        self.lemmas: List[str] = []
        self.dict_lemmas_to_words: Dict[str, str] = {}
        self.counter: Dict[str, int] = {}
        self.__autho_key: str = AUTHORIZATION_KEY

    def __get_answer(self, prompt: str) -> str:
        """Get an answer from the GigaChat API.

        :param prompt: The prompt to send to the API
        :return: The response from the API
        """
        with GigaChat(credentials=self.__autho_key, verify_ssl_certs=False) as giga:
            response = giga.chat(prompt)
        return response.choices[0].message.content

    def _cleaning_answers(self, words: List[str]) -> List[str]:
        """Clean the input words by removing stopwords and bad words.

        :param words: List of words to clean
        :return: List of cleaned words
        """
        nltk.download('stopwords')
        stopwords_ru = stopwords.words("russian")

        reg_stop = '|'.join(map(lambda x: f'(^{x}$)', stopwords_ru)) + '|(^из-за$)|(^из-под$)|(^.$)'
        reg = r'[^А-Яа-я /-]'
        self.bad_reg = r'(.*ху.*)|(.*пиз.*)|(.*еба.*)|(.*еби.*)' + rf'|{reg_stop}'

        clean_words = [re.sub(reg, '', word.lower()).strip() for word in words]
        return clean_words

    def _lemmatize_answers(self, clean_words: List[str]) -> np.ndarray:
        """Lemmatize the cleaned words.

        :param clean_words: List of cleaned words
        :return: Array of word vectors
        """
        morph = pymorphy2.MorphAnalyzer()
        self.dict_lemmas_to_words = {}
        lemmas = []

        assert self.bad_reg is not None, "Bad_reg doesn't exist"

        for word in clean_words:
            data = word.split()
            if len(data) == 1:
                lemmas.append(morph.parse(word)[0].normal_form)
                if word:
                    self.dict_lemmas_to_words[lemmas[-1]] = word
            else:
                lemmas.append(' '.join([morph.parse(w)[0].normal_form
                                        for w in data if not re.search(self.bad_reg, w)]))
                if word:
                    self.dict_lemmas_to_words[lemmas[-1]] = word

        self.lemmas = [word for word in lemmas if not re.search(self.bad_reg, word) and word]

        self.model_lemma = SentenceTransformer('stsb-roberta-base')
        word_vectors = self.model_lemma.encode(self.lemmas)
        return word_vectors

    def preprocessing_data(self, words: List[str]) -> np.ndarray:
        """Preprocess the input words by cleaning and lemmatizing them.

        :param words: List of words to preprocess
        :return: Array of preprocessed word vectors
        """
        clean_words = self._cleaning_answers(words)
        self.clean_data = self._lemmatize_answers(clean_words)
        return self.clean_data

    def load_data(self, data: pd.DataFrame) -> None:
        """Load the data into the model.

        :param data: DataFrame containing the data to load
        """
        self.answers_data = data
        self.answers_data.fillna('')

    def _count_clusters(self, clean_data: np.ndarray) -> int:
        """Count the optimal number of clusters using the elbow method and gradient search.

        :param clean_data: Array of preprocessed word vectors
        :return: Optimal number of clusters
        """
        assert self.lemmas is not None, 'Firstly, you need preprocessing data'

        ran = range(1, min(len(self.lemmas) + 1, 40))
        inertia_df = pd.DataFrame(data=[], index=ran, columns=['inertia'])
        effective_clusters = -1
        for n_clusters in tqdm(ran):
            try:
                effective_clusters += 1
                _, _ = kmeans_plusplus(np.array(clean_data), n_clusters=n_clusters, random_state=10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                _ = kmeans.fit_predict(clean_data)
                inertia_df.loc[n_clusters] = kmeans.inertia_
            except:
                break
        inertia_df = inertia_df.iloc[:effective_clusters]
        inertia_arr = np.array(inertia_df).flatten()
        _ = inertia_arr[:-1] - inertia_arr[1:]

        x, y = np.array(range(inertia_arr.shape[0])), np.array(inertia_arr).flatten()
        y_norm = y.max()
        y /= y_norm

        grad = GradSearch(a=1.0, b=0.0, x0=1.0, l=1e-2)
        grad.fit(x, y, epochs=100_000)
        a, b, x0 = grad.get_a_b_x0()
        n_clusters = np.ceil(x0).astype(int)

        return n_clusters

    def clustering(self, words: List[str]) -> Tuple[pd.DataFrame, int]:
        """Perform clustering on the input words.

        :param words: List of words to cluster
        :return: DataFrame containing the clustered words and the number of clusters
        """
        clean_data = self.preprocessing_data(words)
        n_clusters = self._count_clusters(clean_data)

        _, _ = kmeans_plusplus(np.array(clean_data), n_clusters=n_clusters, random_state=10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(clean_data)

        df = pd.DataFrame(self.lemmas, columns=['word'])
        df['cluster'] = cluster_labels

        return df, n_clusters

    def get_statistic(self, words: List[str]) -> Dict[str, int]:
        """Get statistics of the clustered words.

        :param words: List of words to analyze
        :return: Dictionary containing the word statistics
        """
        df, n_clusters = self.clustering(words)
        assert self.dict_lemmas_to_words is not None, 'Firstly, you need preprocessing data'

        counter = {}
        for i in range(n_clusters):
            w = np.random.choice(df[df['cluster'] == i]['word'])
            amount = df[df['cluster'] == i].shape[0]
            counter[self.dict_lemmas_to_words[w]] = amount

        return counter

    def _get_short_ans(self, data_ans: Dict[str, int]) -> List[str]:
        """Get short answers from the data.

        :param data_ans: Dictionary containing the data answers
        :return: List of short answers
        """
        answers = []
        n = len(data_ans)

        for i in tqdm(range(n)):
            ans = self.__get_answer(f'''
            Выдели из данного предложения основной смысл одной фразой
            (длина до 5 слов): "{list(data_ans.keys())[i]}",
            во фразе обязательно должно быть хотя бы 1 слово.
            ''')
            answers.append(ans.split('.')[0].strip('"'))

        return answers

    def _describe_emotion(self, words: List[str]) -> Dict[str, Tuple[int, int]]:
        """Describe the emotion of the input words.

        :param words: List of words to analyze
        :return: Dictionary containing the words and their emotional scores
        """
        counter = self.get_statistic(words)
        n = len(counter)
        answers = self._get_short_ans(counter)
        marks = []

        for i in tqdm(range(n)):
            try:
                mark = self.__get_answer(f'''
                Оцени эмоциональный окрас этой фразы: {answers[i]} одним числом по шкале от -10 до 10,
                где -10 обозначает абсолютно негативное
                предложение, 10 - абсолютно позитивное. Шаблон ответа: *фраза*. Оценка <>.
                ''')
                marks.append(int(re.findall(r'-?\d{2}|-?\d', mark)[-1]))
            except Exception:
                marks.append(np.random.randint(-5, 1))

        answers_with_emotions = {name: (val, mark)
                                 for name, val, mark in
                                 zip(answers, counter.values(), marks)}

        return answers_with_emotions

    @staticmethod
    def get_bar_statistic(ans_info: Dict[str, Tuple[int, int]], title: str, index: int) -> None:
        """Generate a bar plot for the given answer information.

        :param ans_info: Dictionary containing the answer information
        :param title: Title of the bar plot
        :param index: Index of the bar plot
        """
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
                labels.append(name)
                values.append(cnt)
                emotions.append(emotion)

        if val_cnt != 0:
            labels.append('Иные категории')
            values.append(val_cnt)
            emotions.append(0)
        cluster_colors = [EMOTION_COLORS[np.sign(val)] for val in emotions]

        _, _ = plt.subplots(figsize=(12, 6))
        barplot = sns.barplot(x=values, y=labels, width=0.5, palette=cluster_colors, hue=labels)

        for container in barplot.containers:
            barplot.bar_label(container, fmt=' {:.0f}')

        plt.title(title, fontsize=14)
        plt.gca().axes.get_xaxis().set_visible(False)
        sns.despine(bottom=True)

        plt.show()
        plt.savefig(f'bar_{index}.png')

    @staticmethod
    def get_pie_statistic(ans_info: Dict[str, Tuple[int, int]], title: str, index: int) -> None:
        """Generate a pie chart for the given answer information.

        :param ans_info: Dictionary containing the answer information
        :param title: Title of the pie chart
        :param index: Index of the pie chart
        """
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
                labels.append(name)
                values.append(cnt)

        if val_cnt != 0:
            labels.append('Иные категории')
            values.append(val_cnt)

        colors = sns.color_palette("pastel", len(labels))
        _, _ = plt.subplots(figsize=(18, 6))

        plt.pie(values, colors=colors, startangle=90,
                wedgeprops=dict(width=0.4, edgecolor='white'),
                autopct='%1.0f%%', pctdistance=1.1)

        plt.legend(labels, loc="best")
        plt.axis('equal')
        plt.title(title, fontsize=14)

        plt.show()
        plt.savefig(f'pie_{index}.png')

    def get_graphics(self) -> None:
        """Generate graphics for the loaded data.
        """
        assert self.answers_data is not None, 'Firstly, need load data'

        for question_num, column_name in enumerate(list(self.answers_data.columns)):
            words = np.array(self.answers_data.iloc[:, question_num])
            column_name = list(self.answers_data.columns)[question_num]

            ans_info = self._describe_emotion(words)
            self.get_bar_statistic(ans_info, column_name, question_num)
            self.get_pie_statistic(ans_info, column_name, question_num)

    def get_personal_statistic(self, id: int) -> str:
        """Get personal statistics for a given ID.

        :param id: ID of the person to get statistics for
        :return: HTML string containing the personal statistics
        """
        assert self.answers_data is not None, 'Firstly, need load data'
        personal_info = self.answers_data.iloc[id]
        questions = list(self.answers_data.columns)
        short_ans = self._get_short_ans(personal_info)
        prepared_info = {question: ans for question, ans in zip(questions, short_ans)}

        df_info = pd.DataFrame.from_dict(prepared_info, orient='index', columns=['Основная мысль ответа'])
        return df_info.to_html(justify='center')
