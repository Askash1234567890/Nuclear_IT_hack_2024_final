import unittest

import numpy as np
import pandas as pd

from scripts.analyzer import ExitInterviewAnalyzer


class TestExitInterviewAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ExitInterviewAnalyzer()
        self.sample_data = pd.DataFrame({
            'question1': ['answer1', 'answer2', 'answer3'],
            'question2': ['answer4', 'answer5', 'answer6']
        })
        self.sample_words = ['слово1', 'слово2', 'слово3']

    def test_load_data(self):
        self.analyzer.load_data(self.sample_data)
        self.assertFalse(self.analyzer.answers_data.empty)

    def test_preprocessing_data(self):
        self.analyzer.load_data(self.sample_data)
        preprocessed_data = self.analyzer.preprocessing_data(self.sample_words)
        self.assertIsInstance(preprocessed_data, np.ndarray)
        self.assertGreater(len(preprocessed_data), 0)

    def test_clustering(self):
        self.analyzer.load_data(self.sample_data)
        df, n_clusters = self.analyzer.clustering(self.sample_words)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(n_clusters, 0)

    def test_get_statistic(self):
        self.analyzer.load_data(self.sample_data)
        stats = self.analyzer.get_statistic(self.sample_words)
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)

    def test_get_personal_statistic(self):
        self.analyzer.load_data(self.sample_data)
        personal_stat = self.analyzer.get_personal_statistic(0)
        self.assertIsInstance(personal_stat, str)
        self.assertIn('<table', personal_stat)


if __name__ == '__main__':
    unittest.main()
