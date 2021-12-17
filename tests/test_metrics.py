import unittest
from unittest import TestCase

import torch

from src.metrics import recall, ndcg
from src.metrics import recalls_and_ndcgs_for_ks


class TestMetrics(TestCase):

    def testRecall(self):
        scores = torch.Tensor([[0.5, 0.3, 0.2],
                            [0.2, 0.2, 0.6],
                            [0.3, 0.3, 0.4]])
        labels = torch.Tensor([[0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1]])
        k = 2
        recall_expected = 0.67
        recall_output = recall(scores, labels, k)
        self.assertAlmostEqual(recall_expected, recall_output, places=2)

    def testRecall(self):
        scores = torch.Tensor([[0.5, 0.3, 0.2],
                            [0.2, 0.2, 0.6],
                            [0.3, 0.3, 0.4]])
        labels = torch.Tensor([[0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1]])
        self.assertAlmostEqual(ndcg(scores, labels, k=2).item(), 0.6667, places=4)
        self.assertAlmostEqual(ndcg(scores, labels, k=3).item(), 0.8333, places=4)


    def testRecallNDCGforKs(self):
        scores = torch.Tensor([[0.5, 0.3, 0.2],
                            [0.2, 0.2, 0.6],
                            [0.3, 0.3, 0.4]])
        labels = torch.Tensor([[0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1]])
        k = [2,3]
        expected_output = {'NDCG@2': 0.6666666865348816,
                           'NDCG@3': 0.8333333134651184,
                           'Recall@2': 0.6666666865348816,
                           'Recall@3': 1.0}
        self.assertEqual(recalls_and_ndcgs_for_ks(scores, labels, k), expected_output)

    
if __name__ == '__main__':
    unittest.main()