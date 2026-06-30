import sys
sys.path.append("GestureRecognition")

import numpy as np
from hmmclassifier import HMMClassifier

sequences = [np.random.randn(30, 2) for _ in range(20)]
labels = ["A"] * 10 + ["B"] * 10

train_seqs, train_labels, test_seqs, test_labels = HMMClassifier.train_test_split(sequences, labels, test_ratio=0.2)

print("Train:", len(train_seqs), "Test:", len(test_seqs))
print("Train Labels A/B:", train_labels.count("A"), train_labels.count("B"))
print("Test Labels A/B:", test_labels.count("A"), test_labels.count("B"))

# Funktioniert. 16 Train, 4 Test, gleichmäßig pro Klasse verteilt.