import pickle

# Open the file in read mode
with open('result.pkl', 'rb') as f:
    data = pickle.load(f)

count = 0
N = len(data)
for x in data[:N]:
    # if x['label'] != 'True' and x['a_j_adv'] != 'failed':
    rj0 = x[0]
    rk0 = x[1]
    if rj0>=rk0:
        count += 1
print(count/N)

import numpy as np
from sklearn.calibration import calibration_curve


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def compute_ece(y_true, y_probs, n_bins=10):
    """ Compute Expected Calibration Error (ECE)
    Args:
        y_true (numpy.ndarray): Ground truth labels.
        y_probs (numpy.ndarray): Model's predicted probabilities.
        n_bins (int, optional): The number of bins for the histogram.
    Returns:
        ece (float): The calculated ECE.
    """
    true_probs, pred_probs = calibration_curve(y_true, y_probs, n_bins=n_bins)
    bin_width = 1.0 / n_bins
    ece = np.sum(np.abs(true_probs - pred_probs) * bin_width)
    return ece, true_probs, pred_probs

# Replace with your own y_true and y_probs
y_true = np.random.randint(2, size=1000)  # binary ground truth labels
y_probs = np.random.rand(1000)  # predicted probabilities

ece, true_probs, pred_probs = compute_ece(y_true, y_probs, n_bins=10)

print(f"Expected Calibration Error (ECE): {ece}")

# Plotting the reliability diagram
plt.figure(figsize=(10, 8))
plt.plot(pred_probs, true_probs, marker='o')

plt.plot([0, 1], [0, 1], 'k--')  # identity line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Reliability Diagram')
plt.grid(True)

plt.show()



from transformers import pipeline
generator = pipeline("text-generation", model='/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/llama-7b-armor-new')