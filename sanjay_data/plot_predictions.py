"""
This script will be used for trying to compute precision/recall stuff. I don't
think I can do this entirely automatically as I have to inspect the images.
Based on Sanjay's script. Looks like there are 13 data directories ("trials"),
each of which has 8 "centroid" images.

Some notes:

9scIg, LXBGm, mWMuP, bQ1W1, esaQA, mPjNp
    We should remove all of the first column from consideration. I just
    inspected these to see which ones clearly should not have been used, which
    is what we would do in practice anyway.
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import sys
np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=150)
np.set_printoptions(suppress=True)

# Data Directories
data_dirs = ['all-data/data-9scIg/',
             'all-data/data-AmYoV/',
             'all-data/data-DsTfq/',
             'all-data/data-LXBGm/',
             'all-data/data-UjVWK/',
             'all-data/data-X0ofS/',
             'all-data/data-agzar/',
             'all-data/data-bQ1W1/',
             'all-data/data-esaQA/',
             'all-data/data-hEu1f/',
             'all-data/data-mPjNp/',
             'all-data/data-mWMuP/',
             'all-data/data-wAyDR/']

# Indices of true (i.e. DEFORMED) instances, in the SAME ORDER as data_dirs above.
# Rule: deformed = points which have a centere in a deformed part. If at the
# border, need three such cases to qualify. This is quite manual. =(

deformed = [ 
    [(1,4), (1,5), (2,4), (2,5), (2,7), (2,8), (3,7), (3,8)], # 13/14/22/23 and 25/26/34/35
    [(4,2), (4,3), (1,4), (1,5), (2,5), (3,8), (4,8)], # 38/39 and 13/14/23 and 35/44
    [(1,5), (3,3), (3,4), (4,3), (4,4), (2,8), (3,8)], # 14 and 30/31/39/40 and 26/35
    [(1,5), (2,4), (2,5), (2,6), (3,5), (4,2), (0,8)], # 14/22/23/24/32 and 38 and 8
    [(0,3), (0,4), (1,3), (1,4), (2,3), (2,4)], # 3/4 and 12/13 and 21/22
    [(2,4), (2,5), (2,6), (3,4), (3,5), (3,6)], # 22/23/24 and 31/32/33 (X0ofS)
    [(3,0), (4,0), (4,1), (2,6), (2,7), (3,6), (3,7), (4,6), (4,7)], # 27/26/37 and 24/25/33/34/42/43
    [(3,4), (3,5), (1,6), (1,7)], # 31/32 and 15/16
    [(1,5), (2,4), (2,5), (2,6), (3,5)], # 14/22/23/24/32 (esaQA)
    [(3,2), (4,2), (4,3), (0,3), (0,4)], # 29/38/39 and 3/4 (there's other stuff but I don't *think* its deformed)
    [(4,1), (4,2), (4,3)], # 37/38/39
    [(0,6), (0,7), (1,6), (1,7), (3,3), (3,4)], # 6/7/15/16 and 31/32
    [], # Nothing is deformed!
]


def obtain_predictions():
    """ 
    Iterate through each data directory and compute probabilities (and average
    probabilities over the different perspectives, but must re-count to ensure
    we're dividing by the correct number. Usually it's 8 but sometimes it's
    smaller.
    """
    all_preds = []

    for ddir in data_dirs:
        print("\n\tINSIDE DATA DIRECTORY = {}\n".format(ddir))
    
        # Each data directory has 5x9 centroids and 8 instances of that. This is what we
        # will be using to compute final confidence predictions.
        all_perspectives = np.zeros((5,9,8))
        num_to_divide = np.zeros((5,9))
    
        # Not sure ??? From Sanjay.
        index = 240
    
        # based on the perspective shifs (Daniel: must be because some shifts caused
        # points to be out of the picture? I won't touch this.)
        correction =[(0,0), (0,0), (0,0), (0,0), (0,0), (0,1), (1,1), (1, 0)]
    
        for i in range(0,8):
            index = index + 1
            data = np.load(ddir+"pred_prob"+str(i)+".npy")[:,1].reshape((5,9))
            assert data.shape == (5,9)
            print("index i={}, data=\n{}\n".format(i,data))
        
            for x in range(5):
                for y in range(9):
                    newx = correction[i][0] + x
                    newy = correction[i][1] + y
                    if newx < 5 and newy < 9:
                        all_perspectives[x,y,i] = data[newx, newy]
                        num_to_divide[x,y] += 1
    
        # Order predictions by confidence for debugging purposes.
        predictions = []
        for x in range(5):
            for y in range(9):
                sum_probs = np.sum(all_perspectives[x,y,:])
                avg_probs = sum_probs / num_to_divide[x,y]
                predictions.append( (avg_probs, sum_probs, (x,y)) )
        predictions.sort(reverse=True)
        for p in predictions:
            print p

        all_preds.append(predictions)

    return all_preds


def plot_precision_recalls(all_preds):
    """ 
    Compute precision-recall statistics and then plot it.  Thank goodness for
    scipy and their functions... we only need to create y_true and y_scores.
    Also, here we should detect and delete any of the non-informative cases from
    consideration, which happen when we're predicting stuff that's not even on
    the gauze. The positive cases here are DEFORMED stuff, so make those "1".
    """

    max_num_cases = 5*9*len(all_preds)
    y_true = []
    y_scores = []

    # Elements in `preds` are tuples: (probability, sum of probabilities, [x,y]-coord).
    for (index,preds) in enumerate(all_preds):
        for element in preds:
            
            # Now here, delete non-informative cases.
            # Remove first column of 9scIg.
            if (index == 0 and element[2][1] == 0):
                continue
            # Remove first column of LXBGm.
            if (index == 3 and element[2][1] == 0):
                continue
            # Remove first column of bQ1W1.
            if (index == 7 and element[2][1] == 0):
                continue
            # Remove first column of esaQA.
            if (index == 8 and element[2][1] == 0):
                continue
            # Remove first column of mPjNp.
            if (index == 10 and element[2][1] == 0):
                continue
            # Remove first column of mWMuP.
            if (index == 11 and element[2][1] == 0):
                continue

            y_scores.append(element[0])
            if (element[2] in deformed[index]):
                y_true.append(1) # Positive case, deformed.
            else:
                y_true.append(0) # Negative case, not deformed.

    print("len(y_true) = {}, # deformed = {}, non-def = {}".format(
        len(y_true), y_true.count(1), y_true.count(0)))
    assert len(y_true) <= max_num_cases
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
    print(thresholds)
    print(len(thresholds))
    average_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(10,7))
    plt.title("Deformation Detector Precision-Recall", fontsize=30)
    plt.xlabel("Recall", fontsize=26)
    plt.ylabel("Precision", fontsize=26)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.plot(recall, precision, lw=4, color="blue", 
             label="Precision-Recall (Area={:.4f})".format(average_precision))
    plt.tick_params(axis='both', which='major', labelsize=21)
    plt.tick_params(axis='both', which='minor', labelsize=21)
    plt.legend(loc="lower left", prop={'size':23})
    plt.tight_layout()
    plt.savefig("precision_recall_plot.png")


if __name__ == "__main__":
    assert len(data_dirs) == len(deformed)
    all_preds = obtain_predictions()
    plot_precision_recalls(all_preds)
