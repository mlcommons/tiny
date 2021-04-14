'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

eval_functions_eembc.py: performances evaluation functions from eembc

refs:
https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Methodology/eval_functions_eembc.py
'''

import numpy as np
import matplotlib.pyplot as plt


# Classifier overall accuracy calculation
# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
def calculate_accuracy(y_pred, labels):
    y_pred_label = np.argmax(y_pred, axis=1)
    correct = np.sum(labels == y_pred_label)
    accuracy = 100 * correct / len(y_pred)
    print(f"Overall accuracy = {accuracy:2.1f}")
    return accuracy


# Classifier accuracy per class calculation
# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
# classes are the model's classes
def calculate_all_accuracies(y_pred, labels, classes):
    n_classes = len(classes)

    # Initialize array of accuracies
    accuracies = np.zeros(n_classes)

    # Loop on classes
    for class_item in range(n_classes):
        true_positives = 0
        # Loop on all predictions
        for i in range(len(y_pred)):
            # Check if it matches the class that we are working on
            if (labels[i] == class_item):
                # Get prediction label
                y_pred_label = np.argmax(y_pred[i, :])
                # Check if the prediction is correct
                if (labels[i] == y_pred_label):
                    true_positives += 1

        accuracies[class_item] = 100 * true_positives / np.sum(labels == class_item)
        print(f"Accuracy = {accuracies[class_item]:2.1f} ({classes[class_item]})")

    return accuracies


# Classifier ROC AUC calculation
# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
# classes are the model's classes
# name is the model's name
def calculate_auc(y_pred, labels, classes, name):
    n_classes = len(classes)

    # thresholds, linear range, may need improvements for better precision
    thresholds = np.arange(0.0, 1.01, .01)
    # false positive rate
    fpr = np.zeros([n_classes, len(thresholds)])
    # true positive rate
    tpr = np.zeros([n_classes, len(thresholds)])
    # area under curve
    roc_auc = np.zeros(n_classes)

    # get number of positive and negative examples in the dataset
    for class_item in range(n_classes):
        # Sum of all true positive answers
        all_positives = sum(labels == class_item)
        # Sum of all true negative answers
        all_negatives = len(labels) - all_positives

        # iterate through all thresholds and determine fraction of true positives
        # and false positives found at this threshold
        for threshold_item in range(1, len(thresholds)):
            threshold = thresholds[threshold_item]
            false_positives = 0
            true_positives = 0
            for i in range(len(y_pred)):
                # Check prediction for this threshold
                if (y_pred[i, class_item] > threshold):
                    if labels[i] == class_item:
                        true_positives += 1
                    else:
                        false_positives += 1
            fpr[class_item, threshold_item] = false_positives / float(all_negatives)
            tpr[class_item, threshold_item] = true_positives / float(all_positives)

        # Force boundary condition
        fpr[class_item, 0] = 1
        tpr[class_item, 0] = 1

        # calculate area under curve, trapezoid integration
        for threshold_item in range(len(thresholds) - 1):
            roc_auc[class_item] += .5 * (tpr[class_item, threshold_item] + tpr[class_item, threshold_item + 1]) * (
                        fpr[class_item, threshold_item] - fpr[class_item, threshold_item + 1]);

    # results
    roc_auc_avg = np.mean(roc_auc)
    print(f"Simplified average roc_auc = {roc_auc_avg:.3f}")

    plt.figure()
    for class_item in range(n_classes):
        plt.plot(fpr[class_item, :], tpr[class_item, :],
                 label=f"auc: {roc_auc[class_item]:0.3f} ({classes[class_item]})")
    plt.xlim([0.0, 0.1])
    plt.ylim([0.5, 1.0])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: ' + name)
    plt.grid(which='major')
    plt.show(block=False)

    return roc_auc


# Classifier overall accuracy calculation
# y_pred contains the outputs of the network for the validation data
# y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
# using this function is not recommended
def calculate_ae_accuracy(y_pred, y_true):
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        correct = np.sum(y_pred_binary == y_true)
        accuracy_tmp = 100 * correct / len(y_pred_binary)
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp

    print(f"Overall accuracy = {accuracy:2.1f}")
    return accuracy


# Classifier overall accuracy calculation
# y_pred contains the outputs of the network for the validation data
# y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
# this is the function that should be used for accuracy calculations
def calculate_ae_pr_accuracy(y_pred, y_true):
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    n_normal = np.sum(y_true == 0)
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build matrix of TP, TN, FP and FN
        true_negative = np.sum((y_pred_binary[0:n_normal] == 0))
        false_positive = np.sum((y_pred_binary[0:n_normal] == 1))
        true_positive = np.sum((y_pred_binary[n_normal:] == 1))
        false_negative = np.sum((y_pred_binary[n_normal:] == 0))
        # Calculate and store precision and recall
        precision[threshold_item] = true_positive / (true_positive + false_positive)
        recall[threshold_item] = true_positive / (true_positive + false_negative)
        # See if the accuracy has improved
        accuracy_tmp = 100 * (precision[threshold_item] + recall[threshold_item]) / 2
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp

    # Results
    print(f"Precision/recall accuracy = {accuracy:2.1f}")

    plt.figure()
    plt.plot(recall, precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.grid(which='major')
    plt.show(block=False)

    return accuracy


# Autoencoder ROC AUC calculation
# y_pred contains the outputs of the network for the validation data
# y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
# this is the function that should be used for accuracy calculations
# name is the model's name
def calculate_ae_auc(y_pred, y_true, name):
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.01, .01) * (np.amax(y_pred) - np.amin(y_pred))
    roc_auc = 0

    n_normal = np.sum(y_true == 0)
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(1, len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build TP and FP
        tpr[threshold_item] = np.sum((y_pred_binary[n_normal:] == 1)) / float(len(y_true) - n_normal)
        fpr[threshold_item] = np.sum((y_pred_binary[0:n_normal] == 1)) / float(n_normal)

    # Force boundary condition
    fpr[0] = 1
    tpr[0] = 1

    # Integrate
    for threshold_item in range(len(thresholds) - 1):
        roc_auc += .5 * (tpr[threshold_item] + tpr[threshold_item + 1]) * (
                    fpr[threshold_item] - fpr[threshold_item + 1]);

    # Results
    print(f"Simplified roc_auc = {roc_auc:.3f}")

    plt.figure()
    plt.plot(tpr, fpr, label=f"auc: {roc_auc:0.3f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC: ' + name)
    plt.grid(which='major')
    plt.show(block=False)

    return roc_auc
