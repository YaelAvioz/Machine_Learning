import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import shuffle


def parse_input(train_x_path, train_y_path, test_x_path):
    train_x = np.loadtxt(train_x_path, delimiter=',')
    train_y = np.loadtxt(train_y_path, delimiter=',')
    test_x = np.loadtxt(test_x_path, delimiter=',')
    return train_x, train_y, test_x


def write_res():
    with open(output, 'w') as out:
        for i in range(test_x.shape[0]):
            out.write('knn: {}, perceptron: {}, svm: {}, pa: {}\n'.format(knn_results[i], perceptron_results[i],
                                                                          svm_results[i], pa_results[i]))

def test_knn(train_x, train_y, test_x):
    k=3
    dist_arr = np.zeros(train_x.shape[0])
    results = np.zeros(test_x.shape[0]).astype(int)
    for test_i, test_sample in enumerate(test_x):
        for train_i, train_sample in enumerate(train_x):
            dist_arr[train_i] = np.linalg.norm(test_sample - train_sample)
        dist_arr_sorted_idx = dist_arr.argsort()
        closest_idx = dist_arr_sorted_idx[:k]
        closest_labels = train_y[closest_idx].astype(int)
        results[test_i] = np.argmax(np.bincount(closest_labels))

    return results


def test_perceptron(w, x):
    results = []
    for x_i in x:
        results.append(np.argmax(np.dot(w, x_i)))

    return results


def test_svm(w, x):
    results = []
    for x_i in x:
        results.append(np.argmax(np.dot(w, x_i)))

    return results


def test_pa(w, x):
    results = []
    for x_i in x:
        results.append(np.argmax(np.dot(w, x_i)))

    return results


def train_perceptron(x, y, T):
    lr = 0.2
    w = np.zeros((3, 5))

    for _ in range(T):
        dataset = list(zip(x, y))
        np.random.shuffle(dataset)
        for x_i, y_i in dataset:
            y_hat = np.argmax(np.dot(w, x_i))
            if y_hat != y_i:
                w[y_i, :] = w[y_i, :] + lr * x_i
                w[y_hat, :] = w[y_hat, :] - lr * x_i

    return w


def train_svm(x, y, T):
    lr = 0.001
    w = np.zeros((3, 5))
    lamda = 0.2

    for t in range(T):
        dataset = list(zip(x, y))
        np.random.shuffle(dataset)
        lr /= (t + 1)
        for x_i, y_i in dataset:

            y_hat = np.argmax(np.dot(w, x_i))
            loss =max(0, 1 - np.dot(w[y_i, :], x_i) + np.dot(w[y_hat, :], x_i))
            if  loss> 0:
                w[y_i, :] = (1 - lr * lamda) * w[y_i, :] + lr * x_i
                w[y_hat, :] = (1 - lr * lamda) * w[y_hat, :] - lr * x_i

            w = w * (1 - lr * lamda)

    return w


def train_pa(x, y, T):
    w = np.zeros((3, 5))
    for _ in range(T):
        dataset = list(zip(x, y))
        np.random.shuffle(dataset)
        for x_i, y_i in dataset:
            y_hat = np.argmax(np.dot(w, x_i))
            if y_hat != y_i:
                loss = max(0, 1 - np.dot(w[y_i], x_i) + np.dot(w[y_hat], x_i))
                d = 2 * (np.linalg.norm(x_i) ** 2)
                if loss > 0:
                    tau = loss / d if d != 0 else 1
                    w[y_i, :] = w[y_i, :] + tau * x_i
                    w[y_hat, :] = w[y_hat, :] - tau * x_i

    return w


if __name__ == "__main__":
    train_x_path, train_y_path, test_x_path, output = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x, train_y, test_x = parse_input(train_x_path, train_y_path, test_x_path)

    train_y = train_y.astype(int)

    # normalize train_x
    for c in range(train_x.shape[1]):
        train_x_avg = np.mean(train_x[:, c])
        std = np.std(train_x[:, c])
        if std > 0:
            train_x[:, c] = (train_x[:, c] - train_x_avg) / std

    # normalize test_x
    for c in range(test_x.shape[1]):
        test_x_avg = np.mean(test_x[:, c])
        std = np.std(test_x[:, c])
        if std > 0:
            test_x[:, c] = (test_x[:, c] - test_x_avg) / std

    w_perceptron = train_perceptron(train_x, train_y, 20)
    w_pa = train_pa(train_x, train_y, 20)
    w_svm = train_svm(train_x, train_y, 30)

    knn_results = test_knn(train_x, train_y, test_x)
    perceptron_results = test_perceptron(w_perceptron, test_x)
    svm_results = test_svm(w_svm, test_x)
    pa_results = test_pa(w_pa, test_x)

    write_res()
