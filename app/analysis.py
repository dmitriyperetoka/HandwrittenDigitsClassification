from functools import reduce

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier

TRAIN_SAMPLE_SIZE = 6000
TEST_SIZE = 0.3
RANDOM_STATE = 40


def fetch_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    x = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))[:TRAIN_SAMPLE_SIZE]
    y = y_train[:TRAIN_SAMPLE_SIZE]

    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    prediction = model.predict(features_test)
    return precision_score(target_test, prediction, average='macro')


def main():
    x_train, x_test, y_train, y_test = fetch_dataset()

    models_scores = {
        KNeighborsClassifier(): None,
        DecisionTreeClassifier(): None,
        LogisticRegression(random_state=RANDOM_STATE): None,
        RandomForestClassifier(): None,
    }
    models_scores_norm = models_scores.copy()

    for model in models_scores:
        score = fit_predict_eval(model, x_train, x_test, y_train, y_test)
        models_scores[model] = score

    normalizer = Normalizer()
    x_train_norm = normalizer.transform(x_train)
    x_test_norm = normalizer.transform(x_test)

    for model in models_scores_norm:
        score = fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)
        models_scores_norm[model] = score
        print(f'Model: {model}\nAccuracy: {score}\n')

    is_result_better = max(models_scores_norm.values()) > max(models_scores.values())
    print('The answer to the 1st question: {}\n'.format('yes' if is_result_better else 'no'))

    top_rating = reduce(list.__add__, map(list, sorted(models_scores.items(), key=lambda x: x[1], reverse=True)[:2]))
    print(
        'The answer to the 2nd question: {}-{}, {}-{}'.format(
            top_rating[0].__class__.__name__,
            round(top_rating[1], 3),
            top_rating[2].__class__.__name__,
            round(top_rating[3], 3)
        )
    )


if __name__ == '__main__':
    main()
