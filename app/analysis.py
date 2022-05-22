import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer

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

    models_features = {
        KNeighborsClassifier(): {
            'name': 'K-nearest neighbours algorithm',
            'params': {
                'n_neighbors': [3, 4],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'brute'],
            },
        },
        RandomForestClassifier(random_state=40): {
            'name': 'Random forest algorithm',
            'params': {
                'n_estimators': [300, 500],
                'max_features': ['auto', 'log2'],
                'class_weight': ['balanced', 'balanced_subsample'],
            },
        },
    }
    models_scores = {x: None for x in models_features}
    models_scores_norm = {x: None for x in models_features}

    for model in models_scores:
        score = fit_predict_eval(model, x_train, x_test, y_train, y_test)
        models_scores[model] = score

    normalizer = Normalizer()
    x_train_norm = normalizer.transform(x_train)
    x_test_norm = normalizer.transform(x_test)

    for model in models_scores_norm:
        score = fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)
        models_scores_norm[model] = score

    x_train_best, x_test_best, y_train_best, y_test_best = (
        (x_train_norm, x_test_norm, y_train, y_test)
        if max(models_scores_norm.values()) > max(models_scores.values())
        else (x_train, x_test, y_train, y_test)
    )

    for model, features in models_features.items():
        optimizer = GridSearchCV(model, features['params'], scoring='accuracy', n_jobs=-1)
        optimizer.fit(x_train_best, y_train_best)
        best_estimator = optimizer.best_estimator_
        score = fit_predict_eval(best_estimator, x_train_best, x_test_best, y_train_best, y_test_best)
        print(features['name'])
        print('best estimator:', best_estimator)
        print('accuracy:', round(score, 3))
        print()


if __name__ == '__main__':
    main()
