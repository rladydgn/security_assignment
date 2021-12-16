import json
import pandas as pd

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score


SEED = 41


def read_label_csv(path):
    label_table = dict()
    with open(path, "r", encoding='cp949') as f:
        for line in f.readlines()[1:]:
            fname, label = line.strip().split(",")
            label_table[fname] = int(label)
    return label_table


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(**kwargs):
    if kwargs["model"] == "rf":
        return RandomForestClassifier(random_state=kwargs["random_state"], n_jobs=4)
    elif kwargs["model"] == "dt":
        return DecisionTreeClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "lgb":
        return LGBMClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "svm":
        return SVC(random_state=kwargs["random_state"])
    elif kwargs["model"] == "lr":
        return LogisticRegression(random_state=kwargs["random_state"], n_jobs=-1)
    elif kwargs["model"] == "knn":
        return KNeighborsClassifier(n_jobs=-1)
    elif kwargs["model"] == "adaboost":
        return AdaBoostClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "mlp":
        return MLPClassifier(random_state=kwargs["random_state"])
    else:
        print("Unsupported Algorithm")
        return None


def train(X_train, y_train, model):
    clf = load_model(model=model, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf


def evaluate(X_val, Y_val, model):
    predict = model.predict(X_val)
    print("정확도", model.score(X_val, Y_val))
    return predict

def ensemble_hard(Y, preds):
    final_pred = []
    for i in range(len(Y)):
        s = 0
        for pred in preds:
            s += pred[i]
        if s >= len(preds)/2:
            final_pred.append(1)
        else:
            final_pred.append(0)
    cnt = 0.0
    for i in range(len(Y)):
        if(Y[i] == final_pred[i]):
            cnt += 1
    acc = cnt / len(Y)
    f1 = f1_score(Y, final_pred)
    print(f"앙상블 정확도 : {acc}")
    print(f"양상블 f1 : {f1}")

    return final_pred

def ensemble_soft(X, Y, models):
    final_preds = []
    preds = []
    cnt = 0
    for model in models:
        preds.append(model.predict_proba(X))
    for i in range(len(Y)):
        total1 = 0
        total2 = 0
        for j in range(len(models)):
            total1 += preds[j][i][0]
            total2 += preds[j][i][1]
        if total1 >= total2:
            final_preds += [0]
        else:
            final_preds += [1]
        if final_preds[i] == Y[i]:
            cnt += 1
    acc = cnt / len(Y)
    f1 = f1_score(Y, final_preds)
    print(f"앙상블 정확도 : {acc}")
    print(f"양상블 f1 : {f1}")

    return final_preds

def test_ensemble_soft(X, models):
    final_preds = []
    preds = []
    for model in models:
        preds.append(model.predict_proba(X))
    for i in range(len(X)):
        total1 = 0
        total2 = 0
        for j in range(len(models)):
            total1 += preds[j][i][0]
            total2 += preds[j][i][1]
        if total1 >= total2:
            final_preds += [0]
        else:
            final_preds += [1]

    return final_preds

def save_csv(df, filename):
    answer = pd.DataFrame(df)
    answer.columns = ['파일', '정답']
    answer.to_csv(filename + '.csv', index=False, encoding='cp949')