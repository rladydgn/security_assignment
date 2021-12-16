import os

from common import save_csv
from sklearn.metrics import f1_score

def read_label_csv(path):
    label_table = dict()
    with open(path, "r", encoding='cp949') as f:
        for line in f.readlines()[1:]:
            fname, label = line.strip().split(",")
            label_table[fname] = int(label)
    return label_table
# 검증 csv 파일 불러오기
ember_val_label_table = read_label_csv("ember_val.csv")
peminer_val_label_table = read_label_csv("peminer_val.csv")
pestudio_val_label_table = read_label_csv("pestudio_val.csv")
val_label_table = read_label_csv("D:\대학수업\정보보호와시스템보안\과제\데이터\검증데이터_정답.csv")

val_pred = []
val_answer = []
for key, value in ember_val_label_table.items():
    try:
        total = value + peminer_val_label_table[key] + pestudio_val_label_table[key]
        if(total >= 2):
            val_pred.append(1)
        else:
            val_pred.append(0)
    except KeyError:
        if(value != peminer_val_label_table[key]):
            val_pred.append(1)
        else:
            val_pred.append(value)
    val_answer.append(val_label_table[key])

cnt = 0
for i in range(len(val_answer)):
    if val_pred[i] == val_answer[i]:
        cnt += 1
print("acc : ", cnt/len(val_answer))
f1 = f1_score(val_answer, val_pred)
print("f1 score :", f1)

# test 예측 csv 파일 불러오기
ember_label_table = read_label_csv("ember_answer.csv")
peminer_label_table = read_label_csv("peminer_answer.csv")
pestudio_label_table = read_label_csv("pestudio_answer.csv")

test_pred = []
test_answer = []
filename = []
for key, value in ember_label_table.items():
    try: # pestudio에 파일이 존재하는 경우
        total = value + peminer_label_table[key] + pestudio_label_table[key]
        if(total >= 2):
            test_pred.append(1)
        else:
            test_pred.append(0)
    except KeyError: # pestudio에 파일이 존재하지 않는 경우
        if(value != peminer_label_table[key]):
            test_pred.append(1)
        else:
            test_pred.append(value)
    filename.append(key)

for i in range(len(test_pred)):
    test_answer.append([filename[i], test_pred[i]])

save_csv(test_answer, 'test')