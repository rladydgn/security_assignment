import os

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
val_label_table = read_label_csv("D:\대학수업\정보보호와시스템보안\과제\데이터\검증데이터_정답.csv")

val_pred = []
val_answer = []
for key, value in ember_val_label_table.items():
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

# 예측 csv 파일 불러오기
ember_label_table = read_label_csv("ember_answer.csv")
peminer_label_table = read_label_csv("peminer_answer.csv")
#pestudio_label_table = read_label_csv("pestudio_answer.csv")

test_pred = []
for key, value in ember_label_table.items():
    

# json 파일 이름 불러오기
test_file_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\PEMINER\테스트데이터\\"
test_file_list = os.listdir(test_file_path)