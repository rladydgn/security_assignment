import os

from common import read_json, read_label_csv, train, evaluate, ensemble_hard, ensemble_soft, save_csv, test_ensemble_soft


class PeminerParser:
    def __init__(self, path):
        self.report = read_json(path)
        self.vector = []

    def process_report(self):
        # 사전 KEY값 추출(숫자)
        self.vector = [value for _, value in sorted(self.report.items(), key=lambda x: x[0])]
        self.test = [_ for _, value in sorted(self.report.items(), key=lambda x: x[0])]
        # nouse = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 27, 28, 29, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42,
        #          43, 45, 46, 47, 48, 49, 50, 112, 113, 115, 116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
        #          129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 151,
        #          154, 157, 172, 179, 184, 186]
        # idx = 0
        # for key, value in self.report.items():
        #     if idx not in nouse:
        #         self.vector += [value]
        #     idx += 1
        return self.vector



train_label_table = read_label_csv("D:\대학수업\정보보호와시스템보안\과제\데이터\학습데이터_정답.csv")
X_train = []
Y_train = []

# 학습 데이터
cnt = 0
for key, value in train_label_table.items():
    peminer_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\PEMINER\학습데이터\\" + key + ".json"
    peminer_vector = PeminerParser(peminer_path).process_report()
    X_train.append(peminer_vector)
    Y_train.append(value)
    cnt += 1
    #if cnt == 20:
    #    break

# 학습
models = []
for model in ['rf', 'lgb', 'adaboost']:
    output = train(X_train, Y_train, model)
    models.append(output)
print(models)

test_label_table = read_label_csv("D:\대학수업\정보보호와시스템보안\과제\데이터\검증데이터_정답.csv")
X_val = []
Y_val = []

# 검증 데이터
filename = []
cnt = 0
for key, value in test_label_table.items():
    peminer_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\PEMINER\검증데이터\\" + key + ".json"
    peminer_vector = PeminerParser(peminer_path).process_report()
    X_val.append(peminer_vector)
    Y_val.append(value)
    filename += [key]
    cnt += 1
    #if cnt == 20:
    #    break

preds = []
for model in models:
    preds.append(evaluate(X_val, Y_val, model))

print("입력 벡터 크기 : ", len(X_val[0]))
print(f"rf 훈련 : {models[0].score(X_train, Y_train)}")
#print(f"검증 : {models[0].score(X_val, Y_val)}")
# tmp = []
# for i in range(len(models[0].feature_importances_)):
#     if models[0].feature_importances_[i] > 0:
#         tmp += [i]
# print(f"특징벡터 중요도 : ", tmp)

# 검증 오답
en_hard_pred = ensemble_hard(Y_val, preds)
en_soft_pred = ensemble_soft(X_val, Y_val, models)

failed_file = {}
for i in range(len(Y_val)):
    if(en_soft_pred[i] != Y_val[i]):
        failed_file[filename[i]] = Y_val[i]
print(failed_file)

# 검증 예측 저장
answer = []
for i in range(len(en_soft_pred)):
    answer.append([filename[i], en_soft_pred[i]])
save_csv(answer, 'peminer_val')

# test
X_test = []
cnt = 0
filename = []
peminer_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\PEMINER\테스트데이터\\"
test_file_list = os.listdir(peminer_path)
for key in test_file_list:
    peminer_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\PEMINER\테스트데이터\\" + key
    peminer_vector = PeminerParser(peminer_path).process_report()
    X_test.append(peminer_vector)
    filename += [key]
    cnt += 1
    #if cnt == 20:
    #    break

en_soft_pred = test_ensemble_soft(X_test, models)
answer = []
for i in range(len(en_soft_pred)):
    answer.append([filename[i], en_soft_pred[i]])

save_csv(answer, 'peminer_answer')
