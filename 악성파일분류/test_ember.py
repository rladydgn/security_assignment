import os

from common import read_json, read_label_csv, train, evaluate, ensemble_hard, ensemble_soft, save_csv, test_ensemble_soft

class EmberParser:
    '''
        예제에서 사용하지 않은 특징도 사용하여 벡터화 할 것을 권장
    '''

    def __init__(self, path):
        self.report = read_json(path)
        self.vector = []

    def get_general_file_info(self):
        general = self.report["general"]
        vector = [
            general['size'], general['vsize'], general['has_debug'], general['exports'], general['imports'],
            general['has_relocations'], general['has_resources'], general['has_signature'], general['has_tls'],
            general['symbols']
        ]
        return vector

    # 추가로 작성한 코드

    def get_datadirectories_file_info(self):
        data = self.report["datadirectories"]
        if (len(data) == 15):
            vector = [
                data[0]['size'], data[0]['virtual_address'],
                data[1]['size'], data[1]['virtual_address'],
                data[2]['size'], data[2]['virtual_address'],
                data[3]['size'], data[3]['virtual_address'],
                data[4]['size'], data[4]['virtual_address'],
                data[5]['size'], data[5]['virtual_address'],
                data[6]['size'], data[6]['virtual_address'],
                data[7]['size'], data[7]['virtual_address'],
                data[8]['size'], data[8]['virtual_address'],
                data[9]['size'], data[9]['virtual_address'],
                data[10]['size'], data[10]['virtual_address'],
                data[11]['size'], data[11]['virtual_address'],
                data[12]['size'], data[12]['virtual_address'],
                data[13]['size'], data[13]['virtual_address'],
                data[14]['size'], data[14]['virtual_address']
            ]
        else:
            vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return vector

    def get_header_file_info(self):
        header = self.report["header"]

        vector = [len(header['coff']['characteristics']),
                  len(header['optional']['dll_characteristics']),
                  header['optional']['major_image_version'],
                  header['optional']['minor_image_version'],
                  header['optional']['major_linker_version'],
                  header['optional']['minor_linker_version'],
                  header['optional']['major_operating_system_version'],
                  header['optional']['minor_operating_system_version'],
                  header['optional']['major_subsystem_version'],
                  header['optional']['minor_subsystem_version'],
                  header['optional']['sizeof_headers'],
                  header['optional']['sizeof_heap_commit'],
                  header['optional']['sizeof_code']]
        return vector

    def get_section_file_info(self):
        section = self.report["section"]
        section_num = len(section['sections'])
        section_size_zero = 0
        section_MEM_READ = 0
        section_MEM_EXECUTE = 0
        section_MEM_WRITE = 0
        for i in range(0, section_num):
            if (section['sections'][i]['size'] == 0):
                section_size_zero += 1
            if ("MEM_READ" in section['sections'][i]['props']):
                section_MEM_READ += 1
            if ("MEM_EXECUTE" in section['sections'][i]['props']):
                section_MEM_EXECUTE += 1
            if ("MEM_WRITE" in section['sections'][i]['props']):
                section_MEM_WRITE += 1
        vector = [section_num, section_size_zero, section_MEM_READ, section_MEM_EXECUTE, section_MEM_WRITE]
        return vector

    #
    def get_imports_file_info(self):
        imports = self.report["imports"]
        num = 0
        if (len(imports) == 0):
            pass
        else:
            for i in range(0, len(imports)):
                vector2 = list(imports.keys())
                num += len(imports[vector2[i]])
        vector = [num, 0]
        return vector

    def process_report(self):
        vector = []
        vector += self.get_general_file_info()
        # ================================추가한 함수
        vector += self.get_datadirectories_file_info()
        vector += self.get_header_file_info()
        vector += self.get_section_file_info()
        vector += self.get_imports_file_info()
        '''
            특징 추가
        '''

        return vector

train_label_table = read_label_csv("D:\대학수업\정보보호와시스템보안\과제\데이터\학습데이터_정답.csv")
X_train = []
Y_train = []

# 학습 데이터
cnt = 0
for key, value in train_label_table.items():
    ember_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\EMBER\학습데이터\\" + key + ".json"
    ember_vector = EmberParser(ember_path).process_report()
    X_train.append(ember_vector)
    Y_train.append(value)
    cnt += 1
    #if cnt == 20:
    #    break

# 학습
print("학습데이터 불러오기 완료")
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
    ember_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\EMBER\검증데이터\\" + key + ".json"
    ember_vector = EmberParser(ember_path).process_report()
    X_val.append(ember_vector)
    Y_val.append(value)
    filename += [key]
    cnt += 1
    #if cnt == 20:
    #    break

# 검증
print("검증데이터 불러오기 완료")
preds = []
for model in models:
    preds.append(evaluate(X_val, Y_val, model))

print("입력 벡터 크기 : ", len(X_val[0]))
print(f"rf 훈련 : {models[0].score(X_train, Y_train)}")
#print(f"rf 검증 : {models[0].score(X_val, Y_val)}")
#print(models[3].predict_proba(X_val))

# 앙상블
en_hard_pred = ensemble_hard(Y_val, preds)
en_soft_pred = ensemble_soft(X_val, Y_val, models)

# 검증 오답
failed_file = []
for i in range(len(Y_val)):
    if(en_soft_pred[i] != Y_val[i]):
        failed_file.append([filename[i], Y_val[i]])
#print(failed_file)

# test
X_test = []
cnt = 0
filename = []
ember_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\EMBER\테스트데이터\\"
test_file_list = os.listdir(ember_path)
for key in test_file_list:
    ember_path = "D:\대학수업\정보보호와시스템보안\과제\데이터\EMBER\테스트데이터\\" + key
    ember_vector = EmberParser(ember_path).process_report()
    X_test.append(ember_vector)
    filename += [key]
    cnt += 1
    #if cnt == 20:
    #    break

en_soft_pred = test_ensemble_soft(X_test, models)
answer = []
for i in range(len(en_soft_pred)):
    answer.append([filename[i][:-5], en_soft_pred[i]])

save_csv(answer)