import os, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def pestudio_data_parsing(filepath):
    data = []
    json_file_list = os.listdir(filepath)

    for json_file in json_file_list:
        with open(filepath+json_file, 'r', encoding='cp949') as jf:
            json_data = json.load(jf)
        temp = ""

        if len(json_data['image']['indicators']) == 1:
            temp += 'truevirus '
            continue
        if json_data['image']['overview']['description'] != None:
            temp += json_data['image']['overview']['description'] + " "
        else:
            temp += 'none_description' + " "

        if json_data['image']['overview']['file-type'] != None:
            temp += json_data['image']['overview']['file-type'] + " "
        else:
            temp += 'none_file-type' + " "

        for indicator in json_data['image']['indicators']['indicator']:
            if indicator['@severity'] == ('1' or '2'):
                temp += str(indicator['@detail']) + " "
            if indicator['@xml-id'] == '1120':
                if int(indicator['@detail'].split(" ")[1].split("/")[0]) > 30:
                    temp += 'truevirus '

        try:
            for resource in json_data['image']['resources']['instance']:
                if resource['@language']:
                    temp += resource['@language'] + " "
        except:
            temp += 'none_language '

        data.append(temp)

    return data

def pestudio_data_labeling(filepath, fp):
    json_file_list = os.listdir(filepath)
    answers = {}
    remove = fp.readline()

    while True:
        line = fp.readline()
        if not line:
            break
        temp = line.split(",")
        answers[temp[0]] = temp[1].replace("\n", "")

    data = []
    label = []
    for json_file in json_file_list:
        with open(filepath+json_file, 'r', encoding='cp949') as jf:
            json_data = json.load(jf)

        temp = ""

        if len(json_data['image']['indicators']) == 1:
            temp += 'truevirus '
            continue
        if json_data['image']['overview']['description'] != None:
            temp += json_data['image']['overview']['description'] + " "
        else:
            temp += 'none_description' + " "

        if json_data['image']['overview']['file-type'] != None:
            temp += json_data['image']['overview']['file-type'] + " "
        else:
            temp += 'none_file-type' + " "

        for indicator in json_data['image']['indicators']['indicator']:
            if indicator['@severity'] == ('1' or '2'):
                temp += str(indicator['@detail']) + " "
            if indicator['@xml-id'] == '1120':
                if int(indicator['@detail'].split(" ")[1].split("/")[0]) > 30:
                    temp += "truevirus "
        try:
            for resource in json_data['image']['resources']['instance']:
                if resource['@language'] :
                    temp += resource['@language'] + " "
        except:
            temp += 'none_language '

        label.append(answers[json_file[:-5]])
        data.append(temp)

    return data,label

def predict_label(filepath, pred, fp):
    json_file_list = os.listdir(filepath)
    for i in range(0, len(json_file_list)):
        fp.write(json_file_list[i][:-5] + "," + pred[i]+"\n")

def vectorize(train_x, test_x, val_x):
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)
    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)
    val_vec = tf.transform(val_x)
    return train_vec, test_vec, val_vec

def train(train_vec,train_y):
    rf = RandomForestClassifier()
    rf.fit(train_vec,train_y)
    return rf

def evaluate(test_y,test_vec,rf):
    pred = rf.predict(test_vec)
    print(accuracy_score(test_y, pred))
    return pred

def test(test_vec, rf):
    pred = rf.predict(test_vec)
    return pred

def run():
    fp1 = open('학습데이터_정답.csv', 'r', encoding='cp949')
    fp2 = open('검증데이터_정답.csv', 'r', encoding='cp949')
    fp3 = open('pestudio_검증결과.csv', 'w', encoding='cp949')
    fp4 = open('pestudio_테스트결과.csv', 'w', encoding='cp949')

    train_data_path = 'C:\\Users\\zxcv1\\Desktop\\data\\PESTUDIO\\학습데이터\\'
    test_data_path = 'C:\\Users\\zxcv1\\Desktop\\data\\PESTUDIO\\검증데이터\\'
    no_label_data_path = 'C:\\Users\\zxcv1\\Desktop\\data\\PESTUDIO\\테스트데이터\\'

    train_data_path2 = 'C:\\Users\\zxcv1\\Desktop\\train\\'
    test_data_path2 = 'C:\\Users\\zxcv1\\Desktop\\test\\'
    no_label_data_path2 = 'C:\\Users\\zxcv1\\Desktop\\val\\'

    train_x, train_y = pestudio_data_labeling(train_data_path, fp1)
    test_x, test_y = pestudio_data_labeling(test_data_path, fp2)
    val_x = pestudio_data_parsing(no_label_data_path)

    train_vec, test_vec, val_vec= vectorize(train_x, test_x, val_x)
    rf = train(train_vec, train_y)

    # label 예측
    pred_test = evaluate(test_y, test_vec, rf)
    pred_val = test(val_vec, rf)

    # csv 파일로 정리
    predict_label(test_data_path, pred_test, fp3)
    predict_label(no_label_data_path, pred_val, fp4)


    fp1.close()
    fp2.close()
    fp3.close()
    fp4.close()

run()
