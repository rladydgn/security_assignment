import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential
from keras import backend as K

def recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall
def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision
def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score

def parsing(path):#파싱을 진행하는 함수
    with open(path,'r',encoding='utf-8') as f:#파일을 읽어드리고 ['로그','로그',...] 이런식으로 로그를 구조화
        train=[]
        para=""
        while True:
            l = f.readline() #한줄씩 읽어 옵니다

            if not l:
                break #파일을 전부 읽으면 읽기를 중단합니다.

            if l != "\n":
                para +=l
            else:
                if para!='':
                    if para[:4]=='POST': #Method가 POST인 경우 예외적으로 바디까지 가져옵니다.
                        para+=f.readline()
                    train.append(para)
                    para=""
    return train

# 데이터셋에서 로그 한부위를 따오는 파싱 함수
# def parsing(path):#파싱을 진행하는 함수
#     with open(path,'r',encoding='utf-8') as f:#파일을 읽어드리고 ['로그','로그',...] 이런식으로 로그를 구조화
#         train=[]
#         para =""
#         while True:
#             l = f.readline() #한줄씩 읽어 옵니다
#             if not l:
#                 break #파일을 전부 읽으면 읽기를 중단합니다.
#
#             ################### 변경코드 ##########################
#             if l != "\n":
#                 temp = l.split(" ")
#                 if temp[0] == "POST" or temp[0] == "GET" or temp[0] == "PUT":
#                     para += temp[0] + " " + temp[1]
#             else:
#                 if para!='':
#                     if para[:4]=='POST': #Method가 POST인 경우 예외적으로 바디까지 가져옵니다.
#                         para+= " " + f.readline()
#                     train.append(para)
#                     para=""
#             ##################### 변경코드 ##########################
#     return train

# 데이터셋을 라벨링하는 함수
def dataset(path,mod='train'): #데이터셋을 생성합니다. 파싱한 데이터와 라벨을 생성합니다
    x = parsing(f'{path}norm_{mod}.txt') # mod에 따라 train을 가져올지 test 데이터를 가져올지 결정됩니다.
    y = [0]*len(x) # 정상 라벨 0 을 정상 데이터 개수 만큼 생성
    x += parsing(f'{path}anomal_{mod}.txt')
    y += [1]*(len(x)-len(y)) # 비정상 라벨 1을 비정상 데이터 개수 만큼 생성
    return x, y

# train_x : 정상 학습 데이터 / train_y : 비정상 학습 데이터
# test_x : 정상 테스트 데이터 / test_y : 비정상 테스트 데이터
train_x, train_y = dataset('./', 'train')
test_x, test_y =  dataset('./', 'test')

# --------------- 등장 빈도수가 1인 단어를 제외하기 위한 코드입니다. ---------------
# 문장으로 되어 있는 로그를 고유id를 가진 단어로 변환합니다.
# fit_on_text : 문자데이터 -> 단어리스트
# texts_to_sequences : 단어리스트 -> 고유id 변환
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
train_x_index = tokenizer.texts_to_sequences(train_x)
word_to_index = tokenizer.word_index

rare_words = 0 # 등장 빈도수가 1회인 단어의 수
total_words = len(word_to_index) # 총 단어의 수
for key, value in tokenizer.word_counts.items():
  if(value<=1):
    rare_words += 1
# 전체 단어 수 : 33868
# 등장 빈도수가 1회인 단어 수 : 5920
# RNN 학습에 사용되는 벡터 차원 : 27948
# print("전체 단어 수 : ", total_words)
# print("등장 빈도수가 1회 이하인 단어의 수 : ", rare_words)

# Tokenizer의 파라미터 변수를 total_words - rare_words + 1로 둠으로서
# 로그에 등장한 단어에 제한을 둔다. 등장 빈도수가 2번 이상인 단어만 사용한다.
tokenizer = Tokenizer(num_words = total_words - rare_words + 1)
# train_x의 문자 데이터를 리스트 형태로 변환한다.
tokenizer.fit_on_texts(train_x)
# 리스트로 저장된 단어들을 고유id로 변환한다.
train_x_index = tokenizer.texts_to_sequences(train_x)
word_to_index = tokenizer.word_index
# --------------- 등장 빈도수가 1인 단어를 제외하기 위한 코드입니다. ---------------


# --------------- RNN 학습에 사용될 데이터 전처리 과정입니다. -------------------
# 훈련 데이터의 크기는 항상 max_length 로 고정합니다.
# 모든 로그의 길이를 모두 같게 만들기 위해 패딩을 더합니다.
max_length = max(len(i) for i in train_x_index)
train_x_pad = pad_sequences(train_x_index, maxlen=max_length)
vocab_size = len(word_to_index) + 1
# --------------- RNN 학습에 사용될 데이터 전처리 과정입니다. -------------------


# --------------- RNN 학습 과정입니다. ------------------------------------
# 임베딩 과정을 통해 토큰화된 단어를 좌표평면으로 매핑합니다.
model = Sequential()
model.add(Embedding(vocab_size, 32)) # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid')) # 활성 함수

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1score])
train_y = np.array(train_y)
# 5%는 검증에 사용    # loop은 반복횟수를 의미한다.
loop = 7
history = model.fit(train_x_pad, train_y, epochs=loop, batch_size=64, validation_split=0.05)

test_x_index = tokenizer.texts_to_sequences(test_x)
test_x_pad = pad_sequences(test_x_index, maxlen = max_length)
test_y = np.array(test_y)
_loss, _acc, _precision, _recall, _f1score = model.evaluate(test_x_pad, test_y)
print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(_loss, _acc, _precision, _recall, _f1score))
# print("\n 테스트 정확도: %.4f" % (model.evaluate(test_x_pad, test_y)[1]))
# --------------- RNN 학습 과정입니다. ------------------------------------


# --------------- 학습 모델의 테스트 결과를 그래프로 나타내는 과정입니다. ------------
epochs = range(1, loop+1)
plt.plot(epochs, history.history['loss'], 'ro-')
plt.plot(epochs, history.history['val_loss'], 'bo-')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper right')
plt.grid(True)
plt.show()
# --------------- 학습 모델의 테스트 결과를 그래프로 나타내는 과정입니다. ------------