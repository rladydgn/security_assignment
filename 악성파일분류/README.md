# 정보보호와 시스템보안 프로젝트 : 악성파일분류
목적 : 파일의 악성여부 분류

팀원 : 김용후(소프트/20181589) / 문예찬(정보보안암호수학과/20192230) / 안성열(소프트/20163121)

## 개요
PEMINER, PESTUDIO, EMBER 에서 유의미한 특징들을 추출하여 기계학습을 이용해 모델을 생성하여 학습시킵니다. 학습시킨 모델을 이용해 악성파일과 정상파일을 분류합니다.

가능한 높은 정확도를 갖는 모델을 생성하는 것이 목표입니다.

## 분류 방법

PEMINER - 모든 특징들이 실수형태로 저장되어 있어 전처리가 편했습니다. randomforest, lightgbm, adaboost를 이용하여 학습과 검증을 진행하였습니다. 각각 검증 정확도는 0.952, 0.9507, 0.901이 나왔습니다. 이를 하드보팅, 소프트보팅 방식으로 앙상블을 진행한 결과 각각 0.9513, 0.9541의 정확도가 나왔습니다. f1 score는 각각 0.9667, 0.9685가 나왔습니다. 소프트 보팅의 결과가 더 좋아 소프트 보팅을 이용하였습니다.

EMBER - 히스토그램, 스트링을 특징벡터로 사용할 경우 오히려 정확도가 낮아져 제외시켰습니다. 대부분의 특징들이 실수형태로 되어있습니다. general, datadirectories, header, section 영역의 실수들을 특징벡터로 이용하였고, import에 있는 파일의 개수, export(모두 0)에 있는 파일의 개수를 추가로 특징벡터로 이용하였습니다. PEMINER과 마찬가지로 randomforest, lightgbm, adaboost를 이용하여 학습과 검증을 진행하였습니다. 각각 검증 정확도는 0.9492, 0.9442, 0.8812가 나왔습니다.  하드보팅, 소프트보팅의 정확도는 각각 0.9459, 0.9508이 나왔습니다. f1 score는 각각 0.9631, 0.9663이 나왔습니다. 소프트 보팅의 결과가 더 좋아 소프트 보팅을 이용하였습니다.

PESTUDIO - 특징 대부분이 문자열 형태인 json파일 입니다. 

image 파일 내에 overview, indicators, resources, version을 포함한 정보들이 들어있습니다.
1. overview는 파일의 전반적인 내용을 요약한 정보입니다.
- overview에서는 description과 file-type을 feature로 뽑았습니다.
- description은 파일이 어떤 역할을 하는지 설명하는 정보입니다. 정상 파일의 경우 익숙한 단어인 Adobe Flash 등의 단어가 등장합니다. 그러나 악성 파일의 경우 알 수 없는 단어가 등장합니다.

2. indicators는 pestudio로 분석한 결과를 요약한 정보입니다. 
- 분석기준에 따라 위험등급이 나누어져 있습니다. 위험 등급이 1, 2 인 기준의 내용을 feature로 뽑았습니다.

3. resources는 사용한 resource의 정보를 담고 있습니다.
- 사용한 resource의 language를 fearture로 뽑았습니다. 정상 파일의 경우 English-United States, neutral 값이 대부분이나, 악성파일은 특정 나라의 언어로만 기술되는 경우가 빈번했습니다.

4. version은 파일의 버전 정보를 담고 있습니다.
- 특히 OrginalFilename 과 InternalName을 담고 있는데, 만약 두 내용이 다른 경우 악성코드로 의심해볼 수 있습니다. 그러나 OriginalFilename, InternalName 정보가 없는 json 파일이 많고 처리 과정이 복잡하여 feature로 넣지 못했습니다.



## 파일 
common - 공통적으로 사용되는 함수입니다.

test_ember - ember json 파일의 정보를 추출하여 학습시킨후 검증과 테스트데이터를 예측하는 파일입니다.

test_peminer - peminer json 파일의 정보를 추출하여 학습시킨후 검증과 테스트데이터를 예측하는 파일입니다.

test_pestudio - pestudio json 파일의 정보를 추출하여 학습시킨후 검증과 테스트데이터를 예측하는 파일입니다.

ensemble - 위의 세 종류의 파일을 학습하여 예측한 값을 하드보팅하여 최종적으로 검증을 진행하고 정답을 예측하는 파일입니다.

## 결론

PEMINER - rf, lgbm, adaboost 를 사용하여 소프트 보팅

EMBER - rf, lgbm, adaboost 를 사용하여 소프트 보팅

PESTUDIO - rf 를 사용하여 학습

세 결과를 하드보팅하여 최종적으로 테스트데이터 셋에 대한 예측값을 얻었습니다.

세 결과를 종합한 검증데이터에 대한 정확도는 0.956이 나왔고, f1 score는 0.97이 나왔습니다.

예측한 결과값은 test.csv 파일에 저장되어있습니다.

![캡처1](https://user-images.githubusercontent.com/39542757/146369378-ef5db5f4-1a98-4040-8183-1adfe59f6c08.PNG)
