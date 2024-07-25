import pandas as pd
import requests # HTTP 요청을 보내기 위한 모듈
from tqdm import tqdm
tqdm.pandas()




def average(list):  # 각 문서의 평균 감성 score 반환  # 리스트의 평균 값을 계산하는 함수
    return sum(list)/len(list)


def get_sentiment(row: pd.Series, s_word, values, score):  # 감성 분석 함수
    combined_text = ' '.join(row['Text']) + ' ' + ' '.join(row['Title'])

    temp_s_word = []    # 문서에서 발견된 감성 단어 리스트 초기화
    temp_value = []     # 문서에서 발견된 감성 단어의 polarity 값 리스트 초기화
    for s in sentiword: 
        if s['word'] in combined_text:  # 현재 문서의 단어와 감성 사전의 단어 비교
            if len(s['word']) > 1:  # 한 글자 이상의 감성 단어만 추출
                temp_s_word.append(s['word'])   # 해당하는 감성 단어 추가
                temp_value.append(int(s['polarity']))   # 해당 단어의 polarity 값 추가
    s_word.append(temp_s_word)  # 해당하는 감성 단어 추가
    values.append(temp_value)   # 해당 단어의 polarity 값 추가

    try:
        score.append(average(temp_value))   # 각 문서의 평균 polarity 추가
    except ZeroDivisionError:
        score.append(int(0))

    return s_word, values, score


def classify_sentiment(row):  # 감성 점수를 기반으로 감성 분류
    if len(row["values"]) < 2:  # 감성사전에 있는 단어가 2개 미만인 경우
        return "Neutral"
    elif row["score"] >= 0.8:
        return "Positive"
    elif row["score"] < 0:
        return "Negative"
    else:
        return "Neutral"




if __name__ == "__main__":
    
    # load dataset
    dataset = pd.read_feather("src/data/processed/stopwords-removed.feather")
    print("Stopwords removed dataset loaded!")
    
    # load sentiment dictionary
    url = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/KnuSentiLex/data/SentiWord_info.json"
    sentiword = requests.get(url).json()
    # print(sentiword)

    s_word = []  
    values = []
    score = []

    dataset.progress_apply(lambda row: get_sentiment(row, s_word, values, score), axis=1)  # 데이터프레임의 각 행에 대해 get_sentiment 함수 적용

    # 결과 데이터프레임에 감성분석 결과 삽입
    dataset = dataset.assign(sentiword=s_word, values=values, score=score) # 결과 데이터프레임에 삽입
    print("Sentiment analysis completed!")

    # 감성 분류
    print("Classifying sentiment...")
    dataset["Sentiment"] = dataset.progress_apply(classify_sentiment, axis=1)  # 데이터프레임의 각 행에 대해 classify_sentiment 함수 적용
    print("Sentiment classified!")

    # save sentiment classified dataset
    dataset.to_feather("src/data/processed/added_sentiment.feather")
    print("Sentiment classified dataset saved!")