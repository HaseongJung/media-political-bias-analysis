import pandas as pd
import parmap
import requests




def load_stopwords():
    url = "https://gist.githubusercontent.com/chulgil/d10b18575a73778da4bc83853385465c/raw/a1a451421097fa9a93179cb1f1f0dc392f1f9da9/stopwords.txt"  # 불용어 사전
    response = requests.get(url)    # 불용어 사전 다운로드
    data = response.content.decode("utf-8") # 불용어 사전을 utf-8로 디코딩
    stopwords = data.split("\n")    # 불용어 사전을 줄바꿈을 기준으로 분리
    stopwords = [word for word in stopwords if word]    # 빈 문자열 제거
    print(f'stopwords: {stopwords}')

    return stopwords

def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if (token not in stopwords) and (len(token) > 1)] # 한 글자 초과인 단어만 추출, 불용어 사전으로 제거



if __name__ == "__main__":

    # load dataset
    dataset = pd.read_feather("src/data/processed/pos-filtered.feather")
    print("Dataset loaded!")

    # load stopwords
    stopwords = load_stopwords()
    print("Stopwords loaded!")

    # remove stopwords
    print("Removing stopwords...")
    dataset["Title"] = parmap.map(remove_stopwords, dataset["Title"], stopwords, pm_pbar=True)    # parmap.map 함수를 사용하여 remove_stopwords 함수를 병렬로 적용
    dataset["Text"] = parmap.map(remove_stopwords, dataset["Text"], stopwords, pm_pbar=True)  # jpuyter notebook에서는 실행이 되지 않음, rmStopwords.py 파일에서 실행
    print("Stopwords removed!")

    # save stopwords removed dataset
    dataset.to_feather("src/data/processed/stopwords-removed.feather")