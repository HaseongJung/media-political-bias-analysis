import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import pipeline
# from datasets import Dataset

tqdm.pandas()



def extract_mainTopic(row, classifier, threshold):
    
    doc = row.Title + ', ' + row.Text    # 제목과 본문을 합친 문서
    candidate_labels = ['더불어민주당', '국민의힘']   # 후보군 레이블
    result = classifier(doc, candidate_labels)    #  분류기로 문서 분류
    
    scores = result['scores']    # 각 레이블에 대한 점수
    main_topic = result['labels'][0]    # 분류 결과 반환
    
    if abs(scores[0] - scores[1]) < threshold:    # 두 레이블의 점수 차이가 threshold 보다 작을 경우
        return np.nan    # 중립으로 분류
    elif main_topic == '더불어민주당':
        return '더불어민주당'
    elif main_topic == '국민의힘':
        return '국민의힘'
    

def load_model():
    # device setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # model load
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        # batch_size=16,
                        device=0 if device == "cuda" else -1)

    return classifier


if __name__ == "__main__":

    # load dataset
    dataset = pd.read_feather("src/data/processed/localized.feather")
    # hf_dataset = Dataset.from_pandas(dataset)
    print("Dataset loaded!")

    # load model
    classifier = load_model()
    print("Model loaded! (facebook/bart-large-mnli)")

    # extract political party
    print("Extracting main-topic...")
    threshold = 0.2 
    dataset['Main_topic'] = dataset.progress_apply(extract_mainTopic, args=(classifier, threshold), axis=1)  # 정당 추출
    # hf_dataset = hf_dataset.map(lambda batch: {'Main_topic': extract_mainTopic(batch, classifier, threshold)}, batched=True, batch_size=16)
    print("Main-topic extracted!")

    # save main-topic extracted dataset
    dataset.to_feather("src/data/processed/main-topic-extracted-byBart.feather")
    print("Dataset saved!")
