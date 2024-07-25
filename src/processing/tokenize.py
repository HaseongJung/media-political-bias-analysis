import pandas as pd
from konlpy.tag import Mecab

import multiprocessing as mp
import parmap

if __name__ == "__main__":

    num_workers = mp.cpu_count()    # 병렬 처리를 위한 worker 수 설정
    
    # load dataset
    dataset = pd.read_feather("src/data/processed/localized.feather")
    print("Localized dataset loaded!")

    # tokenize
    mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

    dataset["Title"] = parmap.map(mecab.morphs, dataset["Title"], pm_pbar=True, pm_processes=num_workers)  # parmap.map 함수를 사용하여 mecab.morphs 함수를 병렬로 적용
    dataset["Text"] = parmap.map(mecab.morphs, dataset["Text"], pm_pbar=True, pm_processes=num_workers)
    print(dataset)

    # save tokenized dataset
    dataset.to_feather("src/data/processed/tokenized.feather")
    print("Tokenized dataset saved!")