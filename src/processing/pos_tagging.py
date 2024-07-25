import pandas as pd

from konlpy.tag import Mecab

# for parallel processing
import multiprocessing as mp
import parmap




# pos tagging
def pos_tagging(token: list):

    mecab = Mecab(dicpath='C://mecab//mecab-ko-dic')    # Mecab 형태소 분석기 불러오기

    return mecab.pos(''.join(token))    # ' '.join(token)으로 리스트를 문자열로 변환하여 입력




if __name__ == "__main__":

    num_workers = mp.cpu_count()    # 병렬 처리를 위한 worker 수 설정

    dataset = pd.read_feather("src/data/processed/tokenized.feather")
    print("Dataset loaded!")

    # pos tagging
    dataset["Title"] = parmap.map(pos_tagging, dataset["Title"], pm_pbar=True, pm_processes=num_workers)  
    dataset["Text"] = parmap.map(pos_tagging, dataset["Text"], pm_pbar=True, pm_processes=num_workers)   
    print("Pos tagging done!")

    # save pos-tagged dataset
    dataset.to_feather("src/data/processed/pos-tagged.feather")  # .feather: best format for large dataframe
    print("Pos-tagged dataset saved!")
