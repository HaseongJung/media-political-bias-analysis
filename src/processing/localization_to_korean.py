import re
import pandas as pd
from tqdm import tqdm 

tqdm.pandas()

if __name__ == "__main__":
    
    # load dataset
    dataset = pd.read_feather("src/data/processed/cleaned.feather")
    print("Cleaned dataset loaded!")
    
    # localization to korean
    # title과 Text 열에서 영문 대소문자, 한글, 숫자, 공백 문자를 제외한 모든 문자 삭제
    dataset['Title'] = dataset['Title'].progress_apply(lambda x: re.sub("[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", '', str(x)))
    dataset['Text'] = dataset['Text'].progress_apply(lambda x: re.sub("[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", '', str(x)))
    print("Localization to Korean completed!")

    # save localized dataset
    dataset.to_feather("src/data/processed/localized.feather")
    print("Localized dataset saved!")