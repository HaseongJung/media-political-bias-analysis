import pandas as pd
from tqdm import tqdm 
tqdm.pandas()


def remove_nullValues(data: pd.DataFrame):

    before_nullValues = data.isnull().sum()
    data = data.dropna(axis = 0)
    print(f"Before Null-values: {before_nullValues}\n")
    print(f'After Null-values: {data.isnull().sum()}\n\n')

    return data

def remove_dup(data: pd.DataFrame):

    before_dupTitle = data["Title"].duplicated().sum()
    before_dupText = data["Text"].duplicated().sum()
    
    data = data.drop_duplicates(subset=['Title'])
    data = data.drop_duplicates(subset=['Text'])

    print(f'Title duplicated: {before_dupTitle} -> {data["Title"].duplicated().sum()}')
    print(f'Text duplicated: {before_dupText} -> {data["Text"].duplicated().sum()}\n')


    return data


if __name__ == "__main__":

    # load merged dataset
    dataset = pd.read_feather("src/data/processed/merged.faether")
    print("Merged dataset loaded!")
    
    # remove missing values
    dataset = remove_nullValues(dataset)

    # 'Text' 열에 대해 길이가 10 이상이며 공백만으로 이루어지지 않은 텍스트를 필터링
    dataset = dataset[dataset['Text'].progress_apply(lambda x: len(x) >= 10 and not x.isspace())]

    # remove duplicateds
    dataset = remove_dup(dataset)

    # save cleaned data
    print(dataset.shape)
    datset = dataset.reset_index(drop=True) 
    dataset.to_feather("src/data/processed/cleaned.feather")
    print("Cleaned dataset saved!")