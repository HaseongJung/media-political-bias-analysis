import pandas as pd
import parmap
import multiprocessing as mp
num_workers = mp.cpu_count()    # 병렬 처리를 위한 worker 수 설정




# pos filtering
def pos_filtering(pos_list: list): 

    desired_pos = ['NNG', 'NNP', 'VV', 'VA', 'MAG', 'NR', 'MM']     # 일반명사, 고유명사, 동사, 형용사, 일반부사, 수사, 관형사
    
    toekns = []
    for pos in pos_list:
        if pos[1] in desired_pos:
            toekns.append(pos[0])

    return toekns




if __name__ == "__main__":

    dataset = pd.read_feather("src/data/processed/pos-tagged.feather")
    print("Dataset loaded!")

    # pos filtering (only nouns, verbs, adjectives, adverbsdeterminer, numeral) 
    dataset['Title'] = parmap.map(pos_filtering, dataset['Title'], pm_pbar=True, pm_processes=mp.cpu_count()) 
    dataset['Text'] = parmap.map(pos_filtering, dataset['Text'], pm_pbar=True, pm_processes=mp.cpu_count())   
    print("Pos filtering done!")

    # save pos-filtered dataset
    dataset.to_feather("src/data/processed/pos-filtered.feather")
    print("Pos-filtered dataset saved!")