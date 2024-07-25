import os
import pandas as pd


def load_datasets(crawledData_path):
    
    # load dataset_names
    dataset_names = os.listdir(crawledData_path)
    dataset_names = [data for data in dataset_names if "total" not in data]
    print(f"data_names({len(dataset_names)}): {dataset_names}")

    # load datsets
    datasets = []
    for dataset_name in dataset_names:
        globals()[f"{dataset_name.split('.')[0]}_df"] = pd.read_pickle(crawledData_path + dataset_name)
        datasets.append(globals()[f"{dataset_name.split('.')[0]}_df"])

    return dataset_names, datasets






if __name__ == "__main__":

    # load datasets
    crawledData_path = "src/data/raw/"
    dataset_names, datasets = load_datasets(crawledData_path)
    print("Datasets loaded!")

    # merge datasets
    total_df = pd.concat(datasets)
    print(total_df) 
    
    # save merged dataset
    total_df.to_feather('src/data/processed/merged.faether')
    print("Merged dataset saved!")