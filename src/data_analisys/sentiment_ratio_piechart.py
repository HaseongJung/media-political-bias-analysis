import os
from tqdm import tqdm
import pandas as pd
import plotly.express as px



# Split the data by news company
def splitData_by_company(df, company_list: list):

    data_list = []
    for news in tqdm(company_list, desc="Splitting data by news company..."):
        globals()[f"{news}_df"] = df[df.News == news].reset_index(drop=True)    # drop=True: reset index without adding the original index as a column
        data_list.append(globals()[f"{news}_df"])
        print(f"{news}_df: {globals()[f'{news}_df'].shape}")

    return data_list


# Rename the news company names
def renameCompany(df_name: str, newName: str):

    df = globals()[f"{df_name}"]
    df.loc[:, "News"] = [newName for i in range(len(df))]

    return df


# Plot the pie chart
def get_piechart(df, company: str, show: bool=False, save: bool=True):
    global data_path

    fig = px.pie(df, names='Sentiment', color="Sentiment", color_discrete_map={'Positive': '#004EA1', 'Negative': '#E61E2B'})
    fig.update_layout(title=f"<b><span style='color:brown'>{company}</span> 정치 기사들의 감성 비율</b>", title_font_size=40, title_font_family="나눔고딕",
                       width=800, height=600, margin_t=150)
    fig.update_traces(
        textposition='inside',
        textinfo="label+percent",
        textfont_size=25,
        textfont_color='black',
        marker_line_color='black',
        marker_line_width=1,
    )
    if show:
        fig.show()

    save_dir = "src/output/sentiment_piechart/" 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_path =f"{save_dir}/{company}.png"
    fig.write_image(save_path)


if __name__ == "__main__":

    data_path = "src/data/processed/added_sentiment.feather"
    dataset = pd.read_feather(data_path)  
    print("Data loaded!")

    # Split the data by news company
    company_list = dataset.News.unique()
    data_list = splitData_by_company(dataset, company_list)
    print("Data splitted!")

    # Rename the news company names
    아시아경제_df = renameCompany("asiae_df", "아시아경제")
    매일경제_df = renameCompany("imaeil_df", "매일경제")
    KBS_df = renameCompany("kbs_df", "KBS")
    print("Company names renamed!")

    # Plot the pie chart
    for data in tqdm(data_list, desc="Plotting pie chart..."):
        get_piechart(df=data, company=data.loc[:, "News"][0], show=False, save=True)
    print("Pie-charts are saved!")