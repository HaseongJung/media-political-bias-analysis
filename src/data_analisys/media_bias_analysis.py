import os
import pandas as pd
import plotly.express as px
import tqdm

tqdm.tqdm.pandas()





# Get the total dataframe
def get_totalDf(topic_data, emotion_data):

    total_df = topic_data.copy()
    total_df["Sentiment"] = emotion_data["Sentiment"]

    return total_df


# Mapping the emotion values
def mappingEmotion(df):
    
    emotion_mapping = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }
    df["Sentiment"] = df["Sentiment"].map(emotion_mapping)  # Mapping the emotion values

    return df


# Calculate the bias score
def calculate_bias_score(df):
    
    minju_emotion = df[df['Main_topic'] == '더불어민주당']['Average_sentiment'].mean()
    gookmin_emotion = df[df['Main_topic'] == '국민의힘']['Average_sentiment'].mean()
    
    return round(gookmin_emotion - minju_emotion, 4)


# Plot the bar chart
def get_barchart(bias_scores: pd.DataFrame, save_dir, save_name, show: bool=False, save: bool=True):

    fig = px.bar(bias_scores, 
                x='Bias_score', 
                y='News', 
                orientation='h', # 수평 막대 그래프
                width=1400, height=1000,
                title='<b>언론사별 정치적 편향성</b> (2020.09.02 ~ 2023.12.31)',
                labels={'Bias_score': '진보 <-> 보수', 'News': '언론사'},
                range_x=[-0.15, 0.15],
                color='Bias_score', 
                color_continuous_scale='bluered',
                template='plotly_white')
    
    fig.add_vline(x=0, line_width=1, line_color="black")    # 기준선 추가
    fig.update_layout(title_font_size=50 , title_yanchor="auto",    # 타이틀 위치 조정
                    margin_l=200, margin_r=150, margin_t=150, margin_b=100, font_size=15,)    # 여백 조정
    fig.update_xaxes(title_font_size=25, tickfont_size=20, showline=True, linewidth=3, linecolor='black')   # x축 설정
    fig.update_yaxes(title_font_size=25, tickfont_size=20, showline=True, linewidth=3, linecolor='black')   # y축 설정

    # Show the bar chart
    if show:
        fig.show()

    # Save the bar chart
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    fig.write_image(save_dir+save_name)  # 그래프 저장

    return



if __name__ == "__main__":

    # Setting the data path
    data_path = "src/data/processed/"
    topic_data_name = "main-topic-extracted-byBart.feather"
    sentiment_data_name = "added_sentiment.feather"

    # load the dataW
    topic_df = pd.read_feather(data_path + topic_data_name)
    sentiment_df = pd.read_feather(data_path + sentiment_data_name)
    print("Data loaded!")

    # Get the total dataframe
    total_df = get_totalDf(topic_df, sentiment_df)
    print("Total dataframe created!")

    # Drop the rows with NaN values
    total_df.dropna(inplace=True)
    total_df.reset_index(drop=True, inplace=True)
    print("NaN values dropped!")

    # Rename the news company names
    total_df["News"] = total_df["News"].progress_apply(lambda x: "아시아경제" if x == "asiae" else x)
    total_df["News"] = total_df["News"].progress_apply(lambda x: "매일경제" if x == "imaeil" else x)
    total_df["News"] = total_df["News"].progress_apply(lambda x: "KBS" if x == "kbs" else x)
    print("Company names renamed!")

    # Mapping the emotion values
    total_df = mappingEmotion(total_df)
    print("Emotion values mapped!")

    # Group the data by News and Main_topic
    grouped_emotion_data = total_df.groupby(['News', 'Main_topic']).agg(
        Average_sentiment=('Sentiment', 'mean')
        ).reset_index()
    print("Data grouped by News and Main_topic!")
    
    # Calculate the bias score    
    bias_scores = grouped_emotion_data.groupby('News').apply(calculate_bias_score, include_groups=False).reset_index()   # Calculate the bias score
    # bias_scores = grouped_emotion_data.groupby('News').apply(calculate_bias_score).reset_index()    # Calculate the bias score
    bias_scores.columns = ['News', 'Bias_score']    # Rename the columns
    bias_scores.sort_values(by='Bias_score', ascending=False, inplace=True) # Sort the values
    print("Bias scores calculated!")

    # Plot the bar chart
    save_dir = "./src/output/media_bias/"
    save_name = f"언론사별정치적편향성지표_{topic_data_name.split('.')[0]}.png"
    get_barchart(bias_scores, save_dir=save_dir, save_name=save_name)
    print("Bar chart saved!")