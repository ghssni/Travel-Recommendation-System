import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

def run():
    # EDA title
    st.title('Exploratory Data Analysis')
    # Load Clean Data
    df = pd.read_excel('data_clean_tour.xlsx')
    # Select Box 
    pilihan = st.selectbox('Choose Analysis',
                            ['Place by City','Place by Category','Place Category Rating'])

    if pilihan == 'Place by City':
        st.write('## Place by City')
        fig = plt.figure(figsize=(9,5))
        place_city = df.groupby('city')['place_name'].nunique().reset_index()
        place_city = place_city.sort_values(by ='place_name', ascending=False)
        ax = sns.barplot(data=place_city, x='city', y='place_name')
        plt.title("Place by City")
        plt.xlabel("City")
        plt.ylabel("Total Count")
        for i in ax.containers:
            ax.bar_label(i,)
        st.pyplot(fig)
        st.write('#### Insight')
        st.markdown('''
                    There are 5 cities in the database and each city represents each province on the island of Java. 
                    In the data, it is found that Yogyakarta is the city with the most tourist attractions with a total of 126  attractions based on the city.
                    ''')
    elif pilihan == 'Place by Category':
        st.write('## Place by Category')
        fig = plt.figure(figsize=(10,6))
        tour_cat = df['category'].value_counts().reset_index() 
        plt.title("Place by Category")
        plt.pie(tour_cat['category'], labels=tour_cat['index'], autopct='%1.1f%%')
        plt.show()
        st.pyplot(fig)
        st.write('#### Insight')
        st.markdown('''
                    Amusement parks are the tourism category with the highest number in the overall data with a total of 3024 entertainment venues accounting for "30.5%" of the tourism data population.
                    ''')
    elif pilihan == 'Place Category Rating':
        st.write('## Place Category Rating')
        all_cat_count = df.groupby('category')['user_rating'].count().reset_index()
        all_cat_count = all_cat_count.sort_values(by = 'user_rating', ascending=False)

        all_cat_mean = df.groupby('category')['user_rating'].mean().reset_index()
        all_cat_mean = all_cat_mean.sort_values(by = 'user_rating', ascending=False)
        # Define figure and axes
        fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

        # Bar chart count user rating
        sns.barplot(data=all_cat_count, x='user_rating', y='category', ax=axes[0], ci=None)
        axes[0].set_title("Count Customers Rating")
        # axes[0].set_xticklabels(None)

        # Menambahkan label di dalam bar chart count user rating
        for i, val in enumerate(all_cat_count['user_rating']):
            axes[0].text(val, i, f'{val}', va='center', ha='right', color='black', fontsize=8)

        # Bar chart mean user rating
        sns.barplot(data=all_cat_mean, x='user_rating', y='category', ax=axes[1], ci=None)
        axes[1].set_title("Average Customers Rating")

        # Menambahkan label di dalam bar chart mean user_rating
        for i, val in enumerate(all_cat_mean['user_rating']):
            axes[1].text(val, i, f'{val:.2f}', va='center', ha='right', color='black', fontsize=8)
        # Figure title
        fig.suptitle("Rating Place per Category")
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        st.write('#### Insight')
        st.markdown('''
                    Place of worship that occupies the 2nd favorite place which is inversely proportional to the number of ratings obtained at 382 or the 2nd lowest in the data.
                    ''')

if __name__ == '__main__':
  run()