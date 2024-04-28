import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import cv2
import requests
import numpy as np
# Load model vectorizer
with open('tfidf_matrix.pkl', 'rb') as file_1:
  tfidf_matrix = pickle.load(file_1)

# Load Data Model
dfm = pd.read_excel('data_model.xlsx')

# Load Data Package
dfp = pd.read_csv('package.csv')

# Load Place Data
df_place = pd.read_excel('final.xlsx')
place_dict = df_place.groupby('city')['place_name'].apply(list).to_dict()


# Create A Function for Text Preprocessing
def text_preprocessing(text):
  # Define Stopwords
  stpwds_id = list(set(stopwords.words('indonesian')))
  new_word = ['bahasa', 'inggris', 'selatan', 'utara', 'barat', 'timur', 'km', 'ha', 'meter', 'tinggi', 'lantas', 'sih',
              'dulunya', 'budget', 'mayoritas', 'heran', 'kaum', 'unjung', 'kawula', 'karcis', 'parkir', 'bangun', 'ciri',
              'a', 'm', 'jalan', 'kota', 'buka', 'None']
  stpwds_id.extend(new_word)

  # Define Stemming
  stemmer = StemmerFactory().create_stemmer()

  # Case folding
  text = text.lower()

  # Number Removal
  text = re.sub(r"[0-9]", " ", text)

  # Newline removal (\n)
  text = re.sub(r"\\n", " ",text)

  # Whitespace removal
  text = text.strip()

  # Non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
  text = re.sub("[^A-Za-z\s']", " ", text)

  # Tokenization
  tokens = word_tokenize(text)

  # Stopwords removal
  tokens = [word for word in tokens if word not in stpwds_id]

  # Stemming
  tokens = [stemmer.stem(word) for word in tokens]

  # Combining Tokens
  text = ' '.join(tokens)
  return text

def sorting(mv):
    # Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(dfm['preprocess'])

    # Convert tfidf_matrix to DataFrame with place name as index
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=dfm['place_name'], columns=tfidf_vectorizer.get_feature_names_out())

    # Convert tfidf_df to a sparse matrix
    tfidf_matrix_sprase = scipy.sparse.csr_matrix(tfidf_df.values)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix_sprase)

    # Create DataFrame with movie titles as index and columns
    cosine_sim_df = pd.DataFrame(cosine_sim, index=tfidf_df.index, columns=tfidf_df.index)

    # Ambil kota dari tempat yang dipilih
    city = dfm.loc[dfm['place_name'] == mv, 'city'].iloc[0]
    # Compute the similarity scores and show the top 20 recommended places
    top_sims = cosine_sim_df[mv].sort_values(ascending=False).iloc[0:25]  # Change index from 0 to 1
    # Format the output as a list of tuples
    output = [(top_sims.index[i], top_sims.values[i]) for i in range(len(top_sims))]
    # Print the output
    recommendation_list = []
    count = 0
    for i, (place, sim) in enumerate(output):
        place_city = dfm.loc[dfm['place_name'] == place, 'city'].iloc[0]
        if place_city == city:  # Periksa apakah kota tempat sama dengan kota input
            count += 1
            recommendation_list.append([f'{count}.',place])
            if count == 7:
                break
          
    return recommendation_list

def trip_recom(recommendation_names, place, kota):
    if isinstance(recommendation_names,list) and isinstance(place,str):
        recommendation_names.append(place)
        recommendation_names = ' '.join(recommendation_names)
        new = text_preprocessing(recommendation_names)
        new_data = pd.DataFrame([{'package': 0, 
                                'city': kota, 
                                'place_tourism1':'a', 
                                'place_tourism2':'a', 
                                'place_tourism3':'a',
                                'place_tourism4':'a', 
                                'place_tourism5':'a', 
                                'places': new
                                }])
        # Vectorization
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(dfp['places'].append(new_data['places']))

        # Convert tfidf_matrix to DataFrame with place name as index
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=dfp['places'].append(new_data['places']), columns=tfidf_vectorizer.get_feature_names_out())

        # Convert tfidf_df to a sparse matrix
        tfidf_matrix_sparse = scipy.sparse.csr_matrix(tfidf_df.values)

        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix_sparse)

        # Create DataFrame with movie titles as index and columns
        cosine_sim_df = pd.DataFrame(cosine_sim, index=tfidf_df.index, columns=tfidf_df.index)

        # Compute the similarity scores and show the top 5 recommended places
        top_sims = cosine_sim_df[new].drop(index=new).sort_values(ascending=False).iloc[:29]

        # Format the output as a list of tuples
        output = [(top_sims.index[i], top_sims.values[i]) for i in range(len(top_sims))]

        recommendation_list = []
        count = 0
        for i, (place, sim) in enumerate(output):
            place_city = dfp.loc[dfp['places'] == place, ['city', 'package']].iloc[0]  # Mengambil city dan package
            if place_city['city'] == kota:  # Periksa apakah kota tempat sama dengan kota input
                count += 1
                recommendation_list.append([f'{count}.', place, place_city['package']])  # Menambahkan package ke dalam list
                if count == 3:
                    break
        return recommendation_list
    
def sorting_desc(user_input, city, dfm):
    preprocess = text_preprocessing(user_input)
    new_data = pd.DataFrame([{'place_id': 1000,
                             'place_name': 'place_inf',
                             'category': 'inference',
                             'city': city,
                             'description': user_input,
                             'preprocess': preprocess,
                             }])

    # Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(dfm['preprocess'].append(new_data['preprocess']))

    # Convert tfidf_matrix to DataFrame with place name as index
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=dfm['place_name'].append(new_data['place_name']), columns=tfidf_vectorizer.get_feature_names_out())

    # Convert tfidf_df to a sparse matrix
    tfidf_matrix_sparse = scipy.sparse.csr_matrix(tfidf_df.values)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix_sparse)

    # Create DataFrame with movie titles as index and columns
    cosine_sim_df = pd.DataFrame(cosine_sim, index=tfidf_df.index, columns=tfidf_df.index)

    # Compute the similarity scores and show the top 5 recommended places
    top_sims = cosine_sim_df['place_inf'].drop(index='place_inf').sort_values(ascending=False).iloc[:20]

    # Format the output as a list of tuples
    output = [(top_sims.index[i], top_sims.values[i]) for i in range(len(top_sims))]
    # Print the output
    # print(f'You like {user_input}, so based on our recommender system, We recommend you to go to:')
    recommendation_list = []
    count = 0
    for i, (place, sim) in enumerate(output):
        place_city = dfm.loc[dfm['place_name'] == place, 'city'].iloc[0]
        if place_city == city:  # Periksa apakah kota tempat sama dengan kota input
            count += 1
            recommendation_list.append([f'{count}.',place])
            if count == 7:
                break
    return recommendation_list

def recom_trip_desc(recommendation_names,kota):
    if isinstance(recommendation_names,list):
        recommendation_names = ' '.join(recommendation_names)
        new = text_preprocessing(recommendation_names)
        new_data = pd.DataFrame([{'package': 0, 
                                'city': kota, 
                                'place_tourism1':'a', 
                                'place_tourism2':'a', 
                                'place_tourism3':'a',
                                'place_tourism4':'a', 
                                'place_tourism5':'a', 
                                'places': new
                                }])
        # Vectorization
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(dfp['places'].append(new_data['places']))

        # Convert tfidf_matrix to DataFrame with place name as index
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=dfp['places'].append(new_data['places']), columns=tfidf_vectorizer.get_feature_names_out())

        # Convert tfidf_df to a sparse matrix
        tfidf_matrix_sparse = scipy.sparse.csr_matrix(tfidf_df.values)

        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix_sparse)

        # Create DataFrame with movie titles as index and columns
        cosine_sim_df = pd.DataFrame(cosine_sim, index=tfidf_df.index, columns=tfidf_df.index)

        # Compute the similarity scores and show the top 5 recommended places
        top_sims = cosine_sim_df[new].drop(index=new).sort_values(ascending=False).iloc[:20]

        # Format the output as a list of tuples
        output = [(top_sims.index[i], top_sims.values[i]) for i in range(len(top_sims))]
        # Print the output
        # print(f'You like {recommended_places}, so based on our recommender system, We recommend you to go to:')
        recommendation_list = []
        count = 0
        for i, (place, sim) in enumerate(output):
            place_city = dfp.loc[dfp['places'] == place, ['city', 'package']].iloc[0]  # Mengambil city dan package
            if place_city['city'] == kota:  # Periksa apakah kota tempat sama dengan kota input
                count += 1
                recommendation_list.append([f'{count}.', place, place_city['package']])  # Menambahkan package ke dalam list
                if count == 3:
                    break
        return recommendation_list

def fetch_image(url):
    try:
        # Mengambil gambar dari URL menggunakan library requests
        response = requests.get(url)
        # Memastikan respons adalah OK
        if response.status_code == 200:
            # Membaca gambar menggunakan OpenCV
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        else:
            st.error(f"Failed to fetch image from URL: {url}")
            return None
    except Exception as e:
        st.error(f"Error fetching image from URL: {e}")
        return None

def resize_and_display_image(url, target_size=(400, 400)):
    # Mengambil gambar dari URL
    image = fetch_image(url)
    if image is not None:
        try:
            # Melakukan resizing gambar
            resized_image = cv2.resize(image, target_size)
            # Menampilkan gambar yang sudah diresize
            st.image(resized_image, channels="BGR", width=200)
        except Exception as e:
            st.error(f"Error resizing image: {e}")

def run():
  
  # Select Box
  pilihan = st.selectbox('Choose Input Type',
                          ['Input by Place Name','Input by Description'])

  if pilihan == 'Input by Place Name':
      # Choose City
      kota = st.selectbox('Pilih Kota Tujuan',list(place_dict.keys()))
      place = st.selectbox('Pilih Tempat Wisata',place_dict[kota])
      if st.button("Cari Rekomendasi"):
        st.write('## Rekomendasi Tempat')
        st.write('Rekomendasi berdasarkan kemiripan dengan ', place, ' ', 'di ', kota,':')
        # st.write('----')
        recommendation_list = sorting(place)
        recommendation_names = [recom[1] for recom in recommendation_list[1:]]
        recommendation_for_package = [recom[1] for recom in recommendation_list]

        # Mengambil informasi dari df_place berdasarkan nama tempat yang ada dalam recommendation_names
        recommended_places_info = df_place[df_place['place_name'].isin(recommendation_names)][['url', 'place_name', 'city','rating','price','category']]
        # st.write(recommended_places_info)
        # Menampilkan data dalam format yang diinginkan di Streamlit
        row1 = st.columns(3)
        row2 = st.columns(3)

        for i in range(len(recommended_places_info)):
            if i < 3:
                with row1[i]:
                    resize_and_display_image(recommended_places_info.iloc[i]['url'])
                    # st.image( recommended_places_info.iloc[i]['url'], width=200)
                    st.write("#### ", recommended_places_info.iloc[i]['place_name'] ,' - ',recommended_places_info.iloc[i]['city'])
                    st.write('Kategori Tempat: ',str(recommended_places_info.iloc[i]['category']))
                    st.write('Biaya Masuk: ',str(recommended_places_info.iloc[i]['price']))
                    st.write('Rating Tempat: ', str(recommended_places_info.iloc[i]['rating']))
                    
            else:
                with row2[i - 3]:
                    resize_and_display_image(recommended_places_info.iloc[i]['url'])
                    # st.image( recommended_places_info.iloc[i]['url'], width=200)
                    st.write("#### ", recommended_places_info.iloc[i]['place_name'] ,' - ',recommended_places_info.iloc[i]['city'])
                    st.write('Kategori Tempat: ',str(recommended_places_info.iloc[i]['category']))
                    st.write('Biaya Masuk: ',str(recommended_places_info.iloc[i]['price']))
                    st.write('Rating Tempat: ', str(recommended_places_info.iloc[i]['rating']))

        st.write('----')
        st.write('## Rekomendasi Trip')
        st.write('Rekomendasi perjalanan berdasarkan kemiripan dengan ', place, ' ', 'di ', kota,':')
        # st.write('----')
        trip_list = trip_recom(recommendation_for_package,place,kota)
        trip_package = [recom[2] for recom in trip_list]
        recom_trip = dfp[dfp['package'].isin(trip_package)][['package', 'city', 'place_tourism1', 'place_tourism2', 'place_tourism3','place_tourism4', 'place_tourism5']]
        # Tampilkan dataframe
        st.dataframe(recom_trip,hide_index=True)

        # Mendefinisikan URL WhatsApp
        whatsapp_link = "https://api.whatsapp.com/send?phone=6285330656126&text=Halo Travelind! Saya mau pesan trip dengan nomer package ..."
        # st.link_button("Pesan Trip Rekomendasimu!", whatsapp_link)
        st.markdown(
                    f'<div style="display: flex; justify-content: center;">'
                    f'<a href="{whatsapp_link}" target="_blank">'
                    f'<button style="padding: 10px 15px; background-color: #25D366; color: white; border: none; border-radius: 5px;">Pesan Trip Rekomendasimu!</button>'
                    f'</a>'
                    f'</div>',
                    unsafe_allow_html=True)

  else:
      kota = st.selectbox('Pilih Kota Tujuan',list(place_dict.keys()))
      default = """Saya pengen ke tempat hiburan"""
      user_input = st.text_area("Masukkan deskripsi tempat tujuan", default, height=100)
      if st.button("Cari Rekomendasi"):
        st.write('## Rekomendasi Tempat')
        st.write('Rekomendasi berdasarkan kemiripan dengan ', user_input, ' ', 'di ', kota,':')
        # st.write('----')
        recommendation_list = sorting_desc(user_input, kota, dfm)
        recommendation_for_package = [recom[1] for recom in recommendation_list]
        recommendation_names = [recom[1] for recom in recommendation_list[1:]]

        # Mengambil informasi dari df_place berdasarkan nama tempat yang ada dalam recommendation_names
        recommended_places_info = df_place[df_place['place_name'].isin(recommendation_names)][['url', 'place_name', 'city','rating','price','category']]
        row1 = st.columns(3)
        row2 = st.columns(3)

        for i in range(len(recommended_places_info)):
            if i < 3:
                with row1[i]:
                    st.image(recommended_places_info.iloc[i]['url'], width=200)
                    st.write("#### ", recommended_places_info.iloc[i]['place_name'] ,' - ',recommended_places_info.iloc[i]['city'])
                    st.write('Kategori Tempat: ',str(recommended_places_info.iloc[i]['category']))
                    st.write('Biaya Masuk: ',str(recommended_places_info.iloc[i]['price']))
                    st.write('Rating Tempat: ', str(recommended_places_info.iloc[i]['rating']))
                    
            else:
                with row2[i - 3]:
                    st.image( recommended_places_info.iloc[i]['url'], width=200)
                    st.write("#### ", recommended_places_info.iloc[i]['place_name'] ,' - ',recommended_places_info.iloc[i]['city'])
                    st.write('Kategori Tempat: ',str(recommended_places_info.iloc[i]['category']))
                    st.write('Biaya Masuk: ',str(recommended_places_info.iloc[i]['price']))
                    st.write('Rating Tempat: ', str(recommended_places_info.iloc[i]['rating']))
                    
        
        st.write('----')
        st.write('## Rekomendasi Trip')
        st.write('Rekomendasi perjalanan berdasarkan kemiripan dengan ', user_input, ' ', 'di ', kota,':')
        # st.write('----')
        trip_list = recom_trip_desc(recommendation_for_package,kota)
        trip_package = [recom[2] for recom in trip_list]
        recom_trip = dfp[dfp['package'].isin(trip_package)][['package', 'city', 'place_tourism1', 'place_tourism2', 'place_tourism3','place_tourism4', 'place_tourism5']]
        # Tampilkan dataframe
        st.dataframe(recom_trip,hide_index=True)

        # Mendefinisikan URL WhatsApp
        whatsapp_link = "https://api.whatsapp.com/send?phone=6285330656126&text=Halo Travelind! Saya mau pesan trip dengan nomer package ..."
        # st.link_button("Pesan Trip Rekomendasimu!", whatsapp_link)
        st.markdown(
                    f'<div style="display: flex; justify-content: center;">'
                    f'<a href="{whatsapp_link}" target="_blank">'
                    f'<button style="padding: 10px 15px; background-color: #25D366; color: white; border: none; border-radius: 5px;">Pesan Trip Rekomendasimu!</button>'
                    f'</a>'
                    f'</div>',
                    unsafe_allow_html=True)




if __name__ == '__main__':
  run()