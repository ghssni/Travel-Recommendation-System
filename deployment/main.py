import streamlit as st
import about
import eda
import prediction
from PIL import Image
from streamlit_option_menu import option_menu

# navigation = st.sidebar.selectbox('Pilih Halaman:',('EDA', 'Predict a Player'))

# if navigation == 'EDA':
#     eda.run()
# else:
#     prediction.run()

st.set_page_config(
    page_title='Travelind Trip Recommendation',
    page_icon=':beach_with_umbrella:',
    layout='centered', #wide
    initial_sidebar_state='expanded')

col1, col2, col3 = st.columns([10, 1, 9])
col1.image('logo.png',width=450)
st.write('# Travelind Trip Recommendation')
st.subheader('Search your recommendation trip and travel with us!')
st.markdown('---')

select = option_menu(None, ["About", "EDA", "Predict"], 
    icons=['house-door-fill', 'bar-chart-fill', 'search-heart-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "icon": {"color": "navy", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"1px", "--hover-color": "#69bff1"}, 
        "nav-link-selected": {"background-color": "#1495e0"},
    }
) 

if select == 'About':
    about.run()
elif select == 'EDA':
    eda.run()
else:
    prediction.run()