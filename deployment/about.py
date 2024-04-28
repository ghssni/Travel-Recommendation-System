import streamlit as st

def run():
    st.write('## :busts_in_silhouette: Our Team:')
    st.write('##### :woman: [Ghassani Nurbaningtyas](https://www.linkedin.com/in/ghtyas/) | [Gituhub](https://github.com/ghssni)')
    st.write('##### :man: [Hasan Abdul Hamid](https://www.linkedin.com/in/hasan-abdul-hamid-572841163/) | [Gituhub](https://github.com/hasanhmd)')
    st.write('##### :man: [Dita Injarwanto](https://www.linkedin.com/in/dita-injarwanto-189230253/) | [Gituhub](https://github.com/injarw)')
    st.markdown('---')
    st.write('## :desktop_computer: Background')
    st.markdown('''
                Indonesia has a booming tourism industry, but there's room for growth. 
                We aim to attract more domestic and international visitors by promoting our attractions and offering tailored recommendations and tours. 
                This will boost the local economy.
                ''')
    st.markdown('---')
    st.write('## :thought_balloon: Project Obejctive')
    st.markdown('''
                Develop a recommendation system that can show top 5 best places recommendation.
                This website curates personalized trip recommendations for travelers seeking the best places to visit in Java.
                ''')


if __name__ == '__main__':
    run()