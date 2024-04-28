# Travel-Recommendation-System
This project aims to enhance Indonesia's tourism industry by promoting tourist destinations and attracting domestic and international visitors. Users can explore recommendations for attractions based on their interests and choose pre-designed tour packages. The content-based filtering recommendation method will be employed, developing two recommendation models: one based on tourist destination description similarity and the other on user-input destination descriptions. The model successfully displays recommendations with over 50% similarity, it will significantly advance personalized tourism recommendations.

## Project Overview
**Goal:** Develop a personalized travel recommendation system to suggest destinations and tour packages to users.

**Benefits:** 
- Boost User Satisfaction: Find perfect destinations based on user interests.
- Drive Local Economy: Encourage travel and support local businesses.

## Tools and Technologies
- Programming Language: Python
- Data Analysis Libraries: Pandas, Matplotlib, Seaborn
- Machine Learning Libraries: Scikit-Learn and Tensorflow
- User Interface: Streamlit & Huggingface

## Conclusion
The recommender system that has been created can be useful for increasing customer purchasing power. Because customers can see the similarity of tourist attractions they want to go to and have a choice of trips that suit their desires. The model successfully displays recommendations with over 50% similarity, it will significantly advance personalized tourism recommendations.

`Conclusion of EDA`:
1. There are 300 user data that provide ratings in the database.
2. The distribution of user rating data from 1-5 tends to be balanced for each rating.
3. There is no correlation between user rating and price variables.
4. There are 5 cities representing each province in the data with `Yogyakarta` being the city with the most tourist attractions in the data at 437.
5. There are 6 categories of tourist attractions in the data with `Amusement Park` being the largest category at 135 places.
6. Based on the number of ratings obtained Amusement Park is the most visited place with a total of 3024 visits.
7. Based on the average rating obtained Amusement park is the most visited place with a total rating of 3,118.
8. The number of visits is not directly proportional to the average rating obtained. This is shown by the category of place of worship that occupies the 2nd favorite place with a rating of 3.09 which is inversely proportional to the number of ratings obtained at 382 or the 2nd lowest in the data.

## Suggestion
Improvements that can be made in the future:
- Storing customer travel package purchase history. So that a recommender system can be made with the collaborative filtering method.
- Adding the diversity of tourist attractions and travel packages so that customers have more choices.

## Acknowledgements
- Indonesia Toursim Destination data used in this project was obtainded from  [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)
- Dashboard visualization for this project on [Looker Studio](https://lookerstudio.google.com/u/0/reporting/966225a4-c765-4c38-a8d7-ec7112dca42f/page/6YQvD)
- Model deployment for this project on [Hugging Face](https://huggingface.co/spaces/ghtyas/Travelind)
