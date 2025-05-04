
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_tourism_data.csv")

@st.cache_data
def create_user_attraction_matrix(df):
    user_attraction_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
    return user_attraction_matrix

def get_top_recommendations(user_id, user_attraction_matrix, top_n=5):
    if user_id not in user_attraction_matrix.index:
        return pd.Series(dtype=float)
    
    user_vector = user_attraction_matrix.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(user_vector, user_attraction_matrix.values)[0]
    
    similar_users = user_attraction_matrix.index[np.argsort(similarities)[::-1][1:11]]
    similar_users_ratings = user_attraction_matrix.loc[similar_users]
    
    weighted_ratings = similar_users_ratings.T.dot(similarities[np.argsort(similarities)[::-1][1:11]])
    normalization = np.array([np.abs(similarities[np.argsort(similarities)[::-1][1:11]]).sum()])
    
    scores = weighted_ratings / normalization
    already_rated = user_attraction_matrix.loc[user_id][user_attraction_matrix.loc[user_id] > 0].index
    scores = scores.drop(index=already_rated, errors='ignore')
    
    return scores.sort_values(ascending=False).head(top_n)

# Streamlit UI
st.title("Tourism Attraction Recommendation System")

df = load_data()
user_attraction_matrix = create_user_attraction_matrix(df)

user_ids = df['UserId'].unique()
selected_user = st.selectbox("Select a User ID:", user_ids)

if st.button("Get Recommendations"):
    recommendations = get_top_recommendations(selected_user, user_attraction_matrix)
    if recommendations.empty:
        st.warning("No recommendations found for this user.")
    else:
        st.success("Top Recommended Attraction IDs:")
        st.table(recommendations)
