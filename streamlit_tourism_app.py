
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_tourism_data.csv")

df = load_data()

st.title("Tourism Visit Mode Prediction and Recommendation App")

# Sidebar Inputs
st.sidebar.header("User Input")
user_id = st.sidebar.selectbox("Select User ID", sorted(df['UserId'].unique()))
visit_year = st.sidebar.selectbox("Visit Year", sorted(df['VisitYear'].unique()))
visit_month = st.sidebar.selectbox("Visit Month", sorted(df['VisitMonth'].unique()))
rating = st.sidebar.slider("Rating", 1, 5, 3)
attraction_type = st.sidebar.selectbox("Attraction Type", sorted(df['AttractionTypeId'].unique()))
continent = st.sidebar.selectbox("Continent ID", sorted(df['ContenentId'].unique()))
region = st.sidebar.selectbox("Region ID", sorted(df['RegionId'].unique()))
city = st.sidebar.selectbox("City ID", sorted(df['CityId'].unique()))
country = st.sidebar.selectbox("Country ID", sorted(df['CountryId'].unique()))

# ---- Visit Mode Prediction ----
st.subheader("1. Predict Visit Mode")
features = ['VisitYear', 'VisitMonth', 'Rating', 'AttractionTypeId', 'ContenentId', 'RegionId', 'CityId', 'CountryId']
X = df[features]
y = df['VisitModeId']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

input_features = [[visit_year, visit_month, rating, attraction_type, continent, region, city, country]]
predicted_mode = clf.predict(input_features)[0]
mode_labels = {1: "Business", 2: "Family", 3: "Couple", 4: "Friends", 5: "Solo"}
st.success(f"Predicted Visit Mode: {mode_labels.get(predicted_mode, 'Unknown')}")

# ---- Recommendation Section ----
st.subheader("2. Personalized Attraction Recommendations")
user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating')
if user_id in user_item_matrix.index:
    target_user_ratings = user_item_matrix.loc[user_id]
    other_users = user_item_matrix.drop(index=user_id)
    sim_scores = other_users.apply(lambda x: x.corr(target_user_ratings), axis=1).dropna().sort_values(ascending=False)
    top_sim_users = sim_scores.head(5).index

    recommendations = user_item_matrix.loc[top_sim_users].mean().sort_values(ascending=False)
    already_rated = target_user_ratings.dropna().index
    recommendations = recommendations.drop(index=already_rated, errors='ignore')

    st.write("Top 5 Recommended Attractions:")
    st.write(recommendations.head(5))
else:
    st.warning("User ID not found in the dataset.")

# ---- Visualizations ----
st.subheader("3. Tourism Data Insights")

st.markdown("**Top 10 Most Visited Attractions**")
top_attractions = df['AttractionId'].value_counts().head(10)
fig1, ax1 = plt.subplots()
sns.barplot(x=top_attractions.index.astype(str), y=top_attractions.values, ax=ax1)
ax1.set_xlabel("Attraction ID")
ax1.set_ylabel("Visit Count")
ax1.set_title("Top Attractions")
st.pyplot(fig1)

st.markdown("**Top 10 Regions by Visits**")
top_regions = df['RegionId'].value_counts().head(10)
fig2, ax2 = plt.subplots()
sns.barplot(x=top_regions.index.astype(str), y=top_regions.values, ax=ax2)
ax2.set_xlabel("Region ID")
ax2.set_ylabel("Visit Count")
ax2.set_title("Top Regions")
st.pyplot(fig2)

st.markdown("**User Segments by Visit Mode**")
fig3, ax3 = plt.subplots()
sns.countplot(x='VisitModeId', data=df, order=sorted(df['VisitModeId'].unique()), ax=ax3)
ax3.set_xlabel("Visit Mode ID")
ax3.set_ylabel("User Count")
ax3.set_title("Visit Mode Distribution")
st.pyplot(fig3)
