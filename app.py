import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Connect to MySQL
db_config = {
    'host': os.getenv("DB_HOST"),
    'port': int(os.getenv("DB_PORT")),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME")
}
conn = mysql.connector.connect(**db_config)

# Load data
query = "SELECT * FROM imdb_2024_combined"
df = pd.read_sql(query, conn)
conn.close()

# Clean the Title column remove index numbers
df['Title'] = df['Title'].str.replace(r'^\d+\.\s*', '', regex=True)


# Clean column names
df.columns = [col.strip().capitalize() for col in df.columns]

# Convert to proper dtypes with fallback fill
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Fill missing values with placeholders (not dropped)
df['Rating_filled'] = df['Rating'].fillna(0)
df['Votes'] = df['Votes'].fillna(0)
df['Duration'] = df['Duration'].fillna(0)

# Create duration column in minutes
df['Duration_min'] = (df['Duration'] * 60).astype(int)
# Create formatted duration string for display
df['Duration'] = df['Duration_min'].astype(str) + " mins"

# Streamlit app
st.set_page_config(page_title="IMDb 2024 Dashboard", layout="wide")
st.title("IMDb 2024 Movie Explorer")


# Sidebar Filters
with st.sidebar:
    st.markdown("### **Filter Movies** ")
    st.markdown("Use the controls below to refine the dataset.")
    st.write(f"Loaded Movies: {len(df)}")

    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 7.0)
    genres = st.multiselect("Select Genres", df['Genre'].dropna().unique())

    duration_range = st.slider("Duration (minutes):",
                               min_value=int(df['Duration_min'].min()),
                               max_value=int(df['Duration_min'].max()),
                               value=(int(df['Duration_min'].min()), int(df['Duration_min'].max())))

    votes_range = st.slider("Voting Counts:",
                            min_value=int(df['Votes'].min()),
                            max_value=int(df['Votes'].max()),
                            value=(int(df['Votes'].min()), int(df['Votes'].max())))

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df['Rating_filled'] >= min_rating) &
    (filtered_df['Duration_min'] >= duration_range[0]) &
    (filtered_df['Duration_min'] <= duration_range[1]) &
    (filtered_df['Votes'] >= votes_range[0]) &
    (filtered_df['Votes'] <= votes_range[1])
]

if genres:
    filtered_df = filtered_df[filtered_df['Genre'].isin(genres)]

# Display filtered dataframe
st.dataframe(filtered_df.drop(columns=['Rating_filled', 'Duration_min']))

# 1. Top 10 Movies by Rating and Voting
st.subheader("Top 10 Movies by Rating and Voting Counts")
top10 = filtered_df.sort_values(by=['Rating_filled', 'Votes'], ascending=False).head(10)
st.table(top10[['Title', 'Genre', 'Rating', 'Votes']])

# 2. Genre Distribution
st.subheader("Genre Distribution")
genre_count = filtered_df['Genre'].value_counts()
if not genre_count.empty:
    fig1, ax1 = plt.subplots()
    genre_count.plot(kind='bar', ax=ax1)
    plt.xlabel("Genre")
    plt.ylabel("Number of Movies")
    st.pyplot(fig1)
else:
    st.warning("No genre data available.")

# 3. Average Duration by Genre
st.subheader("Average Duration by Genre")
if not filtered_df.empty:
    duration_by_genre = filtered_df.groupby('Genre')['Duration_min'].mean().sort_values()
    fig2, ax2 = plt.subplots()
    duration_by_genre.plot(kind='barh', ax=ax2)
    plt.xlabel("Avg Duration (min)")
    st.pyplot(fig2)

# 4. Voting Trends by Genre
st.subheader("Average Voting by Genre")
votes_by_genre = filtered_df.groupby('Genre')['Votes'].mean().sort_values()
fig3, ax3 = plt.subplots()
votes_by_genre.plot(kind='bar', ax=ax3)
plt.ylabel("Avg Votes")
st.pyplot(fig3)

# 5. Rating Distribution
st.subheader("Rating Distribution")
fig4, ax4 = plt.subplots()
sns.histplot(filtered_df['Rating'], bins=20, kde=True, ax=ax4)
plt.xlabel("Rating")
st.pyplot(fig4)

# 6. Genre-Based Rating Leaders
st.subheader("Top Rated Movie per Genre")
top_by_genre = filtered_df.sort_values('Rating_filled', ascending=False).groupby('Genre').first()
st.dataframe(top_by_genre[['Title', 'Rating']])

# 7. Most Popular Genres by Voting
st.subheader("Most Popular Genres by Total Votes")
popular_genres = filtered_df.groupby('Genre')['Votes'].sum()
fig5, ax5 = plt.subplots()
popular_genres.plot(kind='pie', autopct='%1.1f%%', ax=ax5)
plt.ylabel("")
plt.title("Share of Votes per Genre")
st.pyplot(fig5)

# 8. Duration Extremes
st.subheader("Duration Extremes")
shortest_idx = filtered_df['Duration_min'].idxmin()
longest_idx = filtered_df['Duration_min'].idxmax()
shortest = filtered_df.loc[[shortest_idx]]
longest = filtered_df.loc[[longest_idx]]
st.markdown("**Shortest Movie**")
st.table(shortest[['Title', 'Genre', 'Duration', 'Rating', 'Votes']])
st.markdown("**Longest Movie**")
st.table(longest[['Title','Genre', 'Duration', 'Rating', 'Votes']])

# 9. Ratings by Genre (Heatmap)
st.subheader("Ratings Heatmap by Genre")
heatmap_data = filtered_df.pivot_table(index='Genre', values='Rating_filled', aggfunc='mean')
fig6, ax6 = plt.subplots()
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", ax=ax6)
plt.title("Average Rating by Genre")
st.pyplot(fig6)

# 10. Correlation Analysis
st.subheader("Correlation: Rating vs Voting Count")
fig7, ax7 = plt.subplots()
sns.scatterplot(data=filtered_df, x='Votes', y='Rating', ax=ax7)
plt.xlabel("Votes")
plt.ylabel("Rating")
st.pyplot(fig7)

st.success("All visualizations loaded successfully")