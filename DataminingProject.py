import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ast

# --- 1. Load Data (with caching for Streamlit) ---
@st.cache_data
def load_data():
    """Loads the movie dataset from a CSV file."""
    # Replace 'movies.csv' with the path to your dataset file
    try:
        df = pd.read_csv("movies.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'movies.csv' not found. Please ensure the dataset is in the same directory.")
        return None

# --- 2. Process Data ---
def process_data(df):
    """Cleans and prepares the data for the recommendation engine."""
    if df is None:
        return None

    # Drop rows with missing critical information
    df = df.dropna(subset=['title', 'overview', 'genres']).copy()
    
    # Combine relevant text features into a single column for TF-IDF
    # Example: 'Action Adventure Sci-Fi ...' from genres
    if 'genres' in df.columns:
        df['combined_features'] = df['genres'].fillna('')
    
    # You can add more features to the 'combined_features' column here.
    # For instance, combining with 'cast', 'director', or 'overview' (plot summary)
    # df['combined_features'] = df['combined_features'] + " " + df['overview'].fillna('')
    
    # Normalize titles for easier lookup
    df['title_normalized'] = df['title'].str.lower().str.strip()
    
    return df

# --- 3. Build Recommendation Model ---
def build_model(df):
    """Creates a TF-IDF matrix and calculates cosine similarity."""
    if df is None:
        return None, None
    
    # Convert text features into a matrix of TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim, tfidf

# --- 4. Get Recommendations ---
def get_recommendations(title, df, cosine_sim):
    """Retrieves a list of recommended movies based on a given title."""
    # Normalize the input title
    title = title.lower().strip()
    
    # Check if the movie exists in the dataset
    if title not in df['title_normalized'].values:
        return pd.DataFrame(columns=['title'])
    
    # Get the index of the movie that matches the title
    indices = pd.Series(df.index, index=df['title_normalized']).drop_duplicates()
    idx = indices[title]

    # Get the similarity scores for all movies with this one
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies (exclude the first one, which is the movie itself)
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies' titles (and their scores, if you wish)
    recommendations = df['title'].iloc[movie_indices]
    return recommendations

# --- 5. Streamlit User Interface ---
def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Discover your next favorite movie based on genres and plot!")

    # Load data
    df = load_data()
    if df is None:
        return
    
    # Process data
    df_processed = process_data(df)
    if df_processed is None:
        return
    
    # Build the model
    cosine_sim, tfidf_vectorizer = build_model(df_processed)
    if cosine_sim is None:
        return
    
    # Get the list of movie titles for the dropdown
    movie_titles = df_processed['title'].tolist()
    
    # Create a select box for the user
    selected_movie = st.selectbox(
        "Select a movie you like:",
        movie_titles
    )
    
    # When the user clicks the button
    if st.button('Get Recommendations'):
        recommendations = get_recommendations(selected_movie, df_processed, cosine_sim)
        
        if not recommendations.empty:
            st.subheader("Top Recommendations")
            # Display recommendations as a list
            for i, movie in enumerate(recommendations):
                st.write(f"{i+1}. {movie}")
        else:
            st.write("Sorry, no recommendations found. Please try another movie.")

if __name__ == "__main__":
    main()
