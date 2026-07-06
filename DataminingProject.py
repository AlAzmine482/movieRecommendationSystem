import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ast

# --- 1. Load Data (with caching for Streamlit) ---
@st.cache_data
def load_data():
    """Loads the movie dataset from a CSV file."""
    try:
        # Load only the movies file
        df = pd.read_csv("tmdb_5000_movies.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'tmdb_5000_movies.csv' not found. Please ensure the dataset is in the same directory.")
        return None

# --- 2. Helper Functions for Parsing JSON Columns ---
def parse_genres(genres_str):
    """Extract genre names from JSON string."""
    try:
        if isinstance(genres_str, str):
            genres_list = ast.literal_eval(genres_str)
            return [genre['name'] for genre in genres_list]
        return []
    except:
        return []

def parse_keywords(keywords_str):
    """Extract keyword names from JSON string."""
    try:
        if isinstance(keywords_str, str):
            keywords_list = ast.literal_eval(keywords_str)
            return [keyword['name'] for keyword in keywords_list[:5]]
        return []
    except:
        return []

# --- 3. Process Data ---
def process_data(df):
    """Cleans and prepares the data for the recommendation engine."""
    if df is None:
        return None

    # Drop rows with missing critical information
    df = df.dropna(subset=['title', 'overview', 'genres']).copy()
    
    # Parse JSON columns into readable formats
    df['genres_list'] = df['genres'].apply(parse_genres)
    df['keywords_list'] = df['keywords'].apply(parse_keywords)
    
    # Combine features for recommendation
    # Genre names + plot overview + keywords
    df['combined_features'] = df.apply(
        lambda row: ' '.join(row['genres_list']) + ' ' + 
                   str(row['overview']) + ' ' +
                   ' '.join(row['keywords_list']), 
        axis=1
    )
    
    # Normalize titles for easier lookup
    df['title_normalized'] = df['title'].str.lower().str.strip()
    
    # Create a display-friendly genres string
    df['genres_display'] = df['genres_list'].apply(lambda x: ', '.join(x[:3]))
    
    return df

# --- 4. Build Recommendation Model ---
def build_model(df):
    """Creates a TF-IDF matrix and calculates cosine similarity."""
    if df is None:
        return None, None
    
    # Convert text features into a matrix of TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim, tfidf

# --- 5. Get Recommendations with Details ---
def get_recommendations(title, df, cosine_sim):
    """Retrieves a list of recommended movies with details based on a given title."""
    # Normalize the input title
    title = title.lower().strip()
    
    # Check if the movie exists in the dataset
    if title not in df['title_normalized'].values:
        return pd.DataFrame()
    
    # Get the index of the movie that matches the title
    indices = pd.Series(df.index, index=df['title_normalized']).drop_duplicates()
    idx = indices[title]

    # Get the similarity scores for all movies with this one
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies (exclude the first one)
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return detailed information about recommended movies
    recommendations = df[['title', 'genres_display', 'genres_list', 'vote_average', 'overview', 'release_date']].iloc[movie_indices].copy()
    recommendations['similarity_score'] = [score for _, score in sim_scores]
    
    return recommendations

# --- 6. Display Movie Card ---
def display_movie_card(movie, index):
    """Display a single movie with details in a card format."""
    genres = movie['genres_display'] if movie['genres_display'] else 'N/A'
    release_year = movie['release_date'][:4] if pd.notna(movie.get('release_date')) else 'N/A'
    
    # Create a styled card
    st.markdown(f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <h4 style="margin-top: 0; color: #1e3c72;">{index}. {movie['title']}</h4>
            <span style="background-color: #6c757d; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; white-space: nowrap;">
                📅 {release_year}
            </span>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
            <span style="background-color: #1e3c72; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px;">
                ⭐ {movie['vote_average']:.1f}/10
            </span>
            <span style="background-color: #28a745; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px;">
                🎯 {movie['similarity_score']:.2%} match
            </span>
        </div>
        <p style="margin: 5px 0; font-size: 14px;">
            <strong>Genres:</strong> {genres}
        </p>
        <p style="margin: 8px 0 0 0; font-size: 13px; color: #666;">
            <em>{movie['overview'][:250]}...</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 7. Streamlit User Interface ---
def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stSelectbox label {
        font-weight: 600;
        font-size: 16px;
    }
    .stButton button {
        width: 100%;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("Discover your next favorite movie based on genres, plot, and keywords!")
    st.markdown("---")

    # Load data
    with st.spinner("Loading movie database..."):
        df = load_data()
    if df is None:
        return
    
    # Process data
    with st.spinner("Processing movie data..."):
        df_processed = process_data(df)
    if df_processed is None:
        return
    
    # Build the model
    with st.spinner("Building recommendation model..."):
        cosine_sim, tfidf_vectorizer = build_model(df_processed)
    if cosine_sim is None:
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get the list of movie titles for the dropdown
        movie_titles = sorted(df_processed['title'].tolist())
        
        # Create a select box for the user with search functionality
        selected_movie = st.selectbox(
            "🎥 Select a movie you like:",
            movie_titles,
            help="Start typing to search for a movie"
        )
        
        # Show selected movie details
        movie_data = df_processed[df_processed['title'] == selected_movie].iloc[0]
        release_year = movie_data['release_date'][:4] if pd.notna(movie_data.get('release_date')) else 'N/A'
        genres = movie_data['genres_display'] if movie_data['genres_display'] else 'N/A'
        
        st.info(f"""
        **Selected Movie:** {selected_movie} 📅 {release_year}  
        **Genres:** {genres}  
        **Rating:** ⭐ {movie_data['vote_average']:.1f}/10
        """)
        
        # When the user clicks the button
        if st.button(' Get Recommendations', type='primary', use_container_width=True):
            with st.spinner("Finding similar movies..."):
                recommendations = get_recommendations(selected_movie, df_processed, cosine_sim)
            
            if not recommendations.empty:
                st.subheader("📽️ Top 10 Recommendations")
                st.markdown("---")
                
                # Display each recommendation
                for i, (idx, movie) in enumerate(recommendations.iterrows(), 1):
                    display_movie_card(movie, i)
            else:
                st.warning("No recommendations found. Please try another movie.")
    
    with col2:
        st.subheader("📊 About This System")
        st.markdown("""
        **How it works:**
        - Uses **TF-IDF** to analyze movie features
        - Calculates **cosine similarity** between movies
        - Considers:
          - 📝 Plot (overview)
          - 🎭 Genres
          - 🔑 Keywords
          - 📅 Release year
        
        **Dataset:** TMDB 5000 Movies
        - 5,000+ movies
        - Rich metadata
        - Ratings & popularity scores
        """)
        
        st.subheader(" Recommendation Score")
        st.markdown("""
        The similarity score shows how closely each recommended movie matches your selection:
        - 🟢 **> 80%**: Very similar
        - 🟡 **50-80%**: Moderately similar
        - 🟠 **< 50%**: Somewhat similar
        """)
        
        # Display random movie stats
        st.subheader(" Quick Stats")
        total_movies = len(df_processed)
        avg_rating = df_processed['vote_average'].mean()
        top_genre = df_processed['genres_list'].explode().value_counts().index[0] if not df_processed.empty else 'N/A'
        
        st.metric("Total Movies", total_movies)
        st.metric("Average Rating", f"{avg_rating:.1f}/10")
        st.metric("Most Common Genre", top_genre)

if __name__ == "__main__":
    main()
