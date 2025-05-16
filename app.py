import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(layout="wide", page_title="Restaurant Recommender")

import pandas as pd
import joblib
import os

class RestaurantRecommender:
    def __init__(self, data_path):
        self.df = self.load_data(data_path)
    
    def load_data(self, path):
        try:
            # Try multiple encodings
            try:
                df = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin1')
            
            # Data cleaning
            df['Cuisines'] = df['Cuisines'].fillna('Unknown').astype(str).str.lower()
            df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce').fillna(0)
            return df
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return pd.DataFrame()

    def recommend(self, preferences):
        results = self.df.copy()
        
        if preferences.get('cuisines'):
            pattern = '|'.join([c.lower() for c in preferences['cuisines']])
            results = results[results['Cuisines'].str.contains(pattern, case=False, na=False)]
        
        if preferences.get('min_rating'):
            results = results[results['Aggregate rating'] >= preferences['min_rating']]
        
        return results.sort_values(['Aggregate rating', 'Votes'], ascending=[False, False]).head(10)

# Configure paths
BASE_PATH = r"C:\Users\user\Desktop\Project(ip)recommendation system for restaurant"
DATA_PATH = os.path.join(BASE_PATH, "cleaned_zomato.csv")
MODEL_PATH = os.path.join(BASE_PATH, "restaurant_recommender_model.joblib")

@st.cache_resource
def initialize_system():
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"Data file not found at: {DATA_PATH}")
            return None, None
            
        if not os.path.exists(MODEL_PATH):
            st.info("Creating new model...")
            recommender = RestaurantRecommender(DATA_PATH)
            joblib.dump(recommender, MODEL_PATH)
            return recommender.df, recommender
        else:
            recommender = joblib.load(MODEL_PATH)
            return recommender.df, recommender
            
    except Exception as e:
        st.error(f"System error: {str(e)}")
        return None, None

# Initialize system
df, recommender = initialize_system()

# Check initialization
if df is None or recommender is None or df.empty:
    st.error("System initialization failed. Please check:")
    st.write(f"1. Data file exists at: {DATA_PATH}")
    st.write("2. You have proper file permissions")
    st.stop()

st.title('🍽 Restaurant Recommender')

# Get unique cuisines
try:
    all_cuisines = sorted({cuisine.strip() for cuisines in df['Cuisines'].str.split(',') 
                         for cuisine in cuisines if pd.notna(cuisine) and cuisine.strip()})
    if not all_cuisines:
        raise ValueError("No cuisines found")
except Exception as e:
    st.warning(f"Using default cuisines: {str(e)}")
    all_cuisines = ['Italian', 'Indian', 'Chinese', 'American']

# Filters
with st.sidebar:
    st.header('Search Filters')
    selected_cuisines = st.multiselect('Select Cuisines', all_cuisines)
    min_rating = st.slider('Minimum Rating', 0.0, 5.0, 3.0, 0.1)

# Recommendations
if st.button('Find Restaurants'):
    if not selected_cuisines:
        st.warning("Please select at least one cuisine")
    else:
        try:
            recommendations = recommender.recommend({
                'cuisines': selected_cuisines,
                'min_rating': min_rating
            })
            
            if recommendations.empty:
                st.info("No restaurants match your criteria")
            else:
                st.success(f"Found {len(recommendations)} restaurants")
                
                for _, row in recommendations.iterrows():
                    with st.expander(f"{row['Restaurant Name']} - ⭐{row['Aggregate rating']:.1f}"):
                        cols = st.columns([2, 1])
                        with cols[0]:
                            st.write(f"**Cuisines:** {row['Cuisines']}")
                            st.write(f"**Location:** {row.get('Locality', 'N/A')}")
                            st.write(f"**Price:** {row.get('Average Cost for two', 'N/A')} {row.get('Currency', '')}")
                            st.write(f"**Votes:** {row.get('Votes', 0)}")
                        
                        with cols[1]:
                            if all(col in row for col in ['Latitude', 'Longitude']):
                                st.map(pd.DataFrame({
                                    'lat': [row['Latitude']],
                                    'lon': [row['Longitude']]
                                }))
        except Exception as e:
            st.error(f"Recommendation error: {str(e)}")