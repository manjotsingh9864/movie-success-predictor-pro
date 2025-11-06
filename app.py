import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cinematic theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 50%, #f0f4ff 100%);
        color: #1a1a2e;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #1a1a2e;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: none;
        -webkit-background-clip: unset;
        -webkit-text-fill-color: unset;
    }
    
    .sub-header {
        text-align: center;
        color: #4a4a6a;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #6c757d;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(173, 216, 230, 0.6);
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(38, 115, 252, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(38, 115, 252, 0.6);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4ff 100%);
        color: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #1a1a2e !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("movie_hit_flop_model.pkl")
    except:
        st.error("⚠️ Model file not found! Please run train_model.py first.")
        return None

model = load_model()

# Header
st.markdown('<h1 class="main-header">🎬 Movie Success Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Machine Learning | Predict Box Office Performance</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 About")
    st.markdown("""
    This AI tool predicts whether a movie will be a **Hit** or **Flop** based on:
    
    - 📅 Release timing
    - 🎭 Genre & cast
    - 💰 Budget & marketing
    - ⭐ Critic & audience scores
    - 🎞️ Production details
    
    **How to use:**
    1. Fill in movie details
    2. Click "Predict Success"
    3. Get instant predictions
    """)
    
    st.markdown("---")
    st.info("💡 Use realistic values based on similar movies.")

if model is None:
    st.stop()

# Input layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-header">📅 Release Information</div>', unsafe_allow_html=True)
    year = st.number_input("Release Year", min_value=1980, max_value=2030, value=2025)
    release_month = st.slider("Release Month", 1, 12, 6, help="1=Jan, 12=Dec")
    runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=240, value=140)
    
    st.markdown('<div class="section-header">🎭 Content Details</div>', unsafe_allow_html=True)
    genre = st.selectbox("Genre", 
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        format_func=lambda x: ["Action", "Comedy", "Drama", "Thriller", "Romance", 
                               "Horror", "Sci-Fi", "Animation", "Documentary", "Fantasy", "Musical"][x],
        index=2)
    sequel = st.selectbox("Sequel?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    franchise = st.selectbox("Part of Franchise?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    st.markdown('<div class="section-header">💰 Financial Details</div>', unsafe_allow_html=True)
    budget = st.number_input("Production Budget (₹ Crores)", min_value=0, max_value=1000, value=80)
    marketing = st.number_input("Marketing Budget (₹ Crores)", min_value=0, max_value=500, value=30)
    
    st.markdown('<div class="section-header">🎬 Production</div>', unsafe_allow_html=True)
    production_size = st.selectbox("Production Size", 
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: ["Independent", "Small", "Medium", "Large", "Major Studio", "Mega Production"][x],
        index=1)
    screen_count = st.number_input("Theater Screens", min_value=0, max_value=10000, value=2500)

with col3:
    st.markdown('<div class="section-header">⭐ Ratings & Popularity</div>', unsafe_allow_html=True)
    director_popularity = st.slider("Director Popularity", 0.0, 10.0, 8.5, 0.1)
    cast_popularity = st.slider("Cast Popularity", 0.0, 10.0, 9.1, 0.1)
    critic_score = st.slider("Expected Critic Score", 0.0, 10.0, 7.9, 0.1)
    audience_score = st.slider("Expected Audience Score", 0.0, 10.0, 8.3, 0.1)
    
    st.markdown('<div class="section-header">🌍 Distribution</div>', unsafe_allow_html=True)
    director = st.number_input("Director Code", min_value=0, max_value=10, value=5)
    language = st.number_input("Language Code", min_value=0, max_value=10, value=3)
    country = st.number_input("Country Code", min_value=0, max_value=10, value=1)

# Collect inputs (REMOVED opening_weekend and total_gross)
movie_data = {
    "year": year,
    "genre": genre,
    "director": director,
    "production_size": production_size,
    "language": language,
    "country": country,
    "budget": budget,
    "marketing": marketing,
    "runtime": runtime,
    "release_month": release_month,
    "sequel": sequel,
    "franchise": franchise,
    "director_popularity": director_popularity,
    "cast_popularity": cast_popularity,
    "critic_score": critic_score,
    "audience_score": audience_score,
    "screen_count": screen_count
}

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🎯 Predict Movie Success", use_container_width=True)

if predict_button:
    with st.spinner("🎬 Analyzing movie data..."):
        df = pd.DataFrame([movie_data])
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        confidence = prob[prediction] * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results
        if prediction == 1:
            st.success(f"### 🎉 BLOCKBUSTER ALERT!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", "HIT ✨", delta="Success")
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%", delta="High" if confidence > 75 else "Moderate")
            with col3:
                estimated_gross = budget * 3.5
                roi = ((estimated_gross - budget - marketing) / (budget + marketing)) * 100
                st.metric("Potential ROI", f"{roi:.1f}%", delta="High Return")
            
            st.balloons()
            st.info("💡 **Recommendation:** Strong box office potential. Consider wider release and aggressive marketing.")
        else:
            st.error(f"### ⚠️ HIGH RISK DETECTED")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", "FLOP 📉", delta="Caution")
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%", delta="Warning")
            with col3:
                estimated_gross = budget * 0.6
                roi = ((estimated_gross - budget - marketing) / (budget + marketing)) * 100
                st.metric("Potential ROI", f"{roi:.1f}%", delta="High Risk")
            
            st.warning("💡 **Recommendation:** Consider revising strategy. Limited release may minimize losses.")
        
        # Details
        with st.expander("📊 View Detailed Analysis"):
            estimated_gross = budget * (3.5 if prediction == 1 else 0.6)
            st.markdown(f"""
            **Financial Breakdown:**
            - Production Budget: ₹{budget} Crores
            - Marketing Spend: ₹{marketing} Crores
            - Total Investment: ₹{budget + marketing} Crores
            - Estimated Gross: ₹{estimated_gross:.1f} Crores
            - Net Profit/Loss: ₹{estimated_gross - budget - marketing:.1f} Crores
            
            **Performance Indicators:**
            - Director Rating: {director_popularity}/10
            - Cast Appeal: {cast_popularity}/10
            - Critical Reception: {critic_score}/10
            - Audience Appeal: {audience_score}/10
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4a4a6a; padding: 2rem;'>
    <p>🎬 Powered by Advanced Machine Learning</p>
    <p style='font-size: 0.9rem;'>Predictions based on historical data. Actual results may vary.</p>
</div>
""", unsafe_allow_html=True)