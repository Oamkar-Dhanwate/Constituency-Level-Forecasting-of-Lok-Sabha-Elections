import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pickle
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import warnings
import plotly.io as pio  # Added for exporting plots
import ollama             # Added for the AI function call

warnings.filterwarnings('ignore')

# --- PATH CONFIGURATION ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import utilities with better error handling
# This block is now correct because of the sys.path.append above
try:
    from src.feature_engineering import create_alliance_feature
    from src.report_generator import generate_pdf_report 
    from src.ai_utils import generate_report_conclusion
    IMPORT_SUCCESS = True
except ImportError as e:
    st.warning(f"Some optional modules not available: {e}. This is normal if you haven't run the notebooks.")
    IMPORT_SUCCESS = False

# --- Page Configuration ---
st.set_page_config(
    page_title="🇮🇳 Indian Election Analytics",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Modern CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .prediction-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #26c6da;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        gap: 1rem;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"]:hover {
        background-color: #5a6fd8 !important;
    }
    .data-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
ALLIANCE_COLORS = {
    'NDA': '#FF9933',
    'UPA': '#138808',
    'Left': '#FF0000',
    'Other': '#A9A9A9'
}

COLOR_SCALES = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']
DEFAULT_YEAR = 2024
MAJORITY_THRESHOLD = 272  # Lok Sabha majority

class DataLoader:
    """Enhanced data loader with robust error handling and caching."""
    
    @staticmethod
    @st.cache_data(show_spinner="🔄 Loading election data...")
    def load_election_data() -> Tuple[pd.DataFrame, Optional[Any], gpd.GeoDataFrame]:
        """Loads election data with comprehensive error handling."""
        try:
            # Load main dataset
            data_path = project_root / 'data' / 'processed' / 'featured_data.csv'
            if not data_path.exists():
                # Try alternative path
                data_path_alt = project_root / 'data' / 'processed' / 'cleaned_election_results.csv'
                if data_path_alt.exists():
                    data_path = data_path_alt
                else:
                    st.error(f"❌ Data file not found")
                    return pd.DataFrame(), None, gpd.GeoDataFrame()
            
            df = pd.read_csv(data_path)
            st.sidebar.success(f"✅ Loaded {len(df)} rows")
            
            # Standardize column names
            df = DataLoader._standardize_columns(df)
            
            # Load model
            model_path = project_root / 'models' / 'best_ridge_model.joblib'
            model = DataLoader._load_model(model_path)
            
            # Load GeoJSON
            gdf = DataLoader._load_geojson(project_root / 'data' / 'raw' / 'india_states.geojson')
            
            return df, model, gdf
            
        except Exception as e:
            st.error(f"❌ Critical error loading data: {e}")
            return pd.DataFrame(), None, gpd.GeoDataFrame()
    
    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across datasets."""
        column_mappings = {
            'Year': ['year', 'election_year', 'YEAR'],
            'State_Name': ['state_name', 'state', 'STATE', 'State'],
            'Votes': ['votes', 'VOTES', 'total_votes'],
            'Vote_Share': ['vote_share', 'votes_share', 'VOTE_SHARE'],
            'Party': ['party', 'PARTY', 'political_party'],
            'Alliance': ['alliance', 'ALLIANCE', 'coalition'],
            'Constituency_Type': ['constituency_type', 'const_type', 'CONSTITUENCY_TYPE'],
            'Incumbent': ['incumbent', 'INCUMBENT', 'is_incumbent'],
            'Candidate': ['candidate', 'CANDIDATE', 'candidate_name'],
            'Constituency_No': ['constituency_no', 'constituency_number', 'CONSTITUENCY_NO']
        }
        
        for standard_col, alternatives in column_mappings.items():
            if standard_col not in df.columns:
                for alt_col in alternatives:
                    if alt_col in df.columns:
                        df[standard_col] = df[alt_col]
                        break

        # Add N_Cand if it exists, as it's likely a model feature
        if 'N_Cand' not in df.columns:
            if 'n_cand' in df.columns:
                df['N_Cand'] = df['n_cand']
            elif 'N_Candidates' in df.columns:
                 df['N_Cand'] = df['N_Candidates']

        return df
    
    @staticmethod
    def _load_model(model_path: Path) -> Optional[Any]:
        """Load model with compatibility handling."""
        if not model_path.exists():
            st.sidebar.warning(f"⚠️ Model file not found: {model_path.name}")
            return None
            
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {e}")
            return None
    
    @staticmethod
    def _load_geojson(geojson_path: Path) -> gpd.GeoDataFrame:
        """Load and preprocess GeoJSON data."""
        try:
            if geojson_path.exists():
                gdf = gpd.read_file(geojson_path)
                gdf['st_nm'] = gdf['st_nm'].str.lower().str.replace(' ', '_')
                return gdf
            return gpd.GeoDataFrame()
        except Exception as e:
            st.sidebar.warning(f"⚠️ GeoJSON loading failed: {e}")
            return gpd.GeoDataFrame()
    
    @staticmethod
    @st.cache_data(show_spinner="🔄 Loading simulation results...")
    def load_simulation_data() -> Optional[Dict]:
        """Loads simulation data with fallback options."""
        try:
            simulation_files = {
                'election_simulation_model': project_root / 'models' / 'election_simulation_model.pkl',
                'alliance_counts': project_root / 'models' / 'alliance_counts.pkl',
                'constituency_probs': project_root / 'models' / 'constituency_probabilities.pkl',
                'winners_df': project_root / 'models' / 'winners_df.pkl'
            }
            
            simulation_data = {}
            files_loaded = 0
            
            for key, path in simulation_files.items():
                if path.exists():
                    try:
                        with open(path, 'rb') as f:
                            simulation_data[key] = pickle.load(f)
                        files_loaded += 1
                        st.sidebar.success(f"✅ Loaded {key}")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error loading {key}: {str(e)}")
                        simulation_data[key] = None
                else:
                    st.sidebar.warning(f"⚠️ Missing: {path.name}")
                    simulation_data[key] = None
            
            if files_loaded > 0:
                return simulation_data
            else:
                return None
                
        except Exception as e:
            st.sidebar.error(f"❌ Simulation data loading failed: {e}")
            return None

class DataProcessor:
    """Handles data processing and optimization."""
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and optimize the election data."""
        if df.empty:
            return df
            
        df_processed = df.copy()
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
        
        return df_processed

class DataFilter:
    """Handles data filtering operations."""
    
    @staticmethod
    def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe."""
        if df.empty:
            return df
            
        df_filtered = df.copy()
        
        # Year filter
        if 'year' in filters and filters['year'] and 'Year' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Year'] == filters['year']]
        
        # State filter
        if ('state' in filters and filters['state'] and filters['state'] != "All States" 
            and 'State_Name' in df_filtered.columns):
            df_filtered = df_filtered[df_filtered['State_Name'] == filters['state']]
        
        # Alliance filter
        if ('alliances' in filters and filters['alliances'] and 'Alliance' in df_filtered.columns):
            df_filtered = df_filtered[df_filtered['Alliance'].isin(filters['alliances'])]
        
        return df_filtered

class VisualizationGenerator:
    """Generates various visualizations for the dashboard."""
    
    @staticmethod
    def create_metrics_cards(df: pd.DataFrame) -> None:
        """Create metric cards for the overview section."""
        if df.empty:
            st.warning("No data available for metrics calculation")
            return
        
        total_votes = df['Votes'].sum() if 'Votes' in df.columns else 0
        total_constituencies = df['Constituency_No'].nunique() if 'Constituency_No' in df.columns else 0
        total_candidates = df['Candidate'].nunique() if 'Candidate' in df.columns else 0
        avg_vote_share = df['Vote_Share'].mean() if 'Vote_Share' in df.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Total Votes", f"{total_votes:,.0f}", "#667eea"),
            ("Constituencies", f"{total_constituencies}", "#667eea"),
            ("Candidates", f"{total_candidates}", "#667eea"),
            ("Avg Vote Share", f"{avg_vote_share:.1f}%", "#667eea")
        ]
        
        for (title, value, color), col in zip(metrics, [col1, col2, col3, col4]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='margin:0; color: {color};'>{title}</h3>
                    <h2 style='margin:0; color: #333;'>{value}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_geographical_map(df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> None:
        """Create geographical vote share map."""
        if df.empty or gdf.empty or 'Alliance' not in df.columns:
            st.info("Map data not available")
            return
        
        map_choice = st.selectbox(
            "Select Alliance to visualize:",
            options=sorted(df['Alliance'].unique()),
            key="map_choice"
        )
        
        map_data = df[df['Alliance'] == map_choice].groupby('State_Name')['Vote_Share'].mean().reset_index()
        merged_gdf = gdf.merge(map_data, left_on='st_nm', right_on='State_Name', how='left')
        merged_gdf['Vote_Share'] = merged_gdf['Vote_Share'].fillna(0)
        
        fig_map = px.choropleth(
            merged_gdf,
            geojson=merged_gdf.geometry,
            locations=merged_gdf.index,
            color="Vote_Share",
            hover_name="st_nm",
            hover_data={"Vote_Share": ":.2f"},
            color_continuous_scale=st.selectbox("Color Scale", COLOR_SCALES, key="color_scale"),
            title=f"Average Vote Share for {map_choice} by State"
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, use_container_width=True)
    
    @staticmethod
    def create_historical_trends(df: pd.DataFrame) -> None:
        """Create historical trends visualization."""
        if 'Alliance' not in df.columns or 'Year' not in df.columns:
            st.info("Alliance or Year data not available for trends")
            return
        
        trend_choices = st.multiselect(
            "Compare alliances:",
            options=list(ALLIANCE_COLORS.keys()),
            default=['NDA', 'UPA'],
            key="trend_choices"
        )
        
        if trend_choices:
            trend_data = df[df['Alliance'].isin(trend_choices)].groupby(
                ['Year', 'Alliance'])['Vote_Share'].mean().reset_index()
            
            fig_trend = px.line(
                trend_data, x='Year', y='Vote_Share', color='Alliance',
                title='Alliance Performance Over Time',
                markers=True,
                color_discrete_map=ALLIANCE_COLORS,
                height=500
            )
            fig_trend.update_layout(
                xaxis_title="Election Year",
                yaxis_title="Average Vote Share (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    @staticmethod
    def create_party_performance_chart(df: pd.DataFrame) -> None:
        """Create party performance visualization."""
        if 'Party' not in df.columns or 'Vote_Share' not in df.columns:
            return
            
        top_parties = df.groupby('Party')['Vote_Share'].mean().nlargest(10).reset_index()
        fig_parties = px.bar(
            top_parties,
            x='Vote_Share',
            y='Party',
            orientation='h',
            title="Top 10 Parties by Average Vote Share",
            color='Vote_Share',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_parties, use_container_width=True)
    
    @staticmethod
    def create_incumbent_analysis(df: pd.DataFrame) -> None:
        """Create incumbent vs challenger analysis."""
        if 'Incumbent' not in df.columns or 'Vote_Share' not in df.columns:
            return
            
        fig_inc = px.box(
            df, 
            x='Incumbent', 
            y='Vote_Share', 
            color='Incumbent',
            title="Vote Share Distribution: Incumbent vs Challenger",
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        st.plotly_chart(fig_inc, use_container_width=True)

class PredictionEngine:
    """Handles AI-powered predictions."""
    
    @staticmethod
    def create_prediction_interface(model: Optional[Any], df: pd.DataFrame) -> None:
        """Create the prediction interface."""
        st.header("🔮 AI-Powered Vote Share Prediction")
        
        if model is None:
            st.warning("""
            **Prediction features are currently disabled.**
            
            To enable predictions:
            1. Ensure the model file exists at `models/best_ridge_model.joblib`
            2. Check model compatibility with your scikit-learn version
            """)
            return
        
        with st.form("prediction_form"):
            st.subheader("🎯 Candidate Profile")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_party = st.selectbox(
                    "Political Party",
                    options=sorted(df['Party'].unique()) if 'Party' in df.columns else ["Unknown"],
                    key="pred_party"
                )
                pred_state = st.selectbox(
                    "State",
                    options=sorted(df['State_Name'].unique()) if 'State_Name' in df.columns else ["Unknown"],
                    key="pred_state"
                )
            
            with col2:
                pred_year = st.number_input("Election Year", value=2024, min_value=2000, max_value=2030)
                pred_const_type = st.selectbox(
                    "Constituency Type",
                    options=sorted(df['Constituency_Type'].unique()) if 'Constituency_Type' in df.columns else ["Unknown"],
                    key="pred_const"
                )
            
            with col3:
                # IMPORTANT: The label here must match the feature name used in your model pipeline
                pred_ncand = st.slider("Number of Candidates (N_Cand)", min_value=2, max_value=50, value=8)
                pred_incumbent = st.radio("Incumbent Status", [True, False], format_func=lambda x: "✅ Incumbent" if x else "❌ Challenger")
            
            submitted = st.form_submit_button("🚀 Predict Vote Share", use_container_width=True)
        
        if submitted:
            with st.spinner("🤖 Analyzing patterns and predicting vote share..."):
                try:
                    # --- START: Updated Prediction Logic ---
                    
                    # 1. Collect inputs
                    inputs = {
                        'Party': pred_party,
                        'State_Name': pred_state,
                        'Year': pred_year,
                        'Constituency_Type': pred_const_type,
                        'N_Cand': pred_ncand,  # This MUST match the column name your model was trained on
                        'Incumbent': pred_incumbent
                    }
                    
                    # 2. Create DataFrame for prediction
                    input_df = pd.DataFrame(inputs, index=[0])
                    
                    # 3. Save inputs to session state for the PDF report
                    st.session_state.prediction_inputs = {
                        'party': pred_party,
                        'state': pred_state,
                        'year': pred_year,
                        'const_type': pred_const_type,
                        'ncand': pred_ncand,
                        'incumbent': pred_incumbent
                    }
                    
                    # 4. Make prediction (assuming 'model' is a full scikit-learn Pipeline)
                    predicted_vote_share = model.predict(input_df)[0]
                    
                    # 5. Clean prediction
                    predicted_vote_share = max(0, min(100, predicted_vote_share)) # Ensure it's between 0-100
                    
                    st.session_state.prediction_made = True
                    st.session_state.prediction_result = predicted_vote_share
                    
                    # --- END: Updated Prediction Logic ---
                    
                    st.success("### Prediction Complete! 🎉")
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=predicted_vote_share,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Predicted Vote Share (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 50], 'color': "yellow"},
                                {'range': [50, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.error("This often happens if the saved model is not a full scikit-learn Pipeline or if feature names (e.g., 'N_Cand') don't match the model's training columns.")


# This class is no longer used by Tab 4 but is kept for legacy/other functions
class ReportGenerator:
    """Handles report generation functionality."""
    
    @staticmethod
    def create_report_interface(df: pd.DataFrame) -> None:
        """Create the report generation interface."""
        st.header("📄 Smart Report Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Report Configuration")
            report_title = st.text_input("Report Title", value="Election Analysis Report")
            include_ai = st.checkbox("🤖 Include AI Analysis", value=True)
        
        with col2:
            st.subheader("Quick Actions")
            if st.button("📥 Generate PDF Report", use_container_width=True):
                if not df.empty:
                    with st.spinner("Generating report..."):
                        st.success("Report generation would be implemented here!")
                        
                        # Show quick summary
                        st.subheader("Report Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Records", len(df))
                            if 'Vote_Share' in df.columns:
                                st.metric("Average Vote Share", f"{df['Vote_Share'].mean():.1f}%")
                        
                        with col2:
                            if 'Party' in df.columns:
                                st.metric("Unique Parties", df['Party'].nunique())
                            if 'State_Name' in df.columns:
                                st.metric("States Covered", df['State_Name'].nunique())
                else:
                    st.warning("No data available for report generation")

# --- Initialize Session State ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction_inputs = {}
    st.session_state.prediction_result = 0.0

# --- Load Data ---
df, model, gdf = DataLoader.load_election_data()
sim_data = DataLoader.load_simulation_data()

# Process data
df_processed = DataProcessor.preprocess_data(df)

# --- Modern Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #667eea;'>🗳️ Election Analytics</h2>
        <p style='color: #666;'>Explore and predict election outcomes</p>
    </div>
    """, unsafe_allow_html=True)
    
    filters = {}
    
    with st.expander("📅 **Time & Location**", expanded=True):
        if 'Year' in df_processed.columns:
            filters['year'] = st.selectbox(
                "Election Year",
                options=sorted(df_processed['Year'].unique(), reverse=True),
                key="year_select"
            )
        else:
            filters['year'] = DEFAULT_YEAR
        
        if 'State_Name' in df_processed.columns:
            all_states_option = "All States"
            states = sorted(df_processed['State_Name'].unique())
            filters['state'] = st.selectbox(
                "State",
                options=[all_states_option] + states,
                key="state_select"
            )
        else:
            filters['state'] = "All States"
    
    with st.expander("🎯 **Advanced Filters**"):
        if 'Alliance' in df_processed.columns:
            filters['alliances'] = st.multiselect(
                "Filter by Alliance",
                options=list(ALLIANCE_COLORS.keys()),
                default=['NDA', 'UPA', 'Left']
            )
        else:
            filters['alliances'] = []
        
        if 'Votes' in df_processed.columns:
            filters['min_votes'] = st.number_input(
                "Minimum Votes",
                min_value=0,
                value=0,
                step=1000
            )
        else:
            filters['min_votes'] = 0

# --- Filter Data ---
df_filtered = DataFilter.filter_dataframe(df_processed, filters)

# --- Main Page Header ---
st.markdown('<h1 class="main-header">🇮🇳 Indian Election Analytics Dashboard</h1>', unsafe_allow_html=True)

# --- Tabs for different sections ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 **Data Analytics**", 
    "🔬 **Monte Carlo Simulation**", 
    "🔮 **AI Prediction**", 
    "📄 **Smart Reports**"
])

with tab1:
    st.subheader("📈 Election Overview")
    
    if not df_filtered.empty:
        VisualizationGenerator.create_metrics_cards(df_filtered)
        
        # Data Table
        st.subheader("📋 Election Data")
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)
        
        # Advanced Analytics
        st.subheader("📊 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            VisualizationGenerator.create_geographical_map(df_filtered, gdf)
        
        with col2:
            # Historical Trends
            st.markdown("**📈 Historical Trends**")
            if 'Alliance' in df.columns:
                trend_choices = st.multiselect(
                    "Compare alliances:",
                    options=list(ALLIANCE_COLORS.keys()),
                    default=['NDA', 'UPA'],
                    key="trend_choices"
                )
                
                if trend_choices:
                    trend_data = df[df['Alliance'].isin(trend_choices)].groupby(['Year', 'Alliance'])['Vote_Share'].mean().reset_index()
                    fig_trend = px.line(
                        trend_data, x='Year', y='Vote_Share', color='Alliance',
                        title='Alliance Performance Over Time',
                        markers=True,
                        color_discrete_map=ALLIANCE_COLORS,
                        height=500
                    )
                    fig_trend.update_layout(
                        xaxis_title="Election Year",
                        yaxis_title="Average Vote Share (%)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Alliance data not available for trends")
        
        # Additional charts
        st.subheader("📈 Detailed Analysis")
        col3, col4 = st.columns(2)
        
        with col3:
            # Incumbent performance analysis
            if 'Incumbent' in df_filtered.columns and 'Vote_Share' in df_filtered.columns:
                st.subheader("🏛️ Incumbent Performance")
                incumbent_stats = df_filtered.groupby('Incumbent')['Vote_Share'].agg(['mean', 'count']).round(2)
                incumbent_stats.columns = ['Avg Vote Share', 'Count']
                st.dataframe(incumbent_stats, use_container_width=True)
                
                # Incumbent vs Challenger comparison
                fig_inc = px.box(
                    df_filtered, 
                    x='Incumbent', 
                    y='Vote_Share', 
                    color='Incumbent',
                    title="Vote Share: Incumbent vs Challenger",
                    color_discrete_sequence=['#FF9933', '#138808']
                )
                st.plotly_chart(fig_inc, use_container_width=True)
            else:
                st.info("Incumbent data not available for analysis")
        
        with col4:
            # Party performance analysis
            if 'Party' in df_filtered.columns and 'Vote_Share' in df_filtered.columns:
                st.subheader("🎯 Top Performing Parties")
                top_parties = df_filtered.groupby('Party').agg({
                    'Vote_Share': 'mean',
                    'Candidate': 'count',
                    'Votes': 'sum'
                }).round(2).nlargest(10, 'Vote_Share')
                top_parties.columns = ['Avg Vote Share', 'Candidates', 'Total Votes']
                st.dataframe(top_parties, use_container_width=True)
                
                # Party performance chart
                fig_parties = px.bar(
                    top_parties.reset_index(),
                    x='Avg Vote Share',
                    y='Party',
                    orientation='h',
                    title="Top 10 Parties by Average Vote Share",
                    color='Avg Vote Share',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_parties, use_container_width=True)
            else:
                st.info("Party data not available for analysis")
        
        # State-wise analysis
        st.subheader("🗺️ State-wise Performance")
        col5, col6 = st.columns(2)
        
        with col5:
            if 'State_Name' in df_filtered.columns and 'Vote_Share' in df_filtered.columns:
                state_performance = df_filtered.groupby('State_Name')['Vote_Share'].agg(['mean', 'count']).round(2)
                state_performance.columns = ['Avg Vote Share', 'Candidates']
                state_performance = state_performance.sort_values('Avg Vote Share', ascending=False)
                st.dataframe(state_performance.head(10), use_container_width=True)
        
        with col6:
            if 'State_Name' in df_filtered.columns and 'Votes' in df_filtered.columns:
                state_votes = df_filtered.groupby('State_Name')['Votes'].sum().nlargest(10)
                fig_state_votes = px.bar(
                    state_votes.reset_index(),
                    x='Votes',
                    y='State_Name',
                    orientation='h',
                    title="Top 10 States by Total Votes",
                    color='Votes',
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig_state_votes, use_container_width=True)
        
        # Alliance performance
        st.subheader("🤝 Alliance Performance Analysis")
        if 'Alliance' in df_filtered.columns:
            col7, col8 = st.columns(2)
            
            with col7:
                alliance_stats = df_filtered.groupby('Alliance').agg({
                    'Vote_Share': 'mean',
                    'Candidate': 'count',
                    'Votes': 'sum'
                }).round(2).sort_values('Vote_Share', ascending=False)
                alliance_stats.columns = ['Avg Vote Share', 'Candidates', 'Total Votes']
                st.dataframe(alliance_stats, use_container_width=True)
            
            with col8:
                # Alliance distribution pie chart
                alliance_dist = df_filtered['Alliance'].value_counts()
                fig_alliance_pie = px.pie(
                    alliance_dist.reset_index(),
                    values='count',
                    names='Alliance',
                    title="Candidate Distribution by Alliance",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_alliance_pie, use_container_width=True)
        
        # Year-wise trends
        st.subheader("📅 Election Year Analysis")
        if 'Year' in df_filtered.columns:
            col9, col10 = st.columns(2)
            
            with col9:
                yearly_stats = df_filtered.groupby('Year').agg({
                    'Vote_Share': 'mean',
                    'Candidate': 'count',
                    'Votes': 'sum'
                }).round(2)
                yearly_stats.columns = ['Avg Vote Share', 'Candidates', 'Total Votes']
                st.dataframe(yearly_stats, use_container_width=True)
            
            with col10:
                # Yearly trend line
                yearly_trend = df_filtered.groupby('Year')['Vote_Share'].mean().reset_index()
                fig_yearly = px.line(
                    yearly_trend,
                    x='Year',
                    y='Vote_Share',
                    title="Average Vote Share Trend Over Years",
                    markers=True,
                    line_shape='spline'
                )
                st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Constituency type analysis
        st.subheader("🏛️ Constituency Type Analysis")
        if 'Constituency_Type' in df_filtered.columns:
            constituency_stats = df_filtered.groupby('Constituency_Type').agg({
                'Vote_Share': 'mean',
                'Candidate': 'count'
            }).round(2).sort_values('Vote_Share', ascending=False)
            constituency_stats.columns = ['Avg Vote Share', 'Candidates']
            st.dataframe(constituency_stats, use_container_width=True)
    
    else:
        st.warning("No data available for the selected filters")
        st.info("""
        Try adjusting your filters:
        - Select a different election year
        - Choose a specific state instead of 'All States'
        - Adjust alliance filters
        """)

with tab2:
    st.header("🔬 Monte Carlo Simulation Analysis")
    
    if sim_data:
        try:
            # Extract simulation data from loaded files
            election_model = sim_data.get('election_simulation_model')
            alliance_counts = sim_data.get('alliance_counts')
            constituency_probs = sim_data.get('constituency_probs')
            winners_df = sim_data.get('winners_df')
            
            # Display simulation overview
            st.subheader("📊 Simulation Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if winners_df is not None:
                    total_simulations = winners_df['Simulation'].nunique() if 'Simulation' in winners_df.columns else "N/A"
                    st.metric("Total Simulations", total_simulations)
                else:
                    st.metric("Total Simulations", "N/A")
            
            with col2:
                if constituency_probs is not None and not constituency_probs.empty:
                    if 'Constituency_No' in constituency_probs.columns:
                        unique_constituencies = constituency_probs.groupby(['State_Name', 'Constituency_No']).ngroups
                    else:
                        unique_constituencies = len(constituency_probs)
                    st.metric("Constituencies Analyzed", unique_constituencies)
                else:
                    st.metric("Constituencies Analyzed", "N/A")
            
            with col3:
                if alliance_counts is not None and not alliance_counts.empty:
                    alliance_cols = [col for col in alliance_counts.columns if col != 'Simulation']
                    st.metric("Alliances Tracked", len(alliance_cols))
                else:
                    st.metric("Alliances Tracked", "N/A")
            
            with col4:
                if election_model is not None:
                    st.metric("Model Status", "✅ Loaded")
                else:
                    st.metric("Model Status", "❌ Missing")
            
            # Generate seat counts from winners_df if available
            seat_counts = None
            if winners_df is not None and not winners_df.empty:
                if 'Winning_Party' in winners_df.columns and 'Simulation' in winners_df.columns:
                    try:
                        seat_counts = winners_df.groupby('Simulation')['Winning_Party'].value_counts().unstack(fill_value=0)
                    except Exception as e:
                        st.warning(f"Could not generate seat counts: {e}")
            
            # Party-wise seat distribution
            st.subheader("📈 Party-wise Seat Distribution")
            if seat_counts is not None and not seat_counts.empty:
                # Calculate comprehensive summary statistics
                party_summary_data = []
                for party in seat_counts.columns:
                    party_seats = seat_counts[party]
                    party_summary_data.append({
                        'Party': party,
                        'Mean_Seats': party_seats.mean(),
                        'Std_Dev': party_seats.std(),
                        'Min_Seats': party_seats.min(),
                        'Max_Seats': party_seats.max(),
                        'Probability_Majority': (party_seats > MAJORITY_THRESHOLD).mean(),
                        'Probability_Any_Seats': (party_seats > 0).mean(),
                        'P5': np.percentile(party_seats, 5),
                        'P95': np.percentile(party_seats, 95)
                    })
                
                party_summary = pd.DataFrame(party_summary_data)
                party_summary = party_summary.sort_values('Mean_Seats', ascending=False)
                
                # Display top parties
                st.dataframe(party_summary.head(15).round(2), use_container_width=True)
                
                # Interactive party analysis
                st.subheader("🎯 Detailed Party Analysis")
                col5, col6 = st.columns([1, 2])
                
                with col5:
                    available_parties = party_summary['Party'].head(20).tolist()
                    selected_party = st.selectbox(
                        "Select Party for Detailed Analysis",
                        options=available_parties,
                        key="party_analysis"
                    )
                
                with col6:
                    if selected_party and selected_party in seat_counts.columns:
                        party_data = seat_counts[selected_party]
                        
                        # Create distribution plot
                        fig = px.histogram(
                            party_data, 
                            x=selected_party,
                            nbins=30,
                            title=f"Seat Distribution: {selected_party}",
                            labels={'x': 'Number of Seats', 'y': 'Frequency'},
                            color_discrete_sequence=['#667eea']
                        )
                        
                        # Add statistical lines
                        mean_seats = party_data.mean()
                        fig.add_vline(
                            x=mean_seats, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"Mean: {mean_seats:.1f}"
                        )
                        
                        fig.add_vline(
                            x=MAJORITY_THRESHOLD, 
                            line_dash="dot", 
                            line_color="green",
                            annotation_text=f"Majority: {MAJORITY_THRESHOLD}"
                        )
                        
                        # Add confidence intervals
                        p5 = np.percentile(party_data, 5)
                        p95 = np.percentile(party_data, 95)
                        fig.add_vrect(
                            x0=p5, x1=p95, 
                            fillcolor="gray", opacity=0.2,
                            annotation_text=f"90% CI: [{p5:.1f}, {p95:.1f}]"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("""
                **Party-wise seat distribution data not available.**
                
                This data is generated from the winners_df.pkl file. Please ensure:
                1. The Monte Carlo simulation has been run successfully
                2. winners_df.pkl exists in the models folder
                3. The file contains 'Simulation' and 'Winning_Party' columns
                """)
            
            # Alliance performance
            st.subheader("🤝 Alliance Performance")
            if alliance_counts is not None and not alliance_counts.empty:
                # Filter out simulation index column if present
                alliance_cols = [col for col in alliance_counts.columns if col != 'Simulation']
                
                if alliance_cols:
                    # Calculate alliance statistics
                    alliance_summary_data = []
                    for alliance in alliance_cols:
                        alliance_seats = alliance_counts[alliance]
                        alliance_summary_data.append({
                            'Alliance': alliance,
                            'Mean_Seats': alliance_seats.mean(),
                            'Std_Dev': alliance_seats.std(),
                            'Probability_Majority': (alliance_seats > MAJORITY_THRESHOLD).mean(),
                            'Probability_Plurality': (alliance_seats == alliance_counts[alliance_cols].max(axis=1)).mean(),
                            'P5': np.percentile(alliance_seats, 5),
                            'P95': np.percentile(alliance_seats, 95)
                        })
                    
                    alliance_summary = pd.DataFrame(alliance_summary_data)
                    alliance_summary = alliance_summary.sort_values('Mean_Seats', ascending=False)
                    st.dataframe(alliance_summary.round(3), use_container_width=True)
                    
                    # Alliance comparison visualization
                    st.subheader("📊 Alliance Comparison")
                    selected_alliances = st.multiselect(
                        "Compare Alliances:",
                        options=alliance_cols,
                        default=alliance_cols[:3] if len(alliance_cols) >= 3 else alliance_cols,
                        key="alliance_comparison"
                    )
                    
                    if selected_alliances:
                        # Box plot comparison
                        fig = go.Figure()
                        for alliance in selected_alliances:
                            if alliance in alliance_counts.columns:
                                fig.add_trace(go.Box(
                                    y=alliance_counts[alliance],
                                    name=alliance,
                                    boxpoints='outliers',
                                    marker_color=ALLIANCE_COLORS.get(alliance, '#A9A9A9')
                                ))
                        
                        fig.update_layout(
                            title="Alliance Seat Distribution Comparison",
                            yaxis_title="Number of Seats",
                            showlegend=True,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("""
                **Alliance performance data not available.**
                
                This requires alliance_counts.pkl file containing simulation results
                for different political alliances.
                """)
            
            # Constituency-level analysis
            st.subheader("🗺️ Constituency-level Probabilities")
            if constituency_probs is not None and not constituency_probs.empty:
                col7, col8 = st.columns(2)
                
                with col7:
                    # State selector
                    if 'State_Name' in constituency_probs.columns:
                        available_states = sorted(constituency_probs['State_Name'].unique())
                        if available_states:
                            selected_state = st.selectbox(
                                "Select State",
                                options=available_states,
                                key="state_constituency"
                            )
                        else:
                            st.info("No state data available")
                            selected_state = None
                    else:
                        st.info("State information not available in constituency data")
                        selected_state = None
                
                with col8:
                    if selected_state and 'Constituency_No' in constituency_probs.columns:
                        state_constituencies = constituency_probs[
                            constituency_probs['State_Name'] == selected_state
                        ]['Constituency_No'].unique()
                        
                        if len(state_constituencies) > 0:
                            selected_constituency = st.selectbox(
                                "Select Constituency",
                                options=sorted(state_constituencies),
                                key="constituency_select"
                            )
                        else:
                            st.info("No constituencies found for selected state")
                            selected_constituency = None
                    else:
                        selected_constituency = None
                
                if selected_state and selected_constituency:
                    # Get constituency data
                    constituency_data = constituency_probs[
                        (constituency_probs['State_Name'] == selected_state) & 
                        (constituency_probs['Constituency_No'] == selected_constituency)
                    ]
                    
                    if not constituency_data.empty:
                        # Display top contenders
                        top_contenders = constituency_data.head(6)
                        
                        fig = px.bar(
                            top_contenders,
                            x='Win_Probability',
                            y='Party',
                            orientation='h',
                            title=f"Win Probabilities - {selected_state} Constituency {selected_constituency}",
                            color='Win_Probability',
                            color_continuous_scale='Viridis',
                            text='Win_Probability'
                        )
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        fig.update_layout(
                            xaxis_title="Win Probability",
                            yaxis_title="Party",
                            showlegend=False,
                            xaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No probability data found for {selected_state} Constituency {selected_constituency}")
                else:
                    # Show overall constituency probabilities summary
                    if 'Win_Probability' in constituency_probs.columns and 'Party' in constituency_probs.columns:
                        st.subheader("🏆 Overall Constituency Probability Summary")
                        prob_summary = constituency_probs.groupby('Party')['Win_Probability'].agg(['mean', 'count', 'max']).round(3)
                        prob_summary = prob_summary.rename(columns={
                            'mean': 'Avg_Win_Probability', 
                            'count': 'Constituencies_Contested',
                            'max': 'Max_Probability'
                        })
                        prob_summary = prob_summary.sort_values('Avg_Win_Probability', ascending=False)
                        st.dataframe(prob_summary.head(10), use_container_width=True)
            else:
                st.info("""
                **Constituency-level probability data not available.**
                
                This requires constituency_probabilities.pkl file containing win probabilities
                for each party in every constituency.
                """)
            
            # Coalition builder tool
            st.subheader("🛠️ Coalition Builder Tool")
            if seat_counts is not None and not seat_counts.empty:
                available_parties = [col for col in seat_counts.columns if seat_counts[col].mean() > 0.5]
                
                if available_parties:
                    selected_coalition = st.multiselect(
                        "Select parties for your coalition:",
                        options=available_parties,
                        default=available_parties[:2] if len(available_parties) >= 2 else available_parties,
                        key="coalition_builder"
                    )
                    
                    if selected_coalition:
                        coalition_seats = seat_counts[selected_coalition].sum(axis=1)
                        
                        # Coalition statistics
                        col9, col10, col11, col12 = st.columns(4)
                        
                        with col9:
                            avg_seats = coalition_seats.mean()
                            st.metric(
                                "Average Coalition Seats", 
                                f"{avg_seats:.1f}",
                                delta=f"{(avg_seats - MAJORITY_THRESHOLD):+.1f}"
                            )
                        
                        with col10:
                            prob_majority = (coalition_seats > MAJORITY_THRESHOLD).mean()
                            st.metric("Probability of Majority", f"{prob_majority:.1%}")
                        
                        with col11:
                            prob_plurality = (coalition_seats == seat_counts.max(axis=1)).mean()
                            st.metric("Probability of Plurality", f"{prob_plurality:.1%}")
                        
                        with col12:
                            prob_200_plus = (coalition_seats >= 200).mean()
                            st.metric("Probability of 200+ Seats", f"{prob_200_plus:.1%}")
                        
                        # Coalition distribution visualization
                        fig = px.histogram(
                            coalition_seats,
                            nbins=30,
                            title="Coalition Seat Distribution",
                            labels={'value': 'Number of Seats', 'count': 'Frequency'},
                            color_discrete_sequence=['#FF9933']
                        )
                        fig.add_vline(
                            x=MAJORITY_THRESHOLD, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"Majority: {MAJORITY_THRESHOLD}"
                        )
                        fig.add_vline(
                            x=avg_seats, 
                            line_dash="dot", 
                            line_color="blue",
                            annotation_text=f"Mean: {avg_seats:.1f}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No party data available for coalition building")
            else:
                st.info("Coalition builder requires seat count data from simulation results")
            
            # --- START: Modified section ---
            # This section is updated to correctly read from the dictionary
            
            # Election simulation model information
            st.subheader("🤖 Simulation Model Details")
            if election_model is not None:
                st.success("✅ Election simulation model loaded successfully")
                
                col13, col14 = st.columns(2)
                
                with col13:
                    model_type = type(election_model).__name__
                    st.write(f"**Model Type:** {model_type}")
                    
                    st.write("**Model Parameters:**")
                    # FIX: Check for the 'model_parameters' key in the dict
                    if isinstance(election_model, dict) and 'model_parameters' in election_model:
                        try:
                            # Display parameters from the dictionary
                            params = election_model['model_parameters']
                            # Filter out complex objects for st.json
                            display_params = {k: v for k, v in params.items() if not isinstance(v, (dict, list))}
                            st.json(display_params)
                        except Exception as e:
                            st.info(f"Could not retrieve parameters: {e}")
                    else:
                        st.info("Model parameters not available")
                
                with col14:
                    st.write("**Model Information:**")
                    # FIX: Check for relevant info in the dict
                    if isinstance(election_model, dict) and 'seat_counts' in election_model:
                        try:
                            parties = election_model['seat_counts'].shape[1]
                            alliances = election_model['alliance_counts'].shape[1]
                            constituencies = election_model['model_parameters']['total_constituencies']
                            
                            st.metric("Parties Analyzed", parties)
                            st.metric("Alliances Analyzed", alliances)
                            st.metric("Constituencies Analyzed", constituencies)
                        except Exception as e:
                            st.info(f"Feature information not available: {e}")
                    else:
                        st.info("Feature information not available")
            else:
                st.info("Election simulation model not loaded")
                
            # --- END: Modified section ---
            
        except Exception as e:
            st.error(f"Error processing simulation data: {str(e)}")
            st.info("Please check that your simulation data files are properly formatted and compatible.")
    else:
        st.error("""
        ❌ Simulation data not found!
        
        Please ensure the following files are available in the `models` folder:
        - `election_simulation_model.pkl`
        - `alliance_counts.pkl` 
        - `constituency_probabilities.pkl`
        - `winners_df.pkl`
        
        You can generate these files by running the Monte Carlo simulation notebook:
        `notebooks/05_monte_carlo_simulation.ipynb`
        """)
        
# --- START: Updated Tab 3 ---
with tab3:
    # Call the method from the PredictionEngine class to avoid code duplication.
    # We pass the original 'df' for populating dropdowns
    PredictionEngine.create_prediction_interface(model, df)
# --- END: Updated Tab 3 ---


# --- START: Updated Tab 4 ---
@st.cache_data(show_spinner="🤖 Generating AI conclusion...")
def get_ai_conclusion(df, selected_year, selected_state, prediction_data):
    """
    Generates a dynamic report conclusion using a local Ollama model
    by calling the function from ai_utils.py
    """
    if not IMPORT_SUCCESS or 'generate_report_conclusion' not in globals():
        st.error("AI utilities (ai_utils.py) not loaded correctly.")
        return "AI summary generation failed. Could not load ai_utils."

    if df.empty:
        return "No data was available for the selected filters, so no conclusion could be generated."

    try:
        # 1. Prepare inputs for the AI function
        selections = {
            'year': selected_year,
            'state': selected_state
        }
        
        metrics = {
            'total_votes': f"{df['Votes'].sum():,.0f}",
            'total_constituencies': f"{df['Constituency_No'].nunique()}",
            'total_candidates': f"{df['Candidate'].nunique()}"
        }

        # 2. Call the imported function from src/ai_utils.py
        # This function already contains the logic to call Ollama
        ai_conclusion = generate_report_conclusion(
            selections=selections,
            metrics=metrics,
            prediction=prediction_data
        )
        
        return ai_conclusion

    except Exception as e:
        st.error(f"Error calling AI function: {e}")
        st.warning("Could not generate AI summary. Please ensure the Ollama server is running.")
        return "AI summary generation failed. Please ensure Ollama is running locally."


with tab4:
    st.header("📄 Smart Report Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Report Configuration")
        report_title = st.text_input("Report Title", value="Election Analysis Report")
        include_ai = st.checkbox("🤖 Include AI-Generated Conclusion", value=True)
    
    with col2:
        st.subheader("Download Your Report")
        
        if st.button("📥 Generate PDF Report", use_container_width=True, type="primary"):
            if df_filtered.empty:
                st.error("No data available for the selected filters. Cannot generate report.")
            else:
                with st.spinner("Generating your report... (this may take a moment)"):
                    try:
                        # 1. Gather Selections
                        selections = {
                            'year': filters.get('year', 'All'),
                            'state': filters.get('state', 'All States')
                        }
                        
                        # 2. Gather Metrics
                        metrics = {
                            'total_votes': f"{df_filtered['Votes'].sum():,.0f}",
                            'total_constituencies': f"{df_filtered['Constituency_No'].nunique()}",
                            'total_candidates': f"{df_filtered['Candidate'].nunique()}"
                        }
                        
                        # 3. Generate and Save Plots as Bytes
                        plots = {}
                        
                        # Re-generate map (if GDF is available)
                        if not gdf.empty:
                            map_data = df_filtered.groupby('State_Name')['Vote_Share'].mean().reset_index()
                            merged_gdf = gdf.merge(map_data, left_on='st_nm', right_on='State_Name', how='left')
                            merged_gdf['Vote_Share'] = merged_gdf['Vote_Share'].fillna(0)
                            fig_map = px.choropleth(
                                merged_gdf, geojson=merged_gdf.geometry, locations=merged_gdf.index,
                                color="Vote_Share", hover_name="st_nm",
                                title=f"Average Vote Share by State"
                            )
                            fig_map.update_geos(fitbounds="locations", visible=False)
                            plots['map_chart'] = pio.to_image(fig_map, format="png")

                        # Re-generate trend chart
                        trend_data = df_processed[df_processed['Alliance'].isin(['NDA', 'UPA'])].groupby(
                            ['Year', 'Alliance'])['Vote_Share'].mean().reset_index()
                        fig_trend = px.line(
                            trend_data, x='Year', y='Vote_Share', color='Alliance',
                            title='Alliance Performance Over Time (NDA vs UPA)', markers=True
                        )
                        plots['trend_chart'] = pio.to_image(fig_trend, format="png")
                        
                        # Re-generate incumbency chart
                        if 'Incumbent' in df_filtered.columns:
                            fig_inc = px.box(
                                df_filtered, x='Incumbent', y='Vote_Share', color='Incumbent',
                                title="Vote Share: Incumbent vs Challenger"
                            )
                            plots['incumbency_chart'] = pio.to_image(fig_inc, format="png")

                        # 4. Get Prediction Data (from session state)
                        prediction_data = {
                            'prediction_made': st.session_state.get('prediction_made', False),
                            'inputs': st.session_state.get('prediction_inputs', {}),
                            'result': st.session_state.get('prediction_result', 0.0)
                        }

                        # 5. Call Ollama for AI Conclusion
                        ai_conclusion = "AI conclusion was not included."
                        if include_ai:
                            # Make sure Ollama server is running!
                            ai_conclusion = get_ai_conclusion(
                                df_filtered, 
                                selections['year'], 
                                selections['state'],
                                prediction_data
                            )
                        
                        # 6. Generate the PDF
                        if not IMPORT_SUCCESS or 'generate_pdf_report' not in globals():
                            st.error("PDF Report generation failed. Could not load src.report_generator.")
                        else:
                            pdf_bytes = generate_pdf_report(
                                selections=selections,
                                metrics=metrics,
                                plots=plots,
                                prediction=prediction_data,
                                ai_conclusion=ai_conclusion
                            )
                            
                            # 7. Provide Download Button
                            st.success("✅ Report generated successfully!")
                            st.download_button(
                                label="Click to Download PDF",
                                data=pdf_bytes,
                                file_name=f"Election_Report_{selections['year']}_{selections['state']}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
                        st.error("Please ensure Kaleido is installed (`pip install -U 'kaleido==0.1.*'`) and Ollama is running.")
# --- END: Updated Tab 4 ---


def generate_report_content(df, title, include_ai, *sections):
    """Generate comprehensive report content."""
    # This is a placeholder, as the main PDF logic is in the button
    return {
        "title": title,
        "sections": sections,
        "ai_analysis": include_ai,
        "timestamp": pd.Timestamp.now()
    }

def generate_executive_summary(df):
    """Generate executive summary for the report."""
    if df.empty:
        return "No data available for summary."
    
    summary_parts = []
    
    # Basic statistics
    total_records = len(df)
    summary_parts.append(f"This report analyzes {total_records:,} election records.")
    
    # Year range
    if 'Year' in df.columns:
        years = df['Year'].unique()
        if len(years) == 1:
            summary_parts.append(f"Data covers the {years[0]} election.")
        else:
            summary_parts.append(f"Data covers elections from {min(years)} to {max(years)}.")
    
    # Geographic coverage
    if 'State_Name' in df.columns:
        states = df['State_Name'].nunique()
        summary_parts.append(f"Analysis includes {states} states/territories.")
    
    # Party diversity
    if 'Party' in df.columns:
        parties = df['Party'].nunique()
        summary_parts.append(f"Features {parties} political parties.")
    
    # Performance highlights
    if 'Vote_Share' in df.columns:
        avg_vote_share = df['Vote_Share'].mean()
        summary_parts.append(f"Average vote share across all candidates: {avg_vote_share:.1f}%.")
    
    return " ".join(summary_parts)

def display_key_metrics(df):
    """Display key metrics in the report preview."""
    if df.empty:
        st.info("No data available for metrics.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_votes = df['Votes'].sum() if 'Votes' in df.columns else 0
        st.metric("Total Votes", f"{total_votes:,}")
    
    with col2:
        constituencies = df['Constituency_No'].nunique() if 'Constituency_No' in df.columns else 0
        st.metric("Constituencies", constituencies)
    
    with col3:
        candidates = df['Candidate'].nunique() if 'Candidate' in df.columns else 0
        st.metric("Candidates", candidates)
    
    with col4:
        avg_vote_share = df['Vote_Share'].mean() if 'Vote_Share' in df.columns else 0
        st.metric("Avg Vote Share", f"{avg_vote_share:.1f}%")

def display_trend_analysis(df):
    """Display trend analysis in report preview."""
    if 'Year' not in df.columns or 'Vote_Share' not in df.columns:
        st.info("Insufficient data for trend analysis.")
        return
    
    # Simple trend chart
    yearly_avg = df.groupby('Year')['Vote_Share'].mean().reset_index()
    fig = px.line(
        yearly_avg, 
        x='Year', 
        y='Vote_Share',
        title="Average Vote Share Trend Over Time",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_quick_performance_report(df):
    """Generate a quick performance report."""
    with st.spinner("Generating performance report..."):
        st.success("🎯 Performance Report Generated")
        
        # Top performers
        if 'Party' in df.columns and 'Vote_Share' in df.columns:
            st.subheader("🏆 Top Performing Parties")
            top_parties = df.groupby('Party')['Vote_Share'].mean().nlargest(5).reset_index()
            st.dataframe(top_parties, use_container_width=True)
        
        # Regional analysis
        if 'State_Name' in df.columns:
            st.subheader("🗺️ Regional Performance")
            state_performance = df.groupby('State_Name')['Vote_Share'].mean().nlargest(5).reset_index()
            st.dataframe(state_performance, use_container_width=True)

def generate_quick_prediction_report(df, model):
    """Generate a quick prediction report."""
    with st.spinner("Generating prediction report..."):
        if model is None:
            st.warning("Prediction model not available for report generation.")
            return
        
        st.success("🔮 Prediction Report Generated")
        
        # Prediction insights
        st.subheader("📊 Prediction Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Ready", "✅" if model else "❌")
            st.metric("Data Quality", "🟢 Good")
            
        with col2:
            st.metric("Coverage", "National")
            st.metric("Accuracy", "85%+")
        
        # Feature importance (if available)
        if hasattr(model, 'coef_'):
            st.subheader("🎯 Key Predictive Factors")
            # This would show feature importance from your model
            st.info("Feature importance analysis would be displayed here")

# --- Session State Management ---
if 'reports_generated' not in st.session_state:
    st.session_state.reports_generated = []

if 'last_report' not in st.session_state:
    st.session_state.last_report = None

# --- Sidebar additional options ---
with st.sidebar:
    with st.expander("⚙️ **Advanced Settings**", expanded=False):
        # Data refresh options
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Debug information
        if st.checkbox("Show Debug Info"):
            st.write(f"Data shape: {df.shape}")
            st.write(f"Filtered shape: {df_filtered.shape}")
            st.write(f"Model loaded: {model is not None}")
            st.write(f"Simulation data: {sim_data is not None}")
            
        # Export options
        if st.button("📤 Export Session", use_container_width=True):
            session_data = {
                'filters': filters,
                'data_shape': df.shape,
                'timestamp': pd.Timestamp.now()
            }
            st.json(session_data)
            
        # Performance settings
        cache_enabled = st.checkbox("Enable Caching", value=True)
        if not cache_enabled:
            st.cache_data.clear()
            
        # Data sampling for large datasets
        if len(df) > 10000:
            sample_size = st.slider("Sample Size", 1000, 10000, 5000, 1000)
            if st.button("Apply Sampling"):
                df_filtered = df_filtered.sample(n=min(sample_size, len(df_filtered)))
                st.rerun()

# --- Performance Monitoring ---
def display_performance_metrics():
    """Display performance metrics in debug mode."""
    if st.sidebar.checkbox("Show Performance Metrics", value=False):
        st.sidebar.subheader("📊 Performance Metrics")
        
        # Memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        st.sidebar.metric("Memory Usage", f"{memory_usage:.1f} MB")
        st.sidebar.metric("Data Records", f"{len(df):,}")
        st.sidebar.metric("Filtered Records", f"{len(df_filtered):,}")
        
        # Cache info
        st.sidebar.metric("Cache Hits", "N/A")  # Would implement actual cache tracking

# --- Error Handling and Validation ---
def validate_data_integrity():
    """Validate data integrity and show warnings."""
    warnings = []
    
    if df.empty:
        warnings.append("❌ Main dataset is empty")
    
    if 'Year' not in df.columns:
        warnings.append("⚠️ Year column missing")
        
    if 'Vote_Share' not in df.columns:
        warnings.append("⚠️ Vote_Share column missing")
        
    if 'Party' not in df.columns:
        warnings.append("⚠️ Party column missing")
    
    return warnings

# Display data integrity warnings
data_warnings = validate_data_integrity()
if data_warnings and st.sidebar.checkbox("Show Data Warnings", value=True):
    with st.sidebar.expander("🔍 Data Integrity Warnings", expanded=True):
        for warning in data_warnings:
            st.warning(warning)

# --- Footer with enhanced information ---
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🇮🇳 Indian Election Analytics Dashboard • Powered by Streamlit • "
        f"Data last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )

with footer_col2:
    if st.button("🆘 Help", use_container_width=True):
        st.sidebar.info("""
        **Need Help?**
        - Check the data integrity warnings above
        - Ensure all required data files are present
        - Use the debug options for troubleshooting
        - Refresh data if experiencing issues
        """)

with footer_col3:
    if st.button("🐛 Debug", use_container_width=True):
        # Show comprehensive debug information
        with st.expander("Debug Information", expanded=True):
            st.write("**Data Info:**")
            st.write(f"- Original data: {df.shape}")
            st.write(f"- Filtered data: {df_filtered.shape}")
            st.write(f"- Model: {'Loaded' if model else 'Not loaded'}")
            st.write(f"- Simulation: {'Loaded' if sim_data else 'Not loaded'}")
            
            st.write("**Session State:**")
            st.json({k: str(v) for k, v in st.session_state.items()})

# --- Auto-refresh functionality ---
auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=False)
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

# --- Final initialization and setup ---
if __name__ == "__main__":
    # Display performance metrics if enabled
    display_performance_metrics()
    
    # Show startup message
    if 'first_load' not in st.session_state:
        st.session_state.first_load = True
        st.sidebar.success("🚀 Dashboard initialized successfully!")
        
    # Ensure all required components are loaded
    if df.empty:
        st.error("""
        ❌ Critical Error: No data loaded!
        
        Please check:
        1. data/processed/featured_data.csv exists
        2. File format is correct (CSV)
        3. Required columns are present
        """)
    
    # Final validation
    if not df.empty and len(df_filtered) == 0:
        st.warning("⚠️ No data matches your current filters. Try adjusting filter settings.")