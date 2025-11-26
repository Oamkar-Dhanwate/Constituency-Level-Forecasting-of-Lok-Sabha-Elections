-----
# Constituency Level Forecasting of Lok Sabha Elections

## 📊 Project Overview

This project provides a robust, data-driven framework for forecasting Indian Lok Sabha elections. It combines machine learning regression models to predict constituency-level vote shares with Monte Carlo simulations to translate those vote shares into probabilistic seat outcomes. The system includes an interactive Streamlit dashboard for analytics, "what-if" scenario planning, and automated PDF report generation.

## 🚀 Key Features

  * **Historical Data Analytics**: Analyze election trends from 2009-2019, including party performance, incumbency effects, and voter turnout.
  * **Vote Share Prediction**: Uses Ridge/Lasso regression models to predict vote shares based on features like state, constituency type, incumbency, and historical performance.
  * **Monte Carlo Simulation**: A stochastic simulation engine (using Dirichlet distributions) that runs thousands of election scenarios to estimate:
      * Win probabilities for every constituency.
      * National and state-level seat count distributions.
      * Probability intervals (e.g., 90% confidence) for party/alliance performance.
  * **Alliance Management**: Dynamic mapping of parties to alliances (NDA, UPA, Left, etc.) to analyze coalition impacts.
  * **Interactive Dashboard**: A feature-rich **Streamlit** application that allows users to:
      * Visualize geographical vote shares.
      * Run predictions for specific candidates.
      * Build custom post-poll coalitions.
  * **Smart Reporting**: Generates downloadable PDF reports containing visualizations, key metrics, and AI-generated summaries (powered by local LLMs via Ollama).

## 🛠️ Tech Stack

  * **Language**: Python 3.8+
  * **Data Manipulation**: Pandas, NumPy, GeoPandas
  * **Machine Learning**: Scikit-learn, Statsmodels, XGBoost
  * **Simulation**: SciPy (Dirichlet distribution), PyMC3 (optional)
  * **Visualization**: Plotly, Matplotlib, Seaborn
  * **Dashboarding**: Streamlit
  * **Reporting**: FPDF2, Kaleido
  * **AI Integration**: Ollama (for local LLM inference)

## 📂 Project Structure

```
├── data/
│   ├── raw/                # Raw election CSVs and GeoJSON files
│   └── processed/          # Cleaned datasets (featured_data.csv)
├── models/
│   ├── best_ridge_model.joblib       # Trained vote share predictor
│   ├── election_simulation_model.pkl # Serialized simulation results
│   └── ...
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model_OLS.ipynb
│   └── 05_monte_carlo_simulation.ipynb # Core simulation logic
├── src/
│   ├── data_processing.py    # Data loading and cleaning utilities
│   ├── feature_engineering.py# Alliance mapping and feature creation
│   ├── modeling.py           # Model loading and inference wrappers
│   ├── report_generator.py   # PDF report creation logic
│   └── ai_utils.py           # Integration with Ollama for AI summaries
├── dashboard/
│   └── app_02.py             # Main Streamlit dashboard application
├── requirements.txt          # Project dependencies
└── README.md
```

## ⚙️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/constituency-level-forecasting.git
    cd constituency-level-forecasting
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: For PDF image export support, ensure you have `kaleido` installed.*

4.  **Set up AI (Optional):**
    If you want to use the AI summary feature in the reports, download and install [Ollama](https://ollama.com/) and pull a model (e.g., `llama3` or `mistral`).

    ```bash
    ollama pull mistral
    ```

## 🖥️ Usage

### Running the Dashboard

To launch the interactive election analytics dashboard:

```bash
streamlit run dashboard/app_02.py
```

Access the app in your browser at `http://localhost:8501`.

### Running the Simulation

To retrain the simulation model or generate new probability data:

1.  Navigate to the `notebook/` directory.
2.  Open and run `05_monte_carlo_simulation.ipynb`.
3.  This will generate and save the necessary `.pkl` files to the `models/` directory for the dashboard to consume.

## 📊 Methodology

1.  **Data Preprocessing**: Historical election data is cleaned, standardized, and merged with shapefiles.
2.  **Feature Engineering**: Features such as "Incumbency," "Alliance Affiliation," and "Previous Vote Share" are engineered.
3.  **Vote Share Model**: A Ridge Regression model is trained to predict the base vote share % for a candidate given the constituency context.
4.  **Monte Carlo Simulation**:
      * The predicted vote shares serve as the *alpha* parameters for a **Dirichlet distribution**.
      * We sample 1,000+ election scenarios from this distribution to account for volatility and uncertainty.
      * Winners are determined for every constituency in every simulation.
5.  **Aggregation**: Results are aggregated to calculate the probability of winning for every candidate and the expected seat counts for parties/alliances.

## 🤝 Contributing

Contributions are welcome\! Please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
