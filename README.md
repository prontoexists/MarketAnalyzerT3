# Market Analyzer - Streamlit Frontend

A machine learning-powered market analysis tool built with Streamlit and XGBoost

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## Dependencies

- **streamlit**: Web frontend framework
- **xgboost**: Gradient boosting for predictions
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **plotly**: Interactive visualizations
- **python-dotenv**: Environment variable management

## Development

### Adding New Components

Create new UI components in `src/components/` and import them in `app.py`.

### Data Processing

Add data cleaning and transformation functions in `src/utils/`.

### Model Integration

Use `src/model_logic.py` to load and run your XGBoost models.
