# 💰 ReadyFunds Analytics Dashboard

**Single-Page Data-Driven Credit Solutions for MSMEs**

## 🚀 Quick Start

### Local Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**
   ```bash
   streamlit run app.py
   ```

3. **Open browser**
   - App opens automatically at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Step-by-Step

1. **Upload to GitHub**
   - Create a new repository on GitHub
   - Upload these files:
     - `app.py`
     - `requirements.txt`
     - `.gitignore`
     - `.streamlit/config.toml`
     - `README.md`
     - `msme_refined_dataset.xlsx`

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `app.py`
   - Click "Deploy"

3. **Done!**
   - Your dashboard will be live in 2-3 minutes
   - URL: `https://[your-username]-[repo-name]-app-xxxxx.streamlit.app`

## 📊 Features

### 🏠 Overview
- Market size and key metrics
- Industry distribution
- Adoption likelihood analysis

### 📊 Executive Dashboard
- ML model performance metrics
- Classification, Regression, Clustering results
- Revenue projections (Conservative, Base, Aggressive)
- Feature importance analysis

### 🎯 Customer Intelligence
- 5 customer personas/segments
- Cluster distribution and adoption rates
- Association Rule Mining results
- Product affinity patterns (Support, Confidence, Lift)

### 🔮 Predictive Engine
- **Single Customer Prediction**: Score individual leads
- **Bulk Upload**: Score hundreds of customers via CSV
- Get predictions for:
  - Adoption probability
  - Credit need forecast
  - Customer segment assignment
  - Risk score
  - Pricing recommendations

### 🛡️ Risk Analytics
- Portfolio risk distribution
- Default history analysis
- Risk-based pricing matrix
- Risk categorization (High/Medium/Low)

## 🤖 ML Models

| Model | Algorithm | Metrics |
|-------|-----------|---------|
| Classification | XGBoost | Accuracy, Precision, Recall, F1, ROC-AUC |
| Regression | XGBoost | R², RMSE, MAE, MAPE |
| Clustering | K-Means (5 clusters) | Silhouette Score |
| Association Rules | Apriori | Support, Confidence, Lift |

## 📁 Project Structure

```
readyfunds_simple/
├── app.py                    # Main application (all features in one file)
├── requirements.txt          # Python dependencies
├── msme_refined_dataset.xlsx # Dataset
├── .gitignore               # Git ignore rules
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # This file
```

## 🔧 Troubleshooting

### "Error loading dataset"
- Ensure `msme_refined_dataset.xlsx` is in the same folder as `app.py`

### "Module not found"
- Run: `pip install -r requirements.txt`

### Deployment fails
- Check Streamlit Cloud logs for specific errors
- Verify all files are uploaded to GitHub
- Ensure requirements.txt has correct package names

## 📈 Expected Results

- **Dataset**: 2,000 MSME responses
- **Classification Accuracy**: 75-85%
- **Regression R²**: 70-80%
- **Clustering Quality**: Silhouette > 0.4
- **Association Rules**: 50-200+ patterns

## 💡 Usage Tips

1. **First Run**: Models train automatically (takes 2-3 minutes)
2. **Single Prediction**: Use for quick customer assessment
3. **Bulk Upload**: Use for lead scoring campaigns
4. **Download Results**: Export scored customers for sales team

## 🎯 Business Impact

Use this dashboard to:
- Score leads before sales calls
- Predict credit needs for capital planning
- Segment customers for personalized marketing
- Find cross-sell opportunities
- Manage portfolio risk
- Implement risk-based pricing

## 📞 Support

For issues:
1. Check error messages carefully
2. Verify all files are present
3. Ensure dataset is correctly named
4. Review Streamlit Cloud deployment logs

---

**Built for ReadyFunds** | Empowering MSMEs through Data-Driven Financial Services
