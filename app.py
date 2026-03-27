import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error, silhouette_score
)
from xgboost import XGBClassifier, XGBRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Page config
st.set_page_config(page_title="ReadyFunds Analytics", page_icon="💰", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💰 ReadyFunds Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data-Driven Credit Solutions for MSMEs</p>', unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_dataset():
    try:
        df = pd.read_excel('msme_refined_dataset.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def engineer_features(df):
    df_processed = df.copy()
    
    # Digital Readiness Score - with defaults
    comfort_map = {'Yes': 50, 'No': 0}
    data_share_map = {'Yes': 30, 'No': 0}
    digital_trans_map = {'<20%': 5, '20–50%': 10, '50–80%': 15, '80%+': 20}
    
    df_processed['Digital_Readiness_Score'] = (
        df_processed['Comfort with App'].map(comfort_map).fillna(0) +
        df_processed['Willing to Share Data'].map(data_share_map).fillna(0) +
        df_processed['Digital Transactions %'].map(digital_trans_map).fillna(10)
    )
    
    # Credit Health Score - with defaults
    default_weights = {'Never': 40, 'Occasionally': 20, 'Frequently': 0}
    stability_weights = {
        'Highly stable (±10%)': 30, 'Moderate variation (±30%)': 20, 
        'Seasonal': 15, 'Highly unpredictable': 5
    }
    gst_weights = {'Yes': 30, 'No': 0}
    
    df_processed['Credit_Health_Score'] = (
        df_processed['Default History'].map(default_weights).fillna(20) +
        df_processed['Revenue Stability'].map(stability_weights).fillna(15) +
        df_processed['GST Filing'].map(gst_weights).fillna(0)
    )
    
    # Business Maturity Index - with defaults
    years_weights = {'<1 year': 10, '1–3 years': 25, '3–5 years': 40, '5–10 years': 60, '10+ years': 80}
    revenue_weights = {
        '<50,000': 5, '50,000–2,00,000': 15, '2–10 lakh': 30, 
        '10–50 lakh': 50, '50 lakh+': 70
    }
    
    df_processed['Business_Maturity_Index'] = (
        df_processed['Years in Operation'].map(years_weights).fillna(25) * 0.6 +
        df_processed['Monthly Revenue'].map(revenue_weights).fillna(15) * 0.4
    )
    
    # Binary encodings - with defaults
    df_processed['Uses_Credit_Binary'] = df_processed['Uses Credit Currently'].map({'Yes': 1, 'No': 0}).fillna(0)
    df_processed['GST_Filed_Binary'] = df_processed['GST Filing'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Ordinal encodings - with defaults
    df_processed['Years_Ordinal'] = df_processed['Years in Operation'].map({
        '<1 year': 1, '1–3 years': 2, '3–5 years': 3, '5–10 years': 4, '10+ years': 5
    }).fillna(2)
    
    df_processed['Revenue_Ordinal'] = df_processed['Monthly Revenue'].map({
        '<50,000': 1, '50,000–2,00,000': 2, '2–10 lakh': 3, '10–50 lakh': 4, '50 lakh+': 5
    }).fillna(2)
    
    df_processed['Loan_Size_Ordinal'] = df_processed['Typical Loan Size'].map({
        '<1 lakh': 1, '1–5 lakh': 2, '5–20 lakh': 3, '20–50 lakh': 4, '50 lakh+': 5
    }).fillna(2)
    
    return df_processed

@st.cache_resource
def train_all_models(df):
    with st.spinner('Training ML models... This may take 2-3 minutes...'):
        df_processed = engineer_features(df)
        
        # CRITICAL: Fill any NaN values that might have been created
        df_processed = df_processed.fillna(0)
        
        # CLASSIFICATION
        df_processed['Target_Binary'] = df_processed['Likelihood to Use Platform'].apply(
            lambda x: 1 if x in ['Very Likely', 'Likely'] else 0
        )
        
        feature_cols = [
            'Digital_Readiness_Score', 'Credit_Health_Score', 'Business_Maturity_Index',
            'Uses_Credit_Binary', 'GST_Filed_Binary', 'Years_Ordinal', 'Revenue_Ordinal', 'Loan_Size_Ordinal'
        ]
        
        X_class = df_processed[feature_cols].copy()
        # Fill any remaining NaN
        X_class = X_class.fillna(0)
        
        for col in ['Business Type', 'Industry']:
            X_class = pd.concat([X_class, pd.get_dummies(df_processed[col], prefix=col, drop_first=True)], axis=1)
        
        y_class = df_processed['Target_Binary']
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
        
        classification_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=6)
        classification_model.fit(X_train_c, y_train_c)
        
        y_pred_c = classification_model.predict(X_test_c)
        y_pred_proba = classification_model.predict_proba(X_test_c)[:, 1]
        
        class_metrics = {
            'accuracy': accuracy_score(y_test_c, y_pred_c),
            'precision': precision_score(y_test_c, y_pred_c),
            'recall': recall_score(y_test_c, y_pred_c),
            'f1': f1_score(y_test_c, y_pred_c),
            'roc_auc': roc_auc_score(y_test_c, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test_c, y_pred_c),
            'feature_importance': pd.DataFrame({
                'feature': X_class.columns,
                'importance': classification_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # REGRESSION
        credit_mapping = {
            '<2 lakh': 1.0, '2–10 lakh': 6.0, '10–50 lakh': 30.0,
            '50 lakh–1 crore': 75.0, '1 crore+': 150.0
        }
        df_processed['Credit_Need_Numeric'] = df_processed['Expected Credit Need (12 months)'].map(credit_mapping)
        df_processed['Credit_Need_Numeric'] = df_processed['Credit_Need_Numeric'].fillna(6.0)  # Default to average
        
        X_reg = df_processed[feature_cols].copy()
        X_reg = X_reg.fillna(0)
        
        for col in ['Business Type', 'Industry']:
            X_reg = pd.concat([X_reg, pd.get_dummies(df_processed[col], prefix=col, drop_first=True)], axis=1)
        
        y_reg = df_processed['Credit_Need_Numeric']
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        regression_model = XGBRegressor(n_estimators=100, random_state=42, max_depth=6)
        regression_model.fit(X_train_r, y_train_r)
        
        y_pred_r = regression_model.predict(X_test_r)
        
        reg_metrics = {
            'r2': r2_score(y_test_r, y_pred_r),
            'rmse': np.sqrt(mean_squared_error(y_test_r, y_pred_r)),
            'mae': mean_absolute_error(y_test_r, y_pred_r),
            'mape': np.mean(np.abs((y_test_r - y_pred_r) / y_test_r)) * 100,
            'y_test': y_test_r,
            'y_pred': y_pred_r,
            'feature_importance': pd.DataFrame({
                'feature': X_reg.columns,
                'importance': regression_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # CLUSTERING - WITH NaN HANDLING
        cluster_features = [
            'Digital_Readiness_Score', 'Credit_Health_Score', 'Business_Maturity_Index',
            'Years_Ordinal', 'Revenue_Ordinal', 'Loan_Size_Ordinal'
        ]
        
        X_cluster = df_processed[cluster_features].copy()
        # CRITICAL: Remove any NaN or infinite values
        X_cluster = X_cluster.fillna(0)
        X_cluster = X_cluster.replace([np.inf, -np.inf], 0)
        
        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)
        
        # Additional safety: check for NaN after scaling
        X_cluster_scaled = np.nan_to_num(X_cluster_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        clustering_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = clustering_model.fit_predict(X_cluster_scaled)
        
        df_processed['Cluster'] = clusters
        silhouette = silhouette_score(X_cluster_scaled, clusters)
        
        # ASSOCIATION RULES
        transactions = []
        for idx, row in df_processed.iterrows():
            transaction = []
            if pd.notna(row['Interested Products']):
                products = [p.strip() for p in str(row['Interested Products']).split(',')]
                transaction.extend(products)
            transaction.append(f"Industry_{row['Industry']}")
            transaction.append(f"BizType_{row['Business Type']}")
            transactions.append(transaction)
        
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
        frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            arm_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            arm_rules = arm_rules.sort_values('lift', ascending=False)
        else:
            arm_rules = None
        
        st.success('✅ All models trained successfully!')
        
        return {
            'classification': classification_model,
            'regression': regression_model,
            'clustering': clustering_model,
            'scaler': scaler,
            'class_metrics': class_metrics,
            'reg_metrics': reg_metrics,
            'silhouette': silhouette,
            'arm_rules': arm_rules,
            'df_processed': df_processed,
            'feature_cols_class': X_class.columns.tolist(),
            'feature_cols_reg': X_reg.columns.tolist()
        }

# Load data
df = load_dataset()

if df is None:
    st.error("⚠️ Dataset not found! Please ensure 'msme_refined_dataset.xlsx' is in the same folder.")
    st.stop()

# Train models
if 'models' not in st.session_state:
    with st.spinner('🤖 Training AI models for the first time...'):
        st.session_state.models = train_all_models(df)

models = st.session_state.models

# Main Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", 
    "📊 Executive Dashboard", 
    "🎯 Customer Intelligence",
    "🔮 Predictive Engine",
    "🛡️ Risk Analytics"
])

# TAB 1: OVERVIEW
with tab1:
    st.markdown("## 🎯 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total MSMEs", f"{len(df):,}", "Surveyed")
    
    with col2:
        likely = len(df[df['Likelihood to Use Platform'].isin(['Very Likely', 'Likely'])])
        st.metric("High Adoption", f"{likely:,}", f"{likely/len(df)*100:.1f}%")
    
    with col3:
        credit_mapping = {'<2 lakh': 1, '2–10 lakh': 6, '10–50 lakh': 30, '50 lakh–1 crore': 75, '1 crore+': 150}
        avg_credit = df['Expected Credit Need (12 months)'].map(credit_mapping).mean()
        st.metric("Avg Credit Need", f"₹{avg_credit:.1f}L", "Next 12 Months")
    
    with col4:
        digital_ready = len(df[df['Comfort with App'] == 'Yes'])
        st.metric("Digital Ready", f"{digital_ready/len(df)*100:.0f}%", "App Comfortable")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏢 Industry Distribution")
        industry_counts = df['Industry'].value_counts()
        fig = px.pie(values=industry_counts.values, names=industry_counts.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Adoption Likelihood")
        adoption_counts = df['Likelihood to Use Platform'].value_counts()
        fig = px.bar(x=adoption_counts.index, y=adoption_counts.values, 
                     color=adoption_counts.values, color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title="Likelihood", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: EXECUTIVE DASHBOARD
with tab2:
    st.markdown("## 📊 Business Metrics & Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Classification Model")
        st.metric("Accuracy", f"{models['class_metrics']['accuracy']:.2%}")
        st.metric("Precision", f"{models['class_metrics']['precision']:.2%}")
        st.metric("Recall", f"{models['class_metrics']['recall']:.2%}")
        st.metric("F1-Score", f"{models['class_metrics']['f1']:.2%}")
        st.metric("ROC-AUC", f"{models['class_metrics']['roc_auc']:.2%}")
    
    with col2:
        st.markdown("### Regression Model")
        st.metric("R² Score", f"{models['reg_metrics']['r2']:.2%}")
        st.metric("RMSE", f"{models['reg_metrics']['rmse']:.2f}")
        st.metric("MAE", f"{models['reg_metrics']['mae']:.2f}")
        st.metric("MAPE", f"{models['reg_metrics']['mape']:.1f}%")
    
    with col3:
        st.markdown("### Clustering Model")
        st.metric("Silhouette Score", f"{models['silhouette']:.3f}")
        st.metric("Number of Clusters", "5")
        
        if models['arm_rules'] is not None:
            st.metric("ARM Rules Found", f"{len(models['arm_rules'])}")
        else:
            st.metric("ARM Rules", "N/A")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Classification Feature Importance")
        top_features = models['class_metrics']['feature_importance'].head(10)
        fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Confusion Matrix")
        cm = models['class_metrics']['confusion_matrix']
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                       x=['Not Likely', 'Likely'], y=['Not Likely', 'Likely'],
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 💰 Revenue Projections")
    
    col1, col2, col3 = st.columns(3)
    
    likely_users = len(df[df['Likelihood to Use Platform'].isin(['Very Likely', 'Likely'])])
    avg_ticket = df['Expected Credit Need (12 months)'].map(credit_mapping).mean()
    
    with col1:
        st.markdown("#### Conservative")
        customers = int(likely_users * 0.5)
        revenue = customers * avg_ticket * 0.02 * 12
        st.metric("Customers", f"{customers:,}")
        st.metric("Annual Revenue", f"₹{revenue:.1f}Cr")
    
    with col2:
        st.markdown("#### Base Case")
        customers = int(likely_users * 0.7)
        revenue = customers * avg_ticket * 0.02 * 12
        st.metric("Customers", f"{customers:,}")
        st.metric("Annual Revenue", f"₹{revenue:.1f}Cr")
    
    with col3:
        st.markdown("#### Aggressive")
        customers = int(likely_users * 0.9)
        revenue = customers * avg_ticket * 0.02 * 12
        st.metric("Customers", f"{customers:,}")
        st.metric("Annual Revenue", f"₹{revenue:.1f}Cr")

# TAB 3: CUSTOMER INTELLIGENCE
with tab3:
    st.markdown("## 🎯 Customer Segmentation & Product Affinity")
    
    df_clustered = models['df_processed']
    
    st.markdown("### 👥 Customer Personas")
    
    cluster_profiles = []
    for i in range(5):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        if len(cluster_data) > 0:
            profile = {
                'Persona': f'Segment {i}',
                'Size': len(cluster_data),
                'Percentage': f"{len(cluster_data)/len(df)*100:.1f}%",
                'Top Industry': cluster_data['Industry'].mode()[0] if len(cluster_data) > 0 else 'N/A',
                'Adoption Rate': f"{(cluster_data['Likelihood to Use Platform'].isin(['Very Likely', 'Likely']).sum() / len(cluster_data) * 100):.1f}%"
            }
            cluster_profiles.append(profile)
    
    st.dataframe(pd.DataFrame(cluster_profiles), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Cluster Distribution")
        cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
        fig = px.pie(values=cluster_sizes.values, 
                     names=[f"Segment {i}" for i in cluster_sizes.index],
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Adoption by Segment")
        adoption_by_cluster = []
        for i in range(5):
            cluster_data = df_clustered[df_clustered['Cluster'] == i]
            if len(cluster_data) > 0:
                adoption_rate = (cluster_data['Likelihood to Use Platform'].isin(['Very Likely', 'Likely']).sum() / len(cluster_data)) * 100
                adoption_by_cluster.append({'Segment': f'Segment {i}', 'Adoption Rate': adoption_rate})
        
        adoption_df = pd.DataFrame(adoption_by_cluster)
        fig = px.bar(adoption_df, x='Segment', y='Adoption Rate', 
                     color='Adoption Rate', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    if models['arm_rules'] is not None and len(models['arm_rules']) > 0:
        st.markdown("### 🔗 Top Association Rules (Product Affinity)")
        
        arm_display = models['arm_rules'].head(10).copy()
        arm_display['antecedents'] = arm_display['antecedents'].astype(str)
        arm_display['consequents'] = arm_display['consequents'].astype(str)
        
        display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        st.dataframe(arm_display[display_cols], use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(models['arm_rules'].head(50), x='support', y='confidence', 
                           size='lift', color='lift', hover_data=['antecedents', 'consequents'])
            fig.update_layout(title="Support vs Confidence (sized by Lift)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(models['arm_rules'], x='lift', nbins=30)
            fig.update_layout(title="Distribution of Lift Values")
            st.plotly_chart(fig, use_container_width=True)

# TAB 4: PREDICTIVE ENGINE
with tab4:
    st.markdown("## 🔮 New Customer Prediction")
    
    pred_tab1, pred_tab2 = st.tabs(["📝 Single Prediction", "📊 Bulk Upload"])
    
    with pred_tab1:
        st.markdown("### Enter Customer Details")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                business_type = st.selectbox("Business Type", 
                    ['Sole Proprietorship', 'Partnership', 'Private Limited', 'LLP', 'Others'])
                industry = st.selectbox("Industry",
                    ['Retail', 'Manufacturing', 'Wholesale/Trading', 'Services', 'E-commerce/D2C'])
                years = st.selectbox("Years in Operation",
                    ['<1 year', '1–3 years', '3–5 years', '5–10 years', '10+ years'])
            
            with col2:
                revenue = st.selectbox("Monthly Revenue",
                    ['<50,000', '50,000–2,00,000', '2–10 lakh', '10–50 lakh', '50 lakh+'])
                stability = st.selectbox("Revenue Stability",
                    ['Highly stable (±10%)', 'Moderate variation (±30%)', 'Seasonal', 'Highly unpredictable'])
                gst = st.radio("GST Filing", ['Yes', 'No'])
            
            with col3:
                uses_credit = st.radio("Uses Credit Currently", ['Yes', 'No'])
                default_history = st.selectbox("Default History", ['Never', 'Occasionally', 'Frequently'])
                loan_size = st.selectbox("Typical Loan Size",
                    ['<1 lakh', '1–5 lakh', '5–20 lakh', '20–50 lakh', '50 lakh+'])
            
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
        
        if submitted:
            # Create input
            input_data = pd.DataFrame([{
                'Business Type': business_type,
                'Industry': industry,
                'Years in Operation': years,
                'Monthly Revenue': revenue,
                'Revenue Stability': stability,
                'GST Filing': gst,
                'Uses Credit Currently': uses_credit,
                'Default History': default_history,
                'Typical Loan Size': loan_size,
                'Comfort with App': 'Yes',
                'Willing to Share Data': 'Yes',
                'Digital Transactions %': '50–80%',
                'Maintains Inventory': 'Yes'
            }])
            
            # Engineer features
            input_eng = engineer_features(input_data)
            
            # Prepare for classification
            feature_cols = [
                'Digital_Readiness_Score', 'Credit_Health_Score', 'Business_Maturity_Index',
                'Uses_Credit_Binary', 'GST_Filed_Binary', 'Years_Ordinal', 'Revenue_Ordinal', 'Loan_Size_Ordinal'
            ]
            
            X_input = input_eng[feature_cols]
            for col in ['Business Type', 'Industry']:
                encoded = pd.get_dummies(input_eng[col], prefix=col, drop_first=True)
                X_input = pd.concat([X_input, encoded], axis=1)
            
            # Align columns
            for col in models['feature_cols_class']:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[models['feature_cols_class']]
            
            # Predict
            adoption_prob = models['classification'].predict_proba(X_input)[0][1]
            
            # Regression prediction
            X_input_reg = input_eng[feature_cols]
            for col in ['Business Type', 'Industry']:
                encoded = pd.get_dummies(input_eng[col], prefix=col, drop_first=True)
                X_input_reg = pd.concat([X_input_reg, encoded], axis=1)
            
            for col in models['feature_cols_reg']:
                if col not in X_input_reg.columns:
                    X_input_reg[col] = 0
            X_input_reg = X_input_reg[models['feature_cols_reg']]
            
            credit_need = models['regression'].predict(X_input_reg)[0]
            
            # Clustering
            cluster_features = [
                'Digital_Readiness_Score', 'Credit_Health_Score', 'Business_Maturity_Index',
                'Years_Ordinal', 'Revenue_Ordinal', 'Loan_Size_Ordinal'
            ]
            X_cluster = input_eng[cluster_features]
            X_cluster_scaled = models['scaler'].transform(X_cluster)
            persona = models['clustering'].predict(X_cluster_scaled)[0]
            
            # Risk score
            risk_score = (
                adoption_prob * 0.4 +
                (input_eng['Credit_Health_Score'].values[0] / 100) * 0.3 +
                (input_eng['Digital_Readiness_Score'].values[0] / 100) * 0.2 +
                (input_eng['Business_Maturity_Index'].values[0] / 100) * 0.1
            ) * 100
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                color = "🟢" if adoption_prob > 0.7 else "🟡" if adoption_prob > 0.5 else "🔴"
                st.metric("Adoption Probability", f"{adoption_prob:.1%}", delta=color)
            
            with col2:
                st.metric("Predicted Credit Need", f"₹{credit_need:.1f}L")
            
            with col3:
                st.metric("Customer Segment", f"Segment {persona}")
            
            with col4:
                st.metric("Risk Score", f"{risk_score:.0f}/100")
            
            # Recommendations
            st.markdown("---")
            
            if adoption_prob > 0.7:
                st.success(f"""
                **🟢 HIGH PROBABILITY CUSTOMER**
                - Adoption Score: {adoption_prob:.1%}
                - Action: Priority Sales Call
                - Expected Conversion: 70-85%
                - Pricing: 1.5-1.8% monthly
                """)
            elif adoption_prob > 0.5:
                st.warning(f"""
                **🟡 MEDIUM PROBABILITY CUSTOMER**
                - Adoption Score: {adoption_prob:.1%}
                - Action: Nurture with Demo
                - Expected Conversion: 40-60%
                - Pricing: 1.8-2.2% monthly
                """)
            else:
                st.error(f"""
                **🔴 LOW PROBABILITY CUSTOMER**
                - Adoption Score: {adoption_prob:.1%}
                - Action: Educational Content
                - Expected Conversion: 15-30%
                - Pricing: 2.5-3.0% monthly
                """)
    
    with pred_tab2:
        st.markdown("### 📊 Bulk Customer Scoring")
        
        st.markdown("#### Download Template")
        sample = pd.DataFrame([{
            'Business Type': 'Sole Proprietorship',
            'Industry': 'Retail',
            'Years in Operation': '1–3 years',
            'Monthly Revenue': '50,000–2,00,000',
            'Revenue Stability': 'Moderate variation (±30%)',
            'GST Filing': 'Yes',
            'Uses Credit Currently': 'Yes',
            'Default History': 'Never',
            'Typical Loan Size': '1–5 lakh'
        }])
        
        csv = sample.to_csv(index=False)
        st.download_button("📥 Download CSV Template", csv, "template.csv", "text/csv")
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            bulk_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(bulk_df)} customers")
            
            if st.button("🔮 Score All Customers", use_container_width=True):
                with st.spinner("Processing..."):
                    # Add missing columns
                    bulk_df['Comfort with App'] = 'Yes'
                    bulk_df['Willing to Share Data'] = 'Yes'
                    bulk_df['Digital Transactions %'] = '50–80%'
                    bulk_df['Maintains Inventory'] = 'Yes'
                    
                    bulk_eng = engineer_features(bulk_df)
                    
                    # Classification
                    feature_cols = [
                        'Digital_Readiness_Score', 'Credit_Health_Score', 'Business_Maturity_Index',
                        'Uses_Credit_Binary', 'GST_Filed_Binary', 'Years_Ordinal', 'Revenue_Ordinal', 'Loan_Size_Ordinal'
                    ]
                    
                    X_bulk = bulk_eng[feature_cols]
                    for col in ['Business Type', 'Industry']:
                        X_bulk = pd.concat([X_bulk, pd.get_dummies(bulk_eng[col], prefix=col, drop_first=True)], axis=1)
                    
                    for col in models['feature_cols_class']:
                        if col not in X_bulk.columns:
                            X_bulk[col] = 0
                    X_bulk = X_bulk[models['feature_cols_class']]
                    
                    adoption_probs = models['classification'].predict_proba(X_bulk)[:, 1]
                    
                    bulk_df['Adoption_Probability'] = adoption_probs
                    bulk_df['Priority'] = pd.cut(adoption_probs, bins=[0, 0.5, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
                    
                    st.markdown("### 📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        high = len(bulk_df[bulk_df['Priority'] == 'High'])
                        st.metric("High Priority", high, f"{high/len(bulk_df)*100:.1f}%")
                    with col2:
                        medium = len(bulk_df[bulk_df['Priority'] == 'Medium'])
                        st.metric("Medium Priority", medium, f"{medium/len(bulk_df)*100:.1f}%")
                    with col3:
                        st.metric("Avg Probability", f"{adoption_probs.mean():.1%}")
                    
                    st.dataframe(bulk_df[['Business Type', 'Industry', 'Adoption_Probability', 'Priority']], 
                               use_container_width=True, hide_index=True)
                    
                    csv_result = bulk_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv_result, "scored_customers.csv", "text/csv")

# TAB 5: RISK ANALYTICS
with tab5:
    st.markdown("## 🛡️ Portfolio Risk Management")
    
    df_risk = models['df_processed'].copy()
    df_risk['Risk_Score'] = (
        df_risk['Credit_Health_Score'] * 0.4 +
        df_risk['Digital_Readiness_Score'] * 0.3 +
        df_risk['Business_Maturity_Index'] * 0.3
    )
    
    df_risk['Risk_Category'] = pd.cut(df_risk['Risk_Score'], 
                                      bins=[0, 40, 60, 80, 100],
                                      labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        low_risk = len(df_risk[df_risk['Risk_Category'].isin(['Very Low Risk', 'Low Risk'])])
        st.metric("Low Risk", f"{low_risk:,}", f"{low_risk/len(df)*100:.1f}%")
    
    with col2:
        medium_risk = len(df_risk[df_risk['Risk_Category'] == 'Medium Risk'])
        st.metric("Medium Risk", f"{medium_risk:,}", f"{medium_risk/len(df)*100:.1f}%")
    
    with col3:
        high_risk = len(df_risk[df_risk['Risk_Category'] == 'High Risk'])
        st.metric("High Risk", f"{high_risk:,}", f"{high_risk/len(df)*100:.1f}%")
    
    with col4:
        never_default = len(df[df['Default History'] == 'Never'])
        st.metric("Clean History", f"{never_default/len(df)*100:.0f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Distribution")
        risk_counts = df_risk['Risk_Category'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    color_discrete_map={
                        'Very Low Risk': '#2ecc71',
                        'Low Risk': '#27ae60',
                        'Medium Risk': '#f39c12',
                        'High Risk': '#e74c3c'
                    }, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Default History")
        default_counts = df['Default History'].value_counts()
        fig = px.bar(x=default_counts.index, y=default_counts.values,
                    color=default_counts.index,
                    color_discrete_map={
                        'Never': '#2ecc71',
                        'Occasionally': '#f39c12',
                        'Frequently': '#e74c3c'
                    })
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 💰 Risk-Based Pricing")
    
    def get_pricing(score):
        if score >= 75: return 1.65
        elif score >= 60: return 2.0
        else: return 2.75
    
    df_risk['Suggested_Rate'] = df_risk['Risk_Score'].apply(get_pricing)
    
    avg_pricing = df_risk.groupby('Risk_Category')['Suggested_Rate'].mean().sort_values(ascending=False)
    
    fig = px.bar(x=avg_pricing.index, y=avg_pricing.values,
                 color=avg_pricing.values, color_continuous_scale='RdYlGn_r')
    fig.update_layout(xaxis_title="Risk Category", yaxis_title="Monthly Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>ReadyFunds Analytics Dashboard</strong> | Built with Streamlit</p>
    <p>Data-Driven Decisions for Sustainable Growth</p>
</div>
""", unsafe_allow_html=True)
