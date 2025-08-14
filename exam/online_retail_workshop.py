"""
Online Retail Workshop: Demonstrate EDA, preprocessing, feature engineering (RFM), supervised & unsupervised learning,
model pipeline, anomaly detection, and association rules on UCI Online Retail dataset.

Usage:
    pip install pandas numpy matplotlib seaborn scikit-learn mlxtend flask
    # Run all steps and skip API
    python online_retail_workshop.py
    # Start Flask API server (after initial run to train pipeline)
    python online_retail_workshop.py --serve
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
from flask import Flask, request, jsonify
import argparse

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, parse_dates=['InvoiceDate'])

def perform_eda(df: pd.DataFrame) -> None:
    print('Data shape:', df.shape)
    print(df.info())
    print(df.describe())
    print('Missing by column:\n', df.isnull().sum())

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
    df = df.dropna(subset=['CustomerID'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    now = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (now - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalAmount':'Monetary'})
    print(rfm.head())
    return rfm

def train_supervised_model(rfm: pd.DataFrame) -> Pipeline:
    rfm['HighValue'] = (rfm['Monetary'] > rfm['Monetary'].median()).astype(int)
    X = rfm[['Recency','Frequency','Monetary']]
    y = rfm['HighValue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()),('clf', LogisticRegression())])
    pipe.fit(X_train, y_train)
    print('Classification score:', pipe.score(X_test, y_test))
    return pipe

def train_unsupervised_model(rfm: pd.DataFrame) -> None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
    kmeans = KMeans(n_clusters=4, random_state=42).fit(X_scaled)
    rfm['Cluster'] = kmeans.labels_
    sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm)
    plt.title('RFM Clusters')
    plt.show()

def detect_anomalies(df: pd.DataFrame) -> None:
    iso = IsolationForest(contamination=0.01, random_state=42)
    outliers = iso.fit_predict(df[['TotalAmount']])
    print('Number of outliers detected:', (outliers == -1).sum())

def association_rules_step(df: pd.DataFrame) -> None:
    basket = df[df['Country']=='United Kingdom']
    basket = basket.groupby(['InvoiceNo','StockCode'])['Quantity'].sum().unstack().fillna(0)
    basket = (basket > 0).astype(int)
    freq = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(freq, metric='lift', min_threshold=1)
    print(rules[['antecedents','consequents','support','confidence','lift']].head())

def create_app(model: Pipeline) -> Flask:
    app = Flask(__name__)
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        df_in = pd.DataFrame(data)
        preds = model.predict(df_in)
        return jsonify(predictions=preds.tolist())
    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', help='Start Flask API after processing')
    args = parser.parse_args()
    df = load_data('data/Online_Retail.csv')
    perform_eda(df)
    df = preprocess_data(df)
    rfm = compute_rfm(df)
    model = train_supervised_model(rfm)
    train_unsupervised_model(rfm)
    detect_anomalies(df)
    association_rules_step(df)
    if args.serve:
        app = create_app(model)
        app.run(debug=True)
    else:
        print("Workshop steps executed. Use --serve to start API server.")

if __name__ == '__main__':
    main()
