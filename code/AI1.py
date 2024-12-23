import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the Dataset
# UCI Online Retail Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
data = pd.read_excel(url)

# Preview the dataset
print("Dataset Overview:")
print(data.head())

# Step 2: Data Cleaning and Preprocessing
# Remove missing values
data = data.dropna()

# Filter out negative or zero quantities
data = data[data['Quantity'] > 0]
data = data[data['UnitPrice'] > 0]

# Create a TotalPrice column
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Step 3: Customer Segmentation using RFM Analysis
# Calculate Recency, Frequency, and Monetary value
import datetime as dt

snapshot_date = data['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Log-transform the data to reduce skewness
rfm[['Recency', 'Frequency', 'Monetary']] = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

# Standardize the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Visualize the segments
sns.pairplot(rfm, hue='Segment', diag_kind='kde')
plt.show()

# Step 4: Dynamic Pricing Model
# Define a function to calculate personalized prices
def personalized_price(base_price, segment, elasticity_factor=0.1):
    """Adjust the price based on the customer segment."""
    segment_multipliers = {
        0: 1.2,  # High-value customers
        1: 0.9,  # Price-sensitive customers
        2: 1.0,  # Average customers
        3: 0.8   # Bargain hunters
    }
    adjusted_price = base_price * segment_multipliers[segment]
    return round(adjusted_price * (1 + np.random.uniform(-elasticity_factor, elasticity_factor)), 2)

# Add personalized pricing for products in the dataset
sample_products = data[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
sample_products['PersonalizedPrice'] = sample_products['UnitPrice']

# Simulate personalized pricing
personalized_prices = []
for _, row in sample_products.iterrows():
    for _, customer in rfm.iterrows():
        price = personalized_price(row['UnitPrice'], customer['Segment'])
        personalized_prices.append({
            'CustomerID': customer['CustomerID'],
            'StockCode': row['StockCode'],
            'PersonalizedPrice': price
        })

pricing_df = pd.DataFrame(personalized_prices)

# Step 5: Evaluate Pricing Impact
# Example: Average price offered to each segment
pricing_summary = pricing_df.merge(rfm[['CustomerID', 'Segment']], on='CustomerID')
average_prices = pricing_summary.groupby('Segment')['PersonalizedPrice'].mean()

print("Average Personalized Prices by Segment:")
print(average_prices)

# Example: Visualizing price distribution by segment
sns.boxplot(data=pricing_summary, x='Segment', y='PersonalizedPrice')
plt.title("Price Distribution by Customer Segment")
plt.show()