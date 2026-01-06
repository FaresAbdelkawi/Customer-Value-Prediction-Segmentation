# Customer Value Prediction & Segmentation

## 📌 Project Overview
In the competitive landscape of e-commerce, understanding customer behavior is pivotal for sustainable growth. This project focuses on **Customer Lifetime Value (CLV) prediction** and **Customer Segmentation** to help businesses identify their most valuable customers and tailor marketing strategies accordingly.

By leveraging historical transaction data, we perform **RFM (Recency, Frequency, Monetary) analysis**, cluster customers into distinct segments, and utilize supervised machine learning (including Deep Learning) to predict the future value of customers.

## 🚀 Key Features
* **Data Preprocessing**: Extensive cleaning of transactional retail data, handling missing values, and filtering outliers.
* **Feature Engineering**: Creation of **RFM** features to quantify customer behavior.
* **Customer Segmentation**: Unsupervised learning to group customers based on purchasing patterns.
* **Value Prediction**: Implementation of multiple regression models (Random Forest, XGBoost) and Deep Learning (TensorFlow/Keras) to forecast customer value.
* **Model Optimization**: Hyperparameter tuning to minimize prediction error (RMSE).

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn`, `xgboost`
* **Deep Learning**: `tensorflow`, `keras`

## 📂 Dataset
The project likely uses the **Online Retail** dataset (or similar transactional data), containing fields such as:
* `InvoiceNo`: Unique transaction ID.
* `StockCode`: Product code.
* `Description`: Product name.
* `Quantity`: Number of products purchased.
* `InvoiceDate`: Date of transaction.
* `UnitPrice`: Price per unit.
* `CustomerID`: Unique customer identifier.
* `Country`: Customer's location.

## 📊 Methodology

### 1. Data Cleaning & EDA
* Removed canceled transactions (invoices starting with 'C').
* Handled missing `CustomerID` values.
* Visualized sales trends and country-wise distribution.

### 2. Feature Engineering (RFM)
We calculated the following for each customer:
* **Recency**: Days since last purchase.
* **Frequency**: Total number of purchases.
* **Monetary**: Total revenue generated.

### 3. Segmentation (Unsupervised Learning)
* Applied clustering algorithms (e.g., K-Means) on RFM scores.
* Grouped customers into segments such as "Champions," "At-Risk," and "New Customers."

### 4. Predictive Modeling (Supervised Learning)
* **Target Variable**: Total spend/value (derived from Quantity * UnitPrice).
* **Models Trained**:
    * **Random Forest Regressor**: Tuned for `n_estimators`, `max_depth`, etc.
    * **XGBoost Regressor**: Optimized for `learning_rate`, `subsample`, `max_depth`.
    * **Neural Networks**: Implemented using TensorFlow/Keras for complex pattern recognition.

## 📉 Results & Performance
* **Best Model**: The XGBoost/Random Forest model achieved the lowest error rates on the validation set.
* *Specific hyperparameters found to be optimal include:*
    * XGBoost: `learning_rate: 0.05`, `max_depth: 3`, `n_estimators: 200`
    * Random Forest: `n_estimators: 300`, `max_depth: 20`

## 💻 How to Run
1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/yourusername/customer-value-prediction.git](https://github.com/yourusername/customer-value-prediction.git)
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
    ```
3.  **Run the Notebook**:
    Open `Customer Value Prediction & Segmentation.ipynb` in Jupyter Notebook or Google Colab to execute the analysis.

## 🔮 Future Improvements
* Integrate **deployment** using Flask/Streamlit to create a web app for real-time predictions.
* Experiment with **Cohort Analysis** to track customer retention over time.
* Add **Market Basket Analysis** to recommend products to specific segments.

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
