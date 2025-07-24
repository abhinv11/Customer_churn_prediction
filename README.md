# Customer Churn Prediction

A comprehensive machine learning project for predicting customer churn using multiple algorithms and advanced data preprocessing techniques.

## 📋 Project Overview

This project analyzes customer data to predict whether a customer will churn (leave the service) or remain with the company. The project implements a complete machine learning pipeline including data exploration, preprocessing, feature engineering, model comparison, and hyperparameter tuning.

## 🎯 Objective

To build an accurate machine learning model that can predict customer churn, helping businesses identify at-risk customers and implement retention strategies.

## 📊 Dataset Features

The dataset includes the following customer attributes:

- **customer_id**: Unique identifier for each customer
- **credit_score**: Customer's credit score
- **age**: Customer's age
- **tenure**: Number of years as a customer
- **acc_balance**: Account balance
- **prod_count**: Number of products owned
- **has_card**: Whether customer has a credit card (binary)
- **is_active**: Whether customer is active (binary)
- **estimated_salary**: Customer's estimated salary
- **country**: Customer's country
- **gender**: Customer's gender
- **exit_status**: Target variable (1 = churned, 0 = retained)

## 🔄 Project Workflow

### 1. Data Exploration and Analysis
- **Data Type Identification**: Analyzed numerical and categorical features
- **Descriptive Statistics**: Generated comprehensive statistics for numerical columns
- **Missing Value Analysis**: Identified and handled missing data strategically
- **Duplicate Detection**: Ensured data integrity by removing duplicates
- **Outlier Analysis**: Used IQR method to detect outliers while preserving valuable data

### 2. Data Preprocessing
- **Missing Value Handling**:
  - Credit score: Filled with median
  - Account balance: Filled with 0 (no balance assumption)
  - Country: Filled with mode
  - Product count: Filled with median
- **Feature Encoding**: 
  - Label encoding for categorical variables (country, gender)
  - Binary features kept as-is
- **Feature Scaling**: StandardScaler applied to numerical features

### 3. Feature Engineering
Selected features for modeling:
- `credit_score`, `age`, `tenure`, `acc_balance`, `prod_count`
- `has_card`, `is_active`, `estimated_salary`
- `country_encoded`, `gender_encoded`

### 4. Model Development and Comparison

#### Models Implemented:
1. **Logistic Regression** (with class balancing)
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**
4. **K-Nearest Neighbors**
5. **Decision Tree Classifier**
6. **AdaBoost Classifier**
7. **Naive Bayes (Gaussian)**

#### Performance Metrics:
- Accuracy
- Precision
- Recall
- F1-Score (primary metric for model selection)

### 5. Hyperparameter Tuning
- Applied GridSearchCV to top 3 performing models
- Used Stratified K-Fold cross-validation (3 folds)
- Optimized based on F1-score to handle class imbalance

### 6. Final Model Selection
- **Selected Model**: Gradient Boosting Classifier
- **Final Parameters**:
  - n_estimators: 200
  - learning_rate: 0.1
  - max_depth: 5
  - random_state: 42

## 📈 Key Insights

### Data Insights:
- Customer churn rate analysis revealed important patterns
- Age distribution varies between churned and retained customers
- Customers with 4 products show higher churn rates
- Credit score distribution differs between customer segments

### Feature Importance:
The model identified the most influential features for churn prediction (top features from final model analysis).

## 📁 Project Structure

```
customer_churn_prediction/
├── customer_churn_prediction.ipynb  # Main Jupyter notebook
├── README.md                        # Project documentation
└── submission.csv                   # Final predictions (generated)
```

## 🚀 Getting Started

### Prerequisites
```python
numpy
pandas
matplotlib
seaborn
scikit-learn
warnings
```

### Installation
1. Clone or download the project files
2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open `customer_churn_prediction.ipynb` in Jupyter Notebook or VS Code

### Running the Project
1. Ensure your data files are in the correct path:
   - Training data: `/kaggle/input/mlp-term-2-2025-kaggle-assignment-2/train.csv`
   - Test data: `/kaggle/input/mlp-term-2-2025-kaggle-assignment-2/test.csv`
2. Run all cells in the notebook sequentially
3. The final predictions will be saved as `submission.csv`

## 📊 Results

The project delivers:
- Comprehensive data analysis and visualizations
- Comparison of 7 different machine learning algorithms
- Hyperparameter-tuned models for optimal performance
- Final predictions with feature importance analysis
- A submission file ready for Kaggle or production use

## 🔧 Model Performance

The final Gradient Boosting model achieved excellent performance metrics through systematic comparison and hyperparameter tuning. The model successfully balances precision and recall while maintaining high overall accuracy.

## 📝 Notes

- The project handles missing values intelligently based on domain knowledge
- Outliers are retained as they represent legitimate customer variations
- Class imbalance is addressed through appropriate model selection and metrics
- Feature scaling ensures all numerical features contribute equally to model training

## 🤝 Contributing

Feel free to fork this project and submit improvements. Areas for enhancement:
- Additional feature engineering techniques
- More advanced ensemble methods
- Deep learning approaches
- Real-time prediction implementation

## 📜 License

This project is available for educational and research purposes.

---

## 👨‍💻 Author

**Abhinav Jaswal**
- GitHub: [@abhinv11](https://github.com/abhinv11)
- LinkedIn: [Connect with me](https://linkedin.com/in/abhinavjaswal001)

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- Scikit-learn community for excellent documentation
- Open source contributors for the libraries used