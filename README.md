# 📊 Telecom Customer Churn Prediction System

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Powered Customer Retention Prediction System</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; line-height: 1.6; margin: 40px; color: #222; }
    h1, h2, h3 { color: #0b5394; }
    h1 { border-bottom: 3px solid #0b5394; padding-bottom: 5px; }
    h2 { margin-top: 40px; }
    ul { margin-left: 20px; }
    code { background: #f4f4f4; padding: 2px 5px; border-radius: 4px; }
    .section { margin-bottom: 40px; }
    .highlight { background: #eef4ff; padding: 15px; border-left: 5px solid #0b5394; }
  </style>
</head>
<body>

<h1>AI Powered Customer Retention Prediction System</h1>

<div class="section">
  <h2>Abstract</h2>
  <p>
    Customer churn is a major challenge in the telecom industry, directly affecting revenue and customer lifetime value. This project focuses on analysing telecom customer data to identify behavioural patterns and key factors influencing customer churn. Exploratory Data Analysis (EDA) and multiple machine learning models were applied to predict churn effectively. The study highlights that customers with shorter tenure, month-to-month contracts, higher monthly charges, and manual payment methods are more likely to churn, while long-term contracts and bundled services improve retention.
  </p>
</div>

<div class="section">
  <h2>Introduction</h2>
  <p>
    Retaining existing customers is more cost-effective than acquiring new ones, making churn prediction a critical business objective. This project develops a machine learning-based system to predict telecom customer churn using demographic, service, contract, tenure, and billing information. Multiple classification algorithms were trained and evaluated to identify the most reliable churn prediction model.
  </p>
</div>

<div class="section">
  <h2>Requirements</h2>
  <ul>
    <li>Python</li>
    <li>NumPy, Pandas</li>
    <li>Matplotlib, Seaborn</li>
    <li>Scikit-learn</li>
    <li>Imbalanced-learn</li>
    <li>Flask</li>
  </ul>
</div>

<div class="section">
  <h2>Data Visualization</h2>
  <p>
    Data visualization was performed using Matplotlib and Seaborn to understand customer behaviour and churn patterns. Various plots such as bar charts, count plots, histograms, and box plots were used.
  </p>
</div>

<div class="section">
  <h2>Feature Engineering</h2>
  <ul>
    <li>Handling missing values</li>
    <li>Data separation (numerical & categorical)</li>
    <li>Variable transformation</li>
    <li>Outlier handling</li>
    <li>Categorical encoding</li>
  </ul>
</div>

<div class="section">
  <h2>Feature Selection</h2>
  <p>
    Filter-based feature selection techniques were used to remove irrelevant and low-variance features. Hypothesis testing and variance threshold methods were applied to improve model stability and performance.
  </p>
</div>

<div class="section">
  <h2>Data Balancing</h2>
  <p>
    The dataset exhibited class imbalance. To address this, SMOTE (Synthetic Minority Over-sampling Technique) was applied to generate synthetic minority class samples and improve model generalization.
  </p>
</div>

<div class="section">
  <h2>Feature Scaling</h2>
  <p>
    Standard Scaling (Z-score normalization) was applied to numerical features to ensure all variables contribute equally during model training.
  </p>
</div>

<div class="section">
  <h2>Model Training</h2>
  <p>
    The problem was framed as a binary classification task. Multiple models including Logistic Regression, KNN, Decision Tree, SVM, and Naive Bayes were trained and evaluated using the same training data.
  </p>
</div>

<div class="section">
  <h2>Model Evaluation</h2>
  <p>
    Models were evaluated using accuracy, ROC curve, and AUC score. ROC–AUC was chosen as the primary metric due to its robustness to class imbalance.
  </p>
</div>

<div class="section">
  <h2>Best Model</h2>
  <div class="highlight">
    <p>
      Naive Bayes achieved the highest ROC–AUC score (~0.8366) among all trained models and was selected as the final model.
    </p>
  </div>
</div>

<div class="section">
  <h2>Hyperparameter Tuning</h2>
  <p>
    GridSearchCV was used to identify the optimal hyperparameter configuration using cross-validation, improving model performance and generalization.
  </p>
</div>

<div class="section">
  <h2>Deployment</h2>
  <p>
    A Flask-based web application was developed to allow users to input customer details and receive real-time churn predictions along with probability scores.
  </p>
</div>

<div class="section">
  <h2>Conclusion</h2>
  <p>
    This project demonstrates how machine learning can be effectively used to predict customer churn and support proactive retention strategies. The end-to-end pipeline—from data preprocessing to deployment—highlights the practical application of data science in real-world business scenarios.
  </p>
</div>

<div class="section">
  <h2>Future Enhancements</h2>
  <ul>
    <li>Integration of deep learning models (ANN, LSTM)</li>
    <li>Real-time churn prediction pipelines</li>
    <li>Interactive dashboards for churn monitoring</li>
    <li>Scalable deployment architecture</li>
  </ul>
</div>

<div class="section">
  <h2>References</h2>
  <ul>
    <li>https://www.kaggle.com/datasets/blastchar/telco-customer-churn</li>
    <li>https://scikit-learn.org/</li>
    <li>https://pandas.pydata.org/</li>
    <li>https://flask.palletsprojects.com/</li>
  </ul>
</div>

</body>
</html>

