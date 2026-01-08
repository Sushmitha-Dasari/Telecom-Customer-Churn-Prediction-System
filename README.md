# 📊 Telecom Customer Churn Prediction System

## 📌 Project Introduction
This project analyses historical telecom customer data to understand customer behaviour and predict churn.  
The dataset includes customer demographics, service subscriptions, billing details, tenure, and churn status.  
Using Exploratory Data Analysis (EDA) with Matplotlib visualizations, the study identifies patterns and trends that influence customer churn.  
The insights gained help the organization make data-driven decisions to improve customer retention and reduce churn.

---

## 📂 Data Collection and Scope
The dataset contains historical telecom customer information where each record represents an individual customer.  
It captures demographic details, service subscriptions, billing information, tenure, and churn status, enabling comprehensive exploratory analysis to identify factors influencing customer retention and churn.

---

## 🗂 Dataset Structure

### 1️⃣ Customer Identification
- Customer ID

### 2️⃣ Demographic Attributes
- Gender  
- Senior Citizen Status  
- Partner  
- Dependents  

### 3️⃣ Service Subscription Attributes
- SIM / Network Provider  
- Phone Service  
- Multiple Lines  
- Internet Service Type (DSL, Fiber Optic, No Internet)  
- Streaming TV  
- Streaming Movies  

### 4️⃣ Value-Added Services
- Online Security  
- Online Backup  
- Device Protection  
- Technical Support  

### 5️⃣ Contract and Tenure Attributes
- Contract Type (Month-to-Month, One-Year, Two-Year)  
- Tenure (Number of months with the service)

### 6️⃣ Billing and Payment Attributes
- Monthly Charges  
- Total Charges  
- Payment Method  
- Paperless Billing Status  

### 7️⃣ Time-Based Attributes
- Customer Joining Year  

### 8️⃣ Target Variable
- Churn Status (Yes / No)

---

## 🔍 Analytical Approach and Visualization Strategy
The project uses Exploratory Data Analysis (EDA) after basic data cleaning and validation.  
Customer demographics, service usage, tenure, contract type, and billing behaviour are analysed to identify churn patterns.

Matplotlib visualizations such as bar charts, histograms, and pie charts are used to:
- Compare customer segments
- Observe tenure trends
- Analyse service adoption
- Identify high-risk churn groups

---

## 📈 Visualizations and Insights

### 1. Partner Distribution
Shows customer distribution based on partner status.
- Helps understand correlation between relationship status and service presence.

---

### 2. Churn Distribution
Displays overall churn vs retained customers.
- Highlights customer retention vs customer loss.
- Reducing churn improves revenue stability.

---

### 3. Churn Possibility per SIM Provider
Compares churn across SIM providers:
- Jio
- Airtel
- VI
- BSNL

**Insight:**  
Some SIM providers show higher churn, indicating potential pricing or service quality issues.

---

### 4. Year-wise Customer Distribution
Shows SIM acquisition trends by year.
- Helps understand growth, stability, or decline in customer onboarding.

---

### 5. Total Charges by SIM Provider
Displays cumulative charges by SIM provider.
- Higher total charges indicate stronger loyalty and longer tenure.

---

### 6. Average Monthly Charges by Churn
Compares billing between churned and retained customers.
- Churned customers generally have higher monthly charges.
- Pricing significantly impacts churn behaviour.

---

### 7. Average Tenure by Churn
Compares customer tenure based on churn status.
- Retained customers have significantly higher tenure.
- Churn probability decreases as tenure increases.

---

### 8. Gender Distribution Among Senior Citizens
Shows male vs female senior customers.
- Senior customers often show higher churn sensitivity.

---

### 9. Paperless Billing Distribution
Displays customers with and without paperless billing.
- Digital billing adoption is high.
- Paperless billing combined with auto-payment reduces churn risk.

---

### 10. Streaming Movies Subscription
Shows adoption of streaming movie services.
- Many customers have not subscribed.
- Internet availability strongly affects adoption.

---

### 11. Streaming TV Subscription
Displays Streaming TV usage.
- Higher adoption among customers with stable internet.
- Streaming subscribers show higher engagement.

---

### 12. Tenure Distribution
Shows customer count by tenure range.
- High churn occurs in early tenure stages.
- Long-tenure customers are more loyal.

---

### 13. Contract Distribution
Shows distribution across contract types.
- Month-to-month contracts dominate.
- Long-term contracts show lower churn risk.

---

### 14. Customers by Dependents
Shows dependent vs non-dependent customers.
- Customers without dependents are more price-sensitive.

---

### 15. Customers by Multiple Lines
Displays single-line, multiple-line, and no phone service users.
- Multiple-line customers show higher engagement.

---

### 16. Gender Distribution
Shows male vs female customers.
- Helps understand overall customer composition.

---

### 17. Phone Service Distribution
Displays phone service adoption.
- Phone service is a core offering.
- Customers without phone service may churn more.

---

### 18. Senior Citizen Distribution
Shows senior vs non-senior customers.
- Non-senior customers form the majority.
- Seniors often need tailored support.

---

### 19. SIM Services Usage
Shows usage of phone, internet, and streaming services.
- Value-added services have lower adoption.
- Bundled services reduce churn.

---

### 20. SIMs Taken by Year
Displays year-wise SIM activation trends.
- Peaks indicate strong promotions or market expansion.

---

### 21. Share of SIM Providers
Shows customer share across:
- Jio
- Airtel
- VI
- BSNL

**Insight:**  
Few providers dominate the market, indicating strong brand presence.

---

### 22. Payment Method Distribution
Displays preferred payment methods:
- Electronic Check
- Credit Card (Auto)
- Bank Transfer (Auto)
- Mailed Check

**Insight:**  
Manual payment methods are linked with higher churn, while auto-pay users show stability.

---

## ✅ Conclusion
The analysis highlights that **tenure, contract type, payment method, service usage, and billing behaviour** are key drivers of customer churn.

Customers with:
- Short tenure
- Month-to-month contracts
- Manual payment methods
- Limited service subscriptions  

are more likely to churn.

Whereas customers with:
- Long-term contracts
- Multiple services
- Automated payments
- Digital billing  

show stronger retention.

These insights support **data-driven retention strategies** and form a strong foundation for building predictive churn models.

---

## 🛠 Tools & Technologies
- Python
- Pandas
- Matplotlib
- Exploratory Data Analysis (EDA)

---

