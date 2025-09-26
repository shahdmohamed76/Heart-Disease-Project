\# Heart Disease Prediction Project 

\#\# 📌 Overview  
This project builds a \*\*complete machine learning pipeline\*\* to analyze and predict the risk of heart disease using the \*\*UCI Heart Disease dataset\*\*.    
The workflow includes:  
\- Data preprocessing & cleaning  
\- Exploratory Data Analysis (EDA)  
\- Dimensionality Reduction (PCA)  
\- Feature Selection  
\- Supervised Learning (classification models)  
\- Unsupervised Learning (clustering)  
\- Hyperparameter Tuning  
\- Model export (saving \`.pkl\` files)

\---

\#\# ⚙️ Steps Performed

\#\#\# 1\. Data Preprocessing & Cleaning  
\- Handled missing values (median for numeric, mode for categorical).  
\- Converted the target column into binary (0 \= No Disease, 1 \= Disease).  
\- Saved a cleaned dataset (\`cleaned\_heart.csv\`).

\#\#\# 2\. Exploratory Data Analysis (EDA)  
\- Correlation heatmap between features.  
\- Distribution of target classes (with/without heart disease).

\#\#\# 3\. Dimensionality Reduction (PCA)  
\- Applied PCA to reduce dimensions while preserving 95% variance.  
\- Plotted explained variance ratio.

\#\#\# 4\. Feature Selection  
\- Feature importance using Random Forest.  
\- Recursive Feature Elimination (RFE) with Logistic Regression.  
\- Selected 8 most important features.

\#\#\# 5\. Supervised Learning (Classification)  
Trained and compared the following models:  
\- Logistic Regression    
\- Decision Tree    
\- Random Forest    
\- Support Vector Machine (SVM)  

Evaluation metrics used:  
\- Accuracy    
\- Precision    
\- Recall    
\- F1-score    
\- ROC-AUC  

Also plotted \*\*confusion matrices\*\* and a \*\*model comparison chart\*\*.

\#\#\# 6\. Unsupervised Learning (Clustering)  
\- \*\*K-Means\*\* clustering (Elbow method to find optimal K).    
\- \*\*Hierarchical clustering\*\* (dendrogram).  

\#\#\# 7\. Hyperparameter Tuning  
\- Applied \*\*GridSearchCV\*\* on Random Forest with different parameters:  
  \- \`n\_estimators \= \[100, 200, 300\]\`  
  \- \`max\_depth \= \[None, 10, 20\]\`  
  \- \`min\_samples\_split \= \[2, 5, 10\]\`

\- Found the \*\*best model parameters\*\* and achieved the best accuracy score.

\#\#\# 8\. Model Export  
\- Saved the trained Random Forest model as \`heart\_disease\_model.pkl\`.  
\- Saved the StandardScaler as \`scaler.pkl\`.  
\- Combined them into a full pipeline (\`final\_pipeline.pkl\`) for reproducibility.

\---

\#\# 🚀 How to Run the Project  
Clone the repository or extract the \`.zip/.rar\` file.

pip install -r requirements.txt
jupyter notebook
