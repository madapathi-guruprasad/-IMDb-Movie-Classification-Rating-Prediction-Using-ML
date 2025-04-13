# IMDb ML Project - KNN Classification & XGBoost Regression with Visuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("C:/Users/madap/Desktop/IMBD.csv")

# Clean and preprocess
df = df[df['rating'].notna() & df['certificate'].notna()]
df['votes'] = df['votes'].str.replace(',', '').astype(float)
df['duration'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['year'] = df['year'].str.extract(r'(\d{4})').astype(float)
df['main_genre'] = df['genre'].str.split(',').str[0]
df = df.dropna(subset=['year', 'duration', 'votes', 'rating'])

# Encoding
label_enc_cert = LabelEncoder()
df['certificate_encoded'] = label_enc_cert.fit_transform(df['certificate'])
label_enc_genre = LabelEncoder()
df['genre_encoded'] = label_enc_genre.fit_transform(df['main_genre'])

# Filter classes with enough samples
valid_classes = df['certificate_encoded'].value_counts()
df = df[df['certificate_encoded'].isin(valid_classes[valid_classes >= 3].index)]

# ----------------- EDA Plots -----------------
sns.set(style='whitegrid')

plt.figure(figsize=(10, 4))
sns.countplot(x='certificate', data=df, order=df['certificate'].value_counts().index)
plt.title('Certificate Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(x='main_genre', data=df, order=df['main_genre'].value_counts().index)
plt.title('Genre Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['rating'], kde=True, bins=20)
plt.title('Rating Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='votes', y='rating', hue='main_genre', alpha=0.6, legend=False)
plt.xscale('log')
plt.title('Votes vs Rating')
plt.tight_layout()
plt.show()

# NEW: Movie Duration Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['duration'], bins=30, kde=True)
plt.title('Movie Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# ----------------- KNN Classification -----------------
X_cls = df[['year', 'duration', 'votes', 'genre_encoded']]
y_cls = df['certificate_encoded']

smote = SMOTE(random_state=42, k_neighbors=1)
X_cls_bal, y_cls_bal = smote.fit_resample(X_cls, y_cls)

scaler_cls = StandardScaler()
X_cls_bal_scaled = scaler_cls.fit_transform(X_cls_bal)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls_bal_scaled, y_cls_bal, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_c, y_train_c)
y_pred_knn = knn.predict(X_test_c)

acc_knn = accuracy_score(y_test_c, y_pred_knn)
print("\nKNN Classification Accuracy: {:.2f}%".format(acc_knn * 100))
print(classification_report(y_test_c, y_pred_knn, zero_division=0))

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test_c, y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ----------------- XGBoost Regression -----------------
df['votes_log'] = np.log1p(df['votes'])
df['year_bin'] = pd.cut(df['year'], bins=5, labels=False)
df_ohe = pd.get_dummies(df[['certificate', 'main_genre']], drop_first=True)

X_reg = pd.concat([df[['year', 'duration', 'votes_log', 'year_bin']], df_ohe], axis=1)
y_reg = df['rating']

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

xgb = XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.1, objective='reg:squarederror', random_state=42)
xgb.fit(X_train_r, y_train_r)

# Predictions
y_pred_test = xgb.predict(X_test_r)
y_pred_train = xgb.predict(X_train_r)

# Evaluation
print("\nXGBoost Regression:")
print("Train R² Score:", r2_score(y_train_r, y_pred_train))
print("Test R² Score:", r2_score(y_test_r, y_pred_test))
print("MSE:", mean_squared_error(y_test_r, y_pred_test))

# Feature Importance
plt.figure(figsize=(10, 6))
plot_importance(xgb, max_num_features=10)
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()

#  NEW: Actual vs Predicted Ratings
plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_test, alpha=0.5)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted IMDb Ratings")
plt.tight_layout()
plt.show()

# NEW: Residuals Plot
residuals = y_test_r - y_pred_test
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=20, kde=True)
plt.title("Residuals Distribution (Actual - Predicted Ratings)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()