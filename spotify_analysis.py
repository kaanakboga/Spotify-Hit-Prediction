
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# Loading and Reviewing the Dataset
data = pd.read_csv('dataset.csv')

print("First 5 Rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Data:")
print(data.isnull().sum())

# Data Cleaning and Target Class Creation

# Clean up missing data
data = data.dropna()
print("\nMissing Data cleared.")

# Converting categorical data
if 'explicit' in data.columns:
    data['explicit'] = data['explicit'].astype(int)

# Remove unnecessary columns and add target class
data = data.drop(columns=['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'])
data['hit'] = data['popularity'].apply(lambda x: 1 if x >= 70 else 0)

print("\nHit Songs:")
print(data[['popularity', 'hit']].head(10))

# =============================
# Feature and Target Distinction
# =============================
X = data[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
          'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
y = data['hit']
print(data["hit"].value_counts())

# =============================
# Separation of Training and Test Data
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nVeri Seti Boyutları:")
print(f"Eğitim verisi boyutu: {X_train.shape}")
print(f"Test verisi boyutu: {X_test.shape}")

# SMOTE
print("\nBefore SMOTE:")
print(y_train.value_counts())

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(y_train.value_counts())

# Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x=y_train.value_counts().index, y=y_train.value_counts().values)
plt.xticks([0, 1], ['Not Hit', 'Hit'])
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.ylim(0, max(y_train.value_counts()) * 1.1)
for index, value in enumerate(y_train.value_counts().values):
    plt.text(index, value + 1000, str(value), ha='center')
plt.show()



# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================

# =============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# First Model Evaluation
y_pred = model.predict(X_test)

print("\n📌 Initial Model (Accuracy):", accuracy_score(y_test, y_pred))
print("\n📌 Initial Model Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix - First Model")
plt.xlabel("Estimated")
plt.ylabel("Actual Value")
plt.show()

# =============================
# 📌 10. Özelliklerin Önemi (Feature Importance)
# =============================
importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance - Base Model")
plt.show()

# =============================

# =============================
cv_scores = cross_val_score(model, X, y, cv=5)

print("\n📌 Cross Validation Sonuçları (5-Fold):", cv_scores)
print("📌 Ortalama Başarı Oranı:", cv_scores.mean())


#  Model Optimization (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("\n📌 Best Parameters:", grid_search.best_params_)

# Train with optimized model
best_params = grid_search.best_params_
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Final Test ve Evaluating
final_predictions = final_model.predict(X_test)

print("\n📌 Accuracy Rate After Optimized Model:", accuracy_score(y_test, final_predictions))
print("\n📌 Classification Report After Optimized Model:\n", classification_report(y_test, final_predictions))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, final_predictions), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix - Optimize Model")
plt.xlabel("Estimated")
plt.ylabel("Actual Value")
plt.show()

# 📌 2. Veri Setinin Yüklenmesi ve İncelenmesi
