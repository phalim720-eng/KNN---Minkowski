import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r'/Users/halim/Desktop/python/buellyUAS/student-mat.csv', sep=';')

print("===== DATASET =====")
print(df.head())

df['status'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# ✅ FIX KUAT: encode semua kolom string/object pakai pd.factorize
for column in df.columns:
    if df[column].dtype == object or df[column].dtype.name == 'category':
        df[column], _ = pd.factorize(df[column])

# Pastikan semua kolom numerik, isi NaN dengan 0
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)

X = df.drop(['G3', 'status'], axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean'
)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\n===== HASIL EVALUASI =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))