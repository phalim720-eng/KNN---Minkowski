import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load & gabung data
mat = pd.read_csv('student-mat.csv', sep=';')
por = pd.read_csv('student-por.csv', sep=';')
df = pd.concat([mat, por], ignore_index=True)

# Fitur & target
X = df[['age', 'studytime', 'failures', 'absences', 'G1', 'G2']]
y = (df['G3'] >= 10).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN dengan Minkowski (p=2 = Euclidean, p=1 = Manhattan)
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2%}")