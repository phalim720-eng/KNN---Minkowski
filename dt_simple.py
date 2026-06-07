import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mat = pd.read_csv('student-mat.csv', sep=';')
por = pd.read_csv('student-por.csv', sep=';')
df = pd.concat([mat, por], ignore_index=True)

X = df[['age', 'studytime', 'failures', 'absences', 'G1', 'G2']]
y = (df['G3'] >= 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print(f"Akurasi: {accuracy_score(y_test, model.predict(X_test)):.2%}")

plt.figure(figsize=(18, 8))
plot_tree(model, feature_names=X.columns.tolist(),
          class_names=['Tidak Lulus', 'Lulus'], filled=True, rounded=True)
plt.title("Decision Tree - Prediksi Kelulusan Siswa")
plt.tight_layout()
plt.savefig('dt_simple.png', dpi=150)
print("Selesai.")