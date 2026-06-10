import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("PREDIKSI RISIKO PUTUS SEKOLAH - KNN HAMMING")

df_mat = pd.read_csv("student-mat.csv", sep=';')
df_por = pd.read_csv("student-por.csv", sep=';')
df = pd.concat([df_mat, df_por], ignore_index=True)

print(f"Data dimuat: {len(df)} siswa")

df['risiko'] = df['G3'].apply(
    lambda x: "TINGGI" if x < 10 else ("SEDANG" if x < 15 else "RENDAH")
)

print(f"Risiko TINGGI: {(df['risiko'] == 'TINGGI').sum()}")
print(f"Risiko SEDANG: {(df['risiko'] == 'SEDANG').sum()}")
print(f"Risiko RENDAH: {(df['risiko'] == 'RENDAH').sum()}")

fitur = ['age', 'studytime', 'failures', 'absences', 'G1', 'G2']
X = df[fitur]
y = df['risiko']

print(f"Fitur: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b)) / len(a)

def knn_hamming_predict(X_train, y_train, X_test, k=5):
    prediksi = []
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    for test_point in X_test:
        jarak = []
        for train_point in X_train.values:
            d = hamming_distance(test_point, train_point)
            jarak.append(d)

        tetangga = sorted(range(len(jarak)), key=lambda i: jarak[i])[:k]

        label_tetangga = [y_train.iloc[i] for i in tetangga]

        hasil = max(set(label_tetangga), key=label_tetangga.count)
        prediksi.append(hasil)

    return prediksi

print("Model KNN (K=5) dilatih")

prediksi = knn_hamming_predict(X_train, y_train, X_test, k=5)

akurasi = accuracy_score(y_test, prediksi)
print(f"Akurasi: {round(akurasi*100, 2)}%")

semua = knn_hamming_predict(X_train, y_train, X, k=5)

for risiko in ['TINGGI', 'SEDANG', 'RENDAH']:
    jumlah = sum(1 for s in semua if s == risiko)
    persen = round(jumlah / len(df) * 100, 1)
    print(f"Risiko {risiko}: {jumlah} siswa ({persen}%)")

print("selesai")