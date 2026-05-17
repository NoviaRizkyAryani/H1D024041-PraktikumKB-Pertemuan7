# Langkah 1: Import library yang diperlukan
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Langkah 2: Muat dataset iris dari file .data (bukan CSV)
# Membaca file iris.data menggunakan pandas tanpa header
dataset = pd.read_csv(
    'iris.data',          
    header=None,          
    sep=',',              
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
)

# Hapus baris yang kosong (baris 151 di iris.data biasanya kosong)
dataset.dropna(inplace=True)

print("5 data pertama:")
print(dataset.head())
print(f"\nJumlah data: {len(dataset)}")
print(f"Distribusi kelas:\n{dataset['species'].value_counts()}")

# Menyusun data X (fitur) dan y (label)
X = dataset.iloc[:, :-1].values   
y = dataset.iloc[:, -1].values    

# Langkah 3: Konversi label string menjadi numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  

print(f"\nKelas yang tersedia: {label_encoder.classes_}")
print(f"Contoh label setelah encoding: {y[:10]}")

# Langkah 4: Split data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nUkuran data training : {X_train.shape}")
print(f"Ukuran data testing  : {X_test.shape}")

# Langkah 5: Buat model Neural Network
# Input: 4 fitur
# Hidden layers: 1000, 500, 300 neuron (ReLU)
# Output: 3 neuron (softmax) untuk 3 kelas
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])

# Tampilkan ringkasan arsitektur model
print("\n=== Model Summary ===")
model.summary()

# Langkah 6: Kompilasi model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Langkah 7: Latih model
print("\n=== Proses Pelatihan Model ===")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Langkah 8: Evaluasi model
print("\n=== Evaluasi Model ===")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss    : {loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Langkah 9: Visualisasi loss dan accuracy selama training
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Training History - Loss & Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
plt.show()
print("Grafik training history disimpan sebagai 'training_history.png'")

# Langkah 10: Prediksi pada data testing
predictions = model.predict(X_test)
# Ambil indeks probabilitas tertinggi
predicted_classes = predictions.argmax(axis=1)

print("\n=== Hasil Prediksi ===")
print(f"Prediksi  : {predicted_classes}")
print(f"Label Asli: {y_test}")

# Langkah 11: Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Iris Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()
print("Confusion matrix disimpan sebagai 'confusion_matrix.png'")

# Langkah 12: Prediksi data baru dari input pengguna
def predict_new_data():
    print("\n=== Prediksi Spesies Bunga Iris ===")
    try:
        sepal_length = float(input("Masukkan sepal length: "))
        sepal_width  = float(input("Masukkan sepal width : "))
        petal_length = float(input("Masukkan petal length: "))
        petal_width  = float(input("Masukkan petal width : "))

        # Buat array data baru
        new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Lakukan prediksi
        prediction = model.predict(new_data, verbose=0)
        predicted_class = prediction.argmax(axis=1)

        # Konversi hasil numerik ke label asli
        predicted_label = label_encoder.inverse_transform(predicted_class)
        print(f"\nHasil prediksi kelas: {predicted_label[0]}")
        print(f"Probabilitas tiap kelas:")
        for cls, prob in zip(label_encoder.classes_, prediction[0]):
            print(f"  {cls}: {prob:.4f} ({prob*100:.2f}%)")

    except ValueError:
        print("Input tidak valid! Masukkan angka desimal.")

predict_new_data()