# Klasifikasi Spesies Bunga Iris Menggunakan Jaringan Syaraf Tiruan (JST)

Repositori ini berisi implementasi Jaringan Syaraf Tiruan (Artificial Neural Network) menggunakan framework **TensorFlow** untuk mengklasifikasikan tiga spesies bunga Iris (Setosa, Versicolor, dan Virginica) berdasarkan dataset Iris klasik dari UCI Machine Learning Repository.

## Struktur File

- `pertemuan7.py` : Skrip Python utama yang mencakup seluruh alur kerja program, mulai dari pemuatan data, pra-pemrosesan, pembangunan model, pelatihan, evaluasi, visualisasi, hingga prediksi data baru secara interaktif.
- `iris.data` : File dataset Iris yang memuat 150 sampel dengan empat fitur (*sepal length*, *sepal width*, *petal length*, *petal width*) dan label spesiesnya.
- `training_history.png` : Grafik hasil visualisasi nilai *accuracy* dan *loss* selama proses pelatihan model.
- `confusion_matrix.png` : Visualisasi *confusion matrix* dalam bentuk *heatmap* untuk mengevaluasi performa klasifikasi model.

## Alur Program

Program dijalankan secara berurutan dengan tahapan berikut:

1. **Memuat Dataset** : Membaca file `iris.data` secara lokal menggunakan Pandas tanpa header, kemudian memisahkan fitur dan label.
2. **Pra-pemrosesan Data** : Mengubah label spesies dari teks menjadi angka menggunakan `LabelEncoder`, lalu membagi data menjadi data latih (80%) dan data uji (20%).
3. **Membangun Model JST** : Menyusun arsitektur *Sequential* dengan 3 *hidden layer* (aktivasi ReLU dengan 1000, 500, dan 300 neuron) serta 1 *output layer* dengan 3 neuron (aktivasi Softmax).
4. **Pelatihan & Evaluasi** : Melatih model selama 50 epoch menggunakan optimizer Adam dan mengevaluasi performa pada data uji.
5. **Visualisasi** :
   - Grafik perubahan nilai *Accuracy* dan *Loss* per epoch selama pelatihan.
   - *Confusion Matrix* dalam bentuk *heatmap* untuk melihat detail hasil prediksi per kelas.
6. **Prediksi Interaktif** : Pengguna dapat memasukkan data ukuran bunga baru melalui terminal untuk diprediksi spesiesnya oleh model yang sudah dilatih.

## Hasil Pelatihan

| Metrik | Nilai |
|--------|-------|
| Accuracy | 96.67% |
| Loss | 0.0699 |

## Arsitektur Model

| Layer | Jumlah Neuron | Aktivasi |
|-------|--------------|----------|
| Input | 4 | - |
| Hidden Layer 1 | 1000 | ReLU |
| Hidden Layer 2 | 500 | ReLU |
| Hidden Layer 3 | 300 | ReLU |
| Output | 3 | Softmax |

## Persyaratan Library

Pastikan Python sudah terinstal, lalu install semua library yang dibutuhkan dengan perintah:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

## Cara Menjalankan

1. Clone repository ini atau unduh semua file ke dalam satu folder.
2. Pastikan file `iris.data` berada di folder yang sama dengan `pertemuan7.py`.
3. Buka terminal dan arahkan ke folder tersebut, lalu jalankan:
```bash
python pertemuan7.py
```
4. Tunggu proses pelatihan selesai (epoch 1 sampai 50 akan tampil di terminal).
5. Setelah pelatihan selesai, jendela grafik *Training History* akan muncul — **tutup jendela tersebut** untuk melanjutkan.
6. Jendela *Confusion Matrix* akan muncul berikutnya — **tutup juga** untuk melanjutkan.
7. Program akan meminta input 4 nilai ukuran bunga (contoh: `5.1`, `3.5`, `1.4`, `0.2`). Masukkan satu per satu lalu tekan Enter untuk melihat hasil prediksi spesiesnya.
