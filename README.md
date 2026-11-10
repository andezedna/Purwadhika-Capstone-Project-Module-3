# Purwadhika-Capstone-Project-Module-3
**Judul : Prediksi Pelanggan Churn Menggunakan Machine Learning di Industri Telekomunikasi**<br>
**Oleh : Annisa Andez Aunillah**
<br>
## **Business Problem Understanding**
**Context**

Industri telekomunikasi yang makin beragam menyebabkan persaingan yang lebih ketat. Pelanggan mudah berpindah ke operator lain. Biaya mendapatkan pelanggan baru umumnya lebih besar daripada biaya mempertahankan pelanggan yang sudah ada (loyal customer). Maka dari itu, pelanggan yang berhenti berlangganan (*customer churn*) berdampak langsung pada pendapatan perusahaan telekomunikasi.<br>
Perusahaan memiliki data pelanggan yang berhenti berlangganan (*customer churn*) atau pelanggan yang lanjut berlangganan (*customer not churn*). Data pelanggan ini memungkinkan digunakan untuk mengenali pola pelanggan yang berhenti berlangganan, sehingga perusahaan dapat melakukan upaya retensi pada pelanggan tersebut.

**Target**:<br>
0 : Customer Not Churn<br>
1 : Customer Churn

**Problem Statement:**

Perusahaan belum memiliki cara untuk mengetahui pelanggan mana yang akan berhenti berlangganan. Dalam proses mempertahankan pelanggan agar tidak berhenti berlangganan, perusahaan harus mengeluarkan biaya untuk memberikan promo atau penawaran khusus. Di sisi lain, proses mendapatkan pelanggan baru jauh lebih sulit.

**Goals:**

Maka berdasarkan permasalahan tersebut, perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan pelangganan akan berhenti berlangganan atau tidak, sehingga dapat memfokuskan upaya retensi pada pelanggan yang kemungkinan akan berhenti. Dan juga, perusahaan ingin mengetahui apa/faktor/variabel apa yang membuat pelangganan akan berhenti berlangganan atau tidak, sehingga mereka dapat membuat rencana yang lebih baik dalam melakukan upaya terbaik terhadap pelanggan yang kemungkinan akan berhenti berlangganan.

**Analytic Approach:**

Jadi yang akan kita lakukan adalah menganalisis data untuk menemukan pola yang membedakan pelangganan akan berhenti berlangganan atau tidak. Kemudian kita akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas pelangganan akan berhenti berlangganan atau tidak.

**Metric Evaluation:**

| Tipe Error | Detail |
| --- | ---|
| False Positive | Model memprediksi pelanggan yang tidak churn sebagai pelanggan yang churn |
| False Negative | Model memprediksi pelanggan yang churn sebagai pelanggan yang tidak churn |


Dari tabel diatas, konsekuensi dari False Positive adalah perusahaan harus mengeluarkan biaya untuk memberikan promo atau penawaran khusus agar pelanggan yang diprediksi churn akan tetap berlangganan. Konsekuensi dari False Negative adalah perusahaan tidak melakukan retensi kepada pelanggan yang churn sehingga pelanggan pindah ke operator lain. Konsekuensi tersebut menyebabkan pelanggan kehilangan pelanggan dan biaya mendapatkan pelanggan baru lebih mahal.

False Positive : Kita asumsikan cost untuk upaya retensi pelangganan akan tetap berlangganan adalah Rp 100,000<br>
False Negative : Kita asumsikan cost untuk kehilangan pelanggan dan biaya akuisisi pelanggan baru adalah Rp 500,000<br>
Kita asumsikan cost False Positive dan False Negaative diatas dengan menggunakan ratio **FP:FN = 1:5**<br>
<br>
Dari penjelasan konsekuensi False Positive dan False Negative diatas, matriks yang cocok untuk adalah F2 karena cost FN lebih mahal daripada cost FP. Matriks F2 dapat menekan recall (menekan cost FN), namun dapat menekan precision (menekan FP) jika FP berlebihan.

## **Data Understanding**
**Context**

Dataset mempresentasikan profil pelanggan yang telah meninggalkan perusahaan telekomunikasi. Istilah **Churn** dalam telekomunikasi dan layanan berbasis langganan lainnya merupakan situasi ketike pelanggan meninggalkan profider layanan tersebut.

**Features**
| Column | Description |
| --- | --- |
| Dependents | Status pelanggan memiliki tanggungan atau tidak |
| Tenure | Jumlah bulan pelanggan sudah berlanggangan dengan perusahaan telekomunikasi |
| OnlineSecurity | Status pelanggan memiliki Online Security atau tidak |
| OnlineBackup | Status pelanggan memiliki Online Backup atau tidak |
| InternetService | Status pelanggan berlangganan layanan internet atau tidak |
| DeviceProtection | Status pelanggan memiliki perlindungan perangkat atau tidak |
| TechSupport| Status pelanggan memiliki dukungan teknis atau tidak |
| Contract | Jenis kontrak berdasarkan durasi |
| PaperlessBilling | Tagihan dalam bentuk tanpa kertas (Yes/No) |
| MonthlyCharges| Total biaya untuk layanan secara bulanan |
| Churn | Status pelanggan **Churn** atau **Not Churn** |

## **Step Pengerjaan**
### 1. Exploratory Data Analysis (EDA)

Pada bagian ini dilakukan Exploratory Data Analysis (EDA) dengan tujuan sebagai berikut:

- Melihat distribusi data pada tiap kolom
- Mengetahui korelasi antarkolom
- Mengetahui persebaran data churn dan tidak churn (target) berdasarkan feature yang ada

### 2. Mendefinisikan X dan y

Pada bagian ini didefinisikan X sebagai feature dan y sebagai target.

- Dataframe `X` berisi semua kolom selain kolom `Churn`, karena pada dataset tidak ada kolom primary key seperti customer ID
- Dataframe `y` berisi kolom `Churn` sebagai target, dengan nilai diubah menjadi numerik:
    - `Yes` diubah menjadi `1`
    - `No` diubah menjadi `0`

### 3. Data Splitting

Pada bagian ini dilakukan pembagian data train dan test dengan proporsi `80:20`.

### 4. Data Preprocessing

Pada bagian ini dilakukan feature encoding untuk tiap feature:

- **One Hot Encoding**: `Dependents`, `OnlineSecurity`, `OnlineBackup`, `InternetService`, `DeviceProtection`, `TechSupport`, `Contract`, `PaperlessBilling`
- **Robust Scaler**: `tenure`, `MonthlyCharges`
- **Binning**: `tenure`

Dibangun 2 transformer:

- Transformer dengan feature `tenure` **tidak** dibinning
- Transformer dengan feature `tenure` **dibinning**

### 5. Cross Validation

Pada bagian ini dilakukan pengecekan skor tiap model dengan 2 transformer yang telah dibuat. Model yang digunakan:

- `LogisticRegression`
- `KNeighborsClassifier`
- `DecisionTreeClassifier`
- `VotingClassifier`
- `StackingClassifier`
- `BaggingClassifier`
- `RandomForestClassifier`
- `AdaBoostClassifier`
- `GradientBoostingClassifier`

Cross validation dilakukan tanpa metode resampling dan dengan metode resampling. Pada tahap resampling dilakukan 4 percobaan:

1. `SMOTE` dengan transformer tanpa binning
2. `RandomOverSampler (ROS)` dengan transformer tanpa binning
3. `SMOTE` dengan transformer yang dibinning
4. `RandomOverSampler (ROS)` dengan transformer yang dibinning

### 6. Hyperparameter Tuning

Model yang dilakukan hyperparameter tuning ada 2:

- `LogisticRegression`
- `AdaBoostClassifier`

Pemilihan kedua model ini berdasarkan skor tertinggi pada tahap cross validation sebelumnya.

### 7. Evaluasi Model

Dari hyperparameter tuning model `LogisticRegression` dan `AdaBoostClassifier` diperoleh:

- Biaya total terkecil `LogisticRegression`: **Rp 41,300,000** dengan threshold `0.41`
- Biaya total terkecil `AdaBoostClassifier`: **Rp 43,800,000** dengan threshold `0.26`

Dari hasil tersebut, model final yang dipilih adalah `LogisticRegression`. Selanjutnya dilakukan model-based feature selection pada model terpilih tersebut.

### 8. Interpretasi Model

Pada bagian ini dilakukan penjelasan model Logistic Regression melalui:

- Koefisien dan odds ratio
- SHAP
- LIME


## **Kesimpulan dan Saran**

## Kesimpulan

**Model final**

- Algoritma: `LogisticRegression` dengan `penalty='l1'`, `C=0.01`, dan `solver='liblinear'`
- Feature `tenure` dibinning menjadi 3 kelompok
- Resampling menggunakan `RandomOverSampler`
- Menggunakan 10 dari 18 feature dan threshold `0.41`
- Dengan threshold 0.41, model dapat menghemat biaya. Total biaya sebesar **Rp 41,300,000**, dengan asumsi cost FP Rp 100,000 dan cost FN Rp 500,000 (rasio FP:FN = 1:5)

**Kondisi model**

- Model final tidak overfitting maupun underfitting
- Model final merupakan hasil feature selection dengan membuang 8 feature yang koefisiennya di bawah 0.05, karena pembuangan tersebut tidak menurunkan skor
- Dengan demikian, model dengan 10 feature lebih ringkas tanpa mengurangi performa

**Faktor yang memengaruhi risiko churn**

- Menurunkan risiko churn:
    - Kontrak tahunan (1 dan 2 tahun)
    - Tenure yang panjang
    - Penggunaan `OnlineSecurity`
- Menaikkan risiko churn:
    - Tenure paling pendek (0-9 bulan)
    - Penggunaan layanan `Fiber optic`
    - Tagihan bulanan yang tinggi

## Saran

1. Prioritaskan pelanggan di tahun pertama (tenure di bawah 12 bulan), berkontrak bulanan (month-to-month), dan menggunakan layanan Fiber optic, karena kelompok ini berisiko tinggi untuk churn.
2. Tingkatkan marketing untuk mendorong pelanggan mengambil kontrak tahunan (1 dan 2 tahun), karena kelompok ini berisiko churn lebih rendah.
3. Evaluasi layanan Fiber optic, mencakup harga, fasilitas, dan kualitas, karena pelanggan Fiber optic berisiko lebih tinggi untuk churn.
4. Tingkatkan marketing untuk menawarkan layanan `OnlineSecurity` dan `OnlineBackup` sebagai bagian dari paket, untuk menurunkan risiko churn.

**Pemanfaatan model**

- Tim Customer Relationship Management (CRM) dapat menggunakan model ini untuk mengetahui probabilitas churn tiap pelanggan, sehingga dapat memprioritaskan pelanggan yang cocok diberi penawaran.
- Tim Marketing, Finance, dan Manajemen dapat memanfaatkan faktor risiko dan analisis biaya untuk menyusun strategi dan anggaran selanjutnya.

## Limitasi Model

1. Cost FP Rp 100,000 dan cost FN Rp 500,000 (rasio FP:FN = 1:5) hanya asumsi. Kesimpulan biaya dapat berubah menyesuaikan data keuangan perusahaan.
2. Threshold final sebesar 0.41 dipilih dari data test, sehingga estimasi performa dan penghematan biaya cenderung sedikit optimis.
3. Feature pada dataset terbatas, sehingga feature pada model final juga terbatas. Masih ada faktor lain yang dapat memengaruhi churn, seperti tingkat kepuasan pelanggan, riwayat keluhan, dan promo paket.

**Link Streamlit** : https://purwadhika-capstone-project-module-3-gj62jsucftyrrqcycryeuc.streamlit.app/
**Link Presentasi** : https://canva.link/uovlsx96e61nhwx
