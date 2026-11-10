# Purwadhika-Capstone-Project-Module-3
**Judul : Prediksi Pelanggan Churn Menggunakan Machine Learning di Industri Telekomunikasi
Oleh : Annisa Andez Aunillah**

## **Business Problem Understanding**
**Context**<br>

Industri telekomunikasi yang makin beragam menyebabkan persaingan yang lebih ketat. Pelanggan mudah berpindah ke operator lain. Biaya mendapatkan pelanggan baru umumnya lebih besar daripada biaya mempertahankan pelanggan yang sudah ada (loyal customer). Maka dari itu, pelanggan yang berhenti berlangganan (*customer churn*) berdampak langsung pada pendapatan perusahaan telekomunikasi.<br>
Perusahaan memiliki data pelanggan yang berhenti berlangganan (*customer churn*) atau pelanggan yang lanjut berlangganan (*customer not churn*). Data pelanggan ini memungkinkan digunakan untuk mengenali pola pelanggan yang berhenti berlangganan, sehingga perusahaan dapat melakukan upaya retensi pada pelanggan tersebut.<br>
<br>
**Target**:<br>
0 : Customer Not Churn<br>
1 : Customer Churn

**Problem Statement:**<br>
<br>
Perusahaan belum memiliki cara untuk mengetahui pelanggan mana yang akan berhenti berlangganan. Dalam proses mempertahankan pelanggan agar tidak berhenti berlangganan, perusahaan harus mengeluarkan biaya untuk memberikan promo atau penawaran khusus. Di sisi lain, proses mendapatkan pelanggan baru jauh lebih sulit.

Proses perekrutan bisa memakan waktu dan sumber daya jika perusahaan menargetkan semua kandidat tanpa melakukan penyaringan terlebih dahulu. Perusahaan ingin meningkatkan efisiensi rekrutmen dengan mengetahui kandidat mana yang mencari perubahan pekerjaan dalam karir mereka sehingga mereka dapat dipekerjakan sebagai Data Scientist.

Dan jika pelatihannnya diberikan secara gratis kepada semua calon/kandidat, maka biaya tersebut akan menjadi sia-sia jika calon/kandidat yang mengikuti pelatihan tersebut tidak mau direkrut oleh perusahaannya.<br>
<br>
**Goals:**<br>
<br>
Maka berdasarkan permasalahan tersebut, perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan pelangganan akan berhenti berlangganan atau tidak, sehingga dapat memfokuskan upaya retensi pada pelanggan yang kemungkinan akan berhenti.

Dan juga, perusahaan ingin mengetahui apa/faktor/variabel apa yang membuat pelangganan akan berhenti berlangganan atau tidak, sehingga mereka dapat membuat rencana yang lebih baik dalam melakukan upaya terbaik terhadap pelanggan yang kemungkinan akan berhenti berlangganan.<br>
<br>
**Analytic Approach:**<br>
<br>
Jadi yang akan kita lakukan adalah menganalisis data untuk menemukan pola yang membedakan pelangganan akan berhenti berlangganan atau tidak

Kemudian kita akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas pelangganan akan berhenti berlangganan atau tidak.

**Metric Evaluation:**<br>

| Tipe Error | Detail |
| --- | ---|
| False Positive | Model memprediksi pelanggan yang tidak churn sebagai pelanggan yang churn |
| False Negative | Model memprediksi pelanggan yang churn sebagai pelanggan yang tidak churn |

<br>
Dari tabel diatas, konsekuensi dari False Positive adalah perusahaan harus mengeluarkan biaya untuk memberikan promo atau penawaran khusus agar pelanggan yang diprediksi churn akan tetap berlangganan. Konsekuensi dari False Negative adalah perusahaan tidak melakukan retensi kepada pelanggan yang churn sehingga pelanggan pindah ke operator lain. Konsekuensi tersebut menyebabkan pelanggan kehilangan pelanggan dan biaya mendapatkan pelanggan baru lebih mahal.<br>
<br>

False Positive : Kita asumsikan cost untuk upaya retensi pelangganan akan tetap berlangganan adalah Rp 100.000<br>
False Negative : Kita asumsikan cost untuk kehilangan pelanggan dan biaya akuisisi pelanggan baru adalah Rp 500.000<br>
Kita asumsikan cost False Positive dan False Negaative diatas dengan menggunakan ratio **FP:FN = 1:5**<br>
<br>
Dari penjelasan konsekuensi False Positive dan False Negative diatas, matriks yang cocok untuk adalah F2 karena cost FN lebih mahal daripada cost FP. Matriks F2 dapat menekan recall (menekan cost FN), namun dapat menekan precision (menekan FP) jika FP berlebihan.
