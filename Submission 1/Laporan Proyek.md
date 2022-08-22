# Laporan Proyek *Machine Learning* - Nur Muhammad Syaifuddin
<hr style="border:1px solid gray">

## Domain Proyek
#### <div align="left"><span style="white-space: pre-wrap; font: normal 12pt Arial; line-height: 1.5;">Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai **pertanian** dengan judul proyek "Prediksi Strategis Pemilihan Jenis Tanaman untuk Lahan Pertanian Tertentu"</span></div>
![image.png](attachment:image.png)

#### <div align="left"><span style="white-space: pre-wrap; font: normal 12pt Arial; line-height: 1.5;">**Latar Belakang**<br>Sektor pertanian di Indonesia hingga saat ini memegang peranan penting dalam perekonomian nasional dan pembangunan negara secara keseluruhan. Pertanian masih menjadi andalan sebagai sumber bahan pangan (Kusumaningrum, 2019) dan sumber mata pencaharian terbesar di Indonesia, bahkan hampir di setiap daerah di Indonesia terdapat lahan pertanian (Mulziatuddin & Aidar, 2018). Sebelum melakukan proses pertanian maka seorang petani harus mengumpulkan berbagai informasi yang dapat mempengaruhi proses pertanian terlebih dahulu. Informasi yang diperlukan antara lain seperti karaktersistik, struktur dan tekstur lahan, jenis tanaman yang akan ditanam, dan target panen (Telaah et al., 2020). Namun pada kenyataannya, dalam melakukan penanaman tanaman pada suatu lahan pertanian, tidak banyak petani yang memperhitungkan apakah lahan yang dimiliki cocok untuk ditanami jenis tanaman yang telah ditentukan atau tidak (Wiratama & Dewi, 2021). Keputusan seorang petani mengenai jenis tanaman apa yang akan ditanam di lahannya pada umumnya tergantung pada intuisi petani itu sendiri, petani akan lebih memilih tanaman yang sedang tren di wilayah sekitarnya dan biasanya mereka tidak memiliki pengetahuan yang cukup tentang kandungan nutrisi lahan seperti nitrogen, fosfor, kalium dalam lahan (Fatiharani, 2022). Sehingga, jika petani mengambil keputusan yang salah pada pemilihan tanaman maka hasil panen menjadi tidak optimal dan berpotensi mengalami kerugian yang cukup besar nantinya.<br><br>Berdasarkan permasalahan tersebut, maka pada proyek ini akan dibangun suatu model machine larning untuk memprediksi jenis tanaman yang cocok ditanam di lahan pertanian tertentu berdasarkan parameter kandungan N, P, K (Nitrogen, Fosfor, Kalium) pada lahan, curah hujan, suhu, kelembaban dan pH. Dengan adanya model machine learning ini, diharapkan dapat membantu dan memudahkan petani dalam mengambil keputusan tentang strategi pertanian khususnya dalam memilih jenis tanaman yang cocok untuk lahan mereka, sehingga dapat meminimalisir kesalahan penanaman serta dapat meningkatkan hasil produksi di sektor pertanian. Kemudian untuk tahap pengembangan selanjutnya implementasi dari model ini dapat dijalankan pada sebuah aplikasi berbasis web ataupun mobile.</span></div>

# *Business Understanding*

### Problem Statements
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut :

-   Bagaimana cara melakukan pra-pemrosesan data lahan sehingga dapat digunakan untuk membuat model yang baik?
-   Bagaimana cara membangun model _machine learning_ untuk mengklasifikasikan data lahan ke dalam jenis tanaman yang cocok ditanam pada lahan tersebut?


### Goals
Tujuan dibuatnya proyek ini adalah sebagai berikut :

-   Melakukan pra-pemrosesan data lahan agar dapat digunakan dalam membangun model.
-   Membangun model _machine learning_ untuk mengklasifikasikan jenis tanaman yang cocok ditanam pada lahan tertentu dengan tingkat akurasi > 98%.

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :

- **Pra-pemrosesan Data**. Pada pra-pemrosesan data dapat dilakukan beberapa tahapan, antara lain :
  
    -   Melakukan label encoding pada fitur target (label).
    -   Melakukan pembagian dataset.
    -   Mengatasi data _outliers_ pada data latih dengan metode LOF (_Local Outlier Factor_).
    -   Standardisasi data pada semua fitur numerik pada dataset.

- **Pembangunan Model**. Pada pembangunan model terdapat beberapa algorima yang digunakan, antara lain :

  - **K-Nearest Neighbor** 
    <br>K-Nearest Neighbor atau KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. KNN bisa digunakan untuk kasus klasifikasi dan regresi [[1](https://towardsdatascience.com/getting-acquainted-with-k-nearest-neighbors-ba0a9ecf354f)]. Cara kerja algoritma KNN adalah sebagai berikut (bersumber dari [[2](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]) :
    
    -   Inisialisasi nilai K (jumlah tetangga)
    -   Hitung jarak antara data baru yang ditanyakan dengan seluruh sampel data pelatihan
    -   Urutkan seluruh jarak berdasarkan jarak minimum dan tetapkan jumlah tetangga (nearest neighbors) sesuai dengan nilai K
    -   Pilih sejumlah K data dengan jarak terdekat
    -   Kemudian tentukan kelas atau label dari data baru
    
    Kelebihan dan kekurangan algoritma K-Nearest Neighbor adalah sebagai berikut (bersumber dari [[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]) :
    
    -   Kelebihan :
        -   Algoritma KNN merupakan algoritma yang sederhana dan mudah untuk diimplementasikan
        -   Dapat diimplementasikan pada beberapa kasus seperti klasifikasi, regresi dan pencarian
    -   Kekurangan :
        -   Algoritma KNN menjadi lebih lambat secara signifikan seiring meningkatnya jumlah sampel dan/atau variabel independen

  - **XGBoost Algorithm**
    <br> XGboost adalah algoritma yang merupakan implementasi lanjutan dari algoritma peningkatan gradien (Gradient Boosting). XGboost menggunakan prinsip ensemble yaitu menggabungkan beberapa set pembelajar (tree) yang lemah menjadi sebuah model yang kuat sehinga menghasilkan prediksi yang kuat. Kelebihan dari algoritma XGBoost adalah sebagai berikut (bersumber dari [[4](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)]) :
    
    -   Kelebihan :
        -   Dapat melakukan pemrosesan paralel yang dapat mempercepat komputasi
        -   Memiliki fitur regularisasi untuk mencegah overfitting
        -   Menangani berbagai jenis pola sparsity dalam data dengan lebih efisien
        -   Dilengkapi dengan built in cross validation
        
  - **Random Forest**
    <br> Random Forest (RF) adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar. Klasifikasi random forest dilakukan melalui penggabungan pohon (tree) dengan melakukan training pada sampel data yang dimiliki. Penggunaan pohon (tree) yang semakin banyak akan mempengaruhi akurasi yang akan didapatkan menjadi lebih baik. Penentuan klasifikasi dengan random forest diambil berdasarkan hasil voting dari tree yang terbentuk. Pemenang dari tree yang terbentuk ditentukan dengan vote terbanyak. Proses klasifikasi pada random forest berawal dari memecah data sampel yang ada kedalam decision tree secara acak. Setelah pohon terbentuk,maka akan dilakukan voting pada setiap kelas dari data sampel. Kemudian, mengkombinasikan vote dari setiap kelas kemudian diambil vote yang paling banyak.Dengan menggunakan random forest pada klasifikasi data maka, akan menghasilkan vote yang paling baik [[5](https://id.wikipedia.org/wiki/Random_forest)]. Kelebihan dan kekurangan algoritma Random Forest adalah sebagai berikut (bersumber dari [[6](https://medium.com/swlh/random-forest-classification-and-its-implementation-d5d840dbead0)]) :
    
    -   Kelebihan :
        -   Algoritma Random Forest merupakan algoritma dengan pembelajaran paling akurat yang tersedia. Untuk banyak kumpulan data, algoritma ini menghasilkan pengklasifikasi yang sangat akurat
        -   Berjalan secara efisien pada data besar
        -   Dapat menangani ribuan variabel input tanpa penghapusan variabel
        -   Memberikan perkiraan variabel apa yang penting dalam klasifikasi
        -   Memiliki metode yang efektif untuk memperkirakan data yang hilang dan menjaga akurasi ketika sebagian besar data hilang
    -   Kekurangan :
        -   Algoritma Random Forest overfiting untuk beberapa kumpulan data dengan tugas klasifikasi/regresi yang bising/noise
        -   Untuk data yang menyertakan variabel kategorik dengan jumlah level yang berbeda, Random Forest menjadi bias dalam mendukung atribut dengan level yang lebih banyak. Oleh karena itu, skor kepentingan variabel dari Random Forest tidak dapat diandalkan untuk jenis data ini.

# *Data Understanding*

- **Informasi Dataset**
  <br> Dataset yang digunakan pada proyek ini yaitu crop dataset, informasi lebih lanjut  mengenai dataset tersebut dapat lihat pada tabel berikut:

  | Jenis                   | Keterangan                                                                              |
  | ----------------------- | --------------------------------------------------------------------------------------- |
  | Sumber                  | Crop Dataset : [Kaggle](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset) |
  | Dataset Owner           | SIDDHARTH SHARMA                                                                          |
  | Lisensi                 | https://creativecommons.org/licenses/by/3.0/igo                                                                                 |
  | Kategori                | agriculture, sistem rekomendasi                                               |
  | Jenis dan Ukuran Berkas | CSV (150.03 kb)                                                                         |

  Setelah melakukan observasi pada dataset yang diunduh melalui link drive yaitu `Crop_prediction.csv`, didapatakan informasi sebagai berikut :
  
  - Terdapat  2200 baris (records atau jumlah pengamatan) yang berisi informasi mengenai data lahan pertanian
  - Terdapat 8 kolom yaitu `N, P, K, temperature, humidity, ph, rainfall, dan label` yang merupakan veriabel - variabel pada data
  - Dari kolom-kolom tersebut terdapat 3 kolom numerik dengan tipe data int64, yaitu `N, P, K` dan terdapat 4 kolom numerik dengan tipe data float64 yaitu `temperature, humidity, ph dan rainfall` yang merupakan fitur numerik. 
  - Terdapat 1 kolom dengan tipe object yaitu `label`, kolom ini merupakan _categorical features_ (fitur non-numerik) dimana kolom ini merupakan target fitur
  - Tidak terdapat missing value pada dataset. 
  
  Untuk penjelasan mengenai variabel-variable pada crop dataset dapat dilihat pada poin-poin berikut ini:

  - `N` - rasio kandungan Nitrogen dalam tanah
  - `P` - rasio kandungan Fosfor dalam tanah
  - `K` - rasio kandungan Kalium dalam tanah
  - `temperature` - suhu dalam derajat Celcius
  - `humidity` - kelembaban relatif dalam %
  - `ph` - nilai ph tanah
  - `rainfall` - curah hujan dalam mm
  - `label` - label dari crop yang cocok untuk tumbuh di lahan pertanian berdasarkan variabel numerik di atas. Kemudian terdapat 22 label crop pada data ini yaitu `'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'`

- **Sebaran atau Distribusi Data pada Setiap Fitur**
  <br> Berikut merupakan visualisasi data yang menunjukkan sebaran/ distribusi data pada setiap fitur numerik (`N, P, K, temperature, humidity, ph, rainfall`) :

![image-4.png](attachment:image-4.png)
![image-5.png](attachment:image-5.png)
![image-6.png](attachment:image-6.png)
![image-7.png](attachment:image-7.png)
![image-8.png](attachment:image-8.png)
![image-9.png](attachment:image-9.png)
![image-10.png](attachment:image-10.png)

Berdasarkan hasil visualisasi data diatas, dapat terlihat sebaran atau distribusi data yang ada pada setiap fitur. Termasuk nilai minimum, median, maksimum, Q1, Q3, batas atas dan batas bawah. Selain itu dapat dilihat juga pada beberapa fitur masih terdapat nilai outliers.
  
  Berikut merupakan visualisasi data yang menunjukkan sebaran/distribusi data pada fitur target (`label`) :
  
  ![image.png](attachment:image.png)

Berdasarkan hasil visualisasi dari fitur target 'label' dapat memberikan informasi bahwa dataset sudah seimbang dengan jumlah sampel masing-masing label yaitu 100 sampel, sehingga tidak perlu menyeimbangkan data lagi.
  
- **Rata-Rata Nilai pada fitur Numerik di Setiap label**
  <br> Berikut visualisasi rata-rata kandungan `N, P, K` terhadap setiap label :
  
  ![image-4.png](attachment:image-4.png)
  ![image-2.png](attachment:image-2.png)
  ![image-3.png](attachment:image-3.png)
  ![image-5.png](attachment:image-5.png)
  ![image-6.png](attachment:image-6.png)
  ![image-7.png](attachment:image-7.png)
  ![image-8.png](attachment:image-8.png)

 Hasil visualisasi di atas memberikan informasi mengenai rata-rata kandungan N, P, K terhadap setiap label crop. Dimana dapat dilihat bahwa terdapat beberapa label crop yang membutuhkan lahan dengan kandungan N,P,K tinggi dan beberapa label membutuhkan lahan dengan kandungan N,P,K rendah.
  
  Berikut visualisasi rata-rata tingkat `temperature, humidity dan rainfall` terhadap setiap label :
  
  ![image.png](attachment:image.png)
  ![image-2.png](attachment:image-2.png)
  ![image-3.png](attachment:image-3.png)
  ![image-4.png](attachment:image-4.png)
  ![image-5.png](attachment:image-5.png)
  ![image-6.png](attachment:image-6.png)
  ![image-7.png](attachment:image-7.png)

  Hasil visualisasi di atas memberikan informasi mengenai tingkat temperature, humidity dan rainfall terhadap setiap label crop. Dimana dapat dilihat bahwa terdapat beberapa label crop yang membutuhkan lahan dengan tingkat temperature, humidity dan rainfall tinggi dan beberapa label membutuhkan lahan dengan tingkat temperature, humidity dan rainfall rendah.

- **Korelasi antar Fitur Numerik**
  <br> Berikut merupakan _Correlation Matrix_ antar fitur numerik :
  
  ![image.png](attachment:image.png)
  
  Dari _Correlation Matrix_ dapat dilihat bahwa fitur P dan K memiliki korelasi yang sedikit tinggi.

# *Data Preparation*

Berikut ini merupakan tahapan-tahapan dalam melakukan pra-pemrosesan data :
- **Melakukan label encoding pada fitur target (label)**
  <br> Sebelum masuk ke tahap pembagian dataset, terlabih dahulu melakukan pemisahan antara variabel independen (N, P, K, temperature, humidity, ph, rainfall) sebagai data `X` dan variabel dependen (label) sebagai data `y`. Karena fitur label pada dataset merupakan fitur non-numerik yang berarti nilai pada fitur tersebut adalah kategorikal, maka sebelum dimasukan ke dalam data `y` telah dilakukan proses label encoding untuk fitur tersebut.  Label encoding merupakan teknik untuk mengubah jenis data kategorikal  menjadi data numerik yang dapat dipahami model. Pada proyek ini, label encoding dilakukan dengan menggunakan modul [LabelEncoder](https://scikitlearn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) dari scikit-learn.
  
- **Melakukan pembagian dataset**
    <br> Untuk mengetahui kinerja model ketika dihadapkan pada data yang belum pernah dilihat sebelumnya maka perlu dilakukan pembagian dataset. Pada proyek ini dataset dibagi menjadi data latih dan data uji dengan rasio 80% untuk data latih dan 20% untuk data uji. Data latih merupakan data yang akan kita latih untuk membangun model _machine learning_, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan digunakan untuk melihat kinerja atau performa dari model yang dilatih.  Pembagian dataset dilakukan dengan modul [train_test_split](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) dari scikit-learn. Setelah melakukan pembagian dataset, didapatkan jumlah sample pada data latih yaitu 1760 sampel dan jumlah sample pada data uji yaitu 440 sampel dari total jumlah sample pada dataset yaitu 2200 sampel.

- **Mengatasi data _outliers_ pada data latih dengan metode LOF (_Local Outlier Factor_)**
   <br> Data pencilan (outliers) merupakan nilai yang tidak normal pada dataset. Adanya data outliers ini akan membuat analisis terhadap serangkaian data menjadi bias, atau tidak mencerminkan fenomena yang sebenarnya sehingga dapat menyebabkan pada pembuatan model menjadi kurang optimal [[5](https://statisticsbyjim.com/basics/remove-outliers/)]. Oleh karena itu, untuk menangani ouliers pada proyek ini menerapkan metode Local Outlier Factor untuk mengidentifikasi outliers dan kemudian menghapusnya dari data latih. [LocalOutlierFactor](https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor), bekerja dengan cara menganalisis nilai lokalitas yang ada pada k-tetangga terdekat, yang jaraknya digunakan untuk memperkirakan kepadatan lokal. Dengan membandingkan kepadatan lokal sampel dengan kepadatan lokal tetangganya, sehingga dapat mengidentifikasi sampel yang memiliki kepadatan jauh lebih rendah daripada tetangganya. Apabila kepadatannya rendah maka ini dianggap sebagai outliers.

- **Standardisasi data pada semua fitur numerik pada dataset**
  <br> Standardisasi merupakan teknik transformasi yang paling umum digunakan dalam tahap data preparation. Standarisasi membantu untuk membuat semua fitur numerik berada dalam skala data yang sama dan membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini, standarisasi data dilakukan dengan menerapkan teknik [StandarScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) dari library Scikitlearn. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.
  
## Modeling
Pada proyek ini, model yang dibuat merupakan kasus _multiclass classification_ yaitu tugas klasifikasi dengan lebih dari dua kelas atau banyak kelas. Proses modeling dalam proyek ini menggunakan 3 algoritma _machine learning_ yaitu `K-Nearest Neighbor`, `Random Forest` dan `XGBoost Algorithm` kemudian membandingkan performanya.

- **K-Nearest Neighbor**
  <br> Pada tahap ini pembuatan model dilakukan dengan menggunakan modul [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) dari library Scikitlearn dengan nilai k = 1. Kemudian proses selanjutnya melakukan prediksi menggunakan data uji dan melakukan pengujian. Hasil pengujian dari model dengan algoritma K-Nearest Neighbor dapat dilihat pada tabel berikut :
  <p>Classification Report</p>

![image.png](attachment:image.png)

Confusion Matrix <br>
![image.png](attachment:image.png)
  
  Berdasarkan hasil pengujian di atas dapat dilihat bahwa model dengan algoritma K-Nearest Neighbor memperoleh nilai akurasi yaitu sebesar 0.986364.

- **XGBoost Algorithm**
  <br> Pada tahap ini pembuatan model dilakukan dengan menggunakan modul [XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html) dari library xgboost. Kemudian proses selanjutnya melakukan prediksi menggunakan data uji dan melakukan pengujian. Hasil pengujian dari model dengan XGBoost Algorithm dapat dilihat pada tabel berikut :
  <p>Classification Report</p>

![image.png](attachment:image.png)

 Confusion Matrix <br>

![image.png](attachment:image.png)

Berdasarkan hasil pengujian di atas dapat dilihat bahwa model dengan algoritma XGBoost memperoleh nilai akurasi yaitu sebesar 0.995455.

- **Random Forest**
  <br> Pada tahap ini pembuatan model dilakukan dengan menggunakan modul [RandomForestClassifier](https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dari library Scikitlearn. Kemudian proses selanjutnya melakukan prediksi menggunakan data uji dan melakukan pengujian. Hasil pengujian dari model dengan algoritma Random Forest dapat dilihat pada tabel berikut :
  <p>Classification Report</p>

![image.png](attachment:image.png)

 Confusion Matrix <br>

![image.png](attachment:image.png)

Berdasarkan hasil pengujian di atas dapat dilihat bahwa model dengan algoritma Random Forest memperoleh nilai akurasi yaitu sebesar 0.997727.

Dari hasil pengujian ketiga model yang telah dibuat, berikut merupakan perbandingan performa antar Model :

![image.png](attachment:image.png)

Dari tabel di atas dapat memberikan informasi bahwa ketiga model yang dibangun memiliki nilai akurasi di atas 98%. Dimana dapat dilihat juga bahwa model dengan algoritma Random Forest memiliki performa (nilai akurasi, precision, dan recall) yang lebih baik dari model dengan algoritma K-Nearest Neighbor dan XGBoost. 

# *Evaluation*

Pada proyek ini, model yang dibuat merupakan kasus _multiclass classification_ dan metrik evaluasi yang digunakan untuk mengukur kinerja model yaitu menggunakan metrik **akurasi, precision dan recall**. Pada klasifikasi dengan jumlah keluaran kelas yang lebih dari dua (multi-class), cara menghitung akurasi, presisi dan recall dapat dilakukan dengan menghitung rata-rata dari nilai akurasi, presisi dan recall pada setiap kelas. Berikut merupakan formula untuk menghitung nilai akurasi, presisi dan recall dari sistem klasifikasi multi-class (bersumber dari [[7](https://achmatim.net/2017/03/19/mengukur-kinerja-algoritma-klasifikasi-dengan-confusion-matrix/)]) :

- Akurasi
  <br> Akurasi merupakan perbandingan antara data yang terklasifikasi benar dengan keseluruhan data. Nilai akurasi dapat diperoleh dengan persamaan berikut :
  
  <img width="266" alt="akurasi" src="https://user-images.githubusercontent.com/71582007/139705435-7ee71aac-b3df-447f-8dc8-9b7315011491.PNG">

- Precision
  <br> Precision menggambarkan jumlah data kategori positif yang diklasifikasikan secara benar dibagi dengan total data yang diklasifikasi positif. Precision dapat diperoleh dengan persamaan berikut :
  
  <img width="266" alt="presisi" src="https://user-images.githubusercontent.com/71582007/139705453-e4e55344-1869-4ad8-88ab-8e0125b45847.PNG">

- Recall
  <br> Sementara itu, recall menunjukkan berapa persen data kategori positif yang terklasifikasikan dengan benar oleh sistem. Recall dapat diperoleh dengan persamaan berikut :
  
  <img width="267" alt="recall" src="https://user-images.githubusercontent.com/71582007/139705495-a20297ee-a787-4ca8-910f-3863246413ab.PNG">

Pada proyek ini, menghitung nilai akurasi, precsion dan recall dilakukan dengan menggunakan modul [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html), [recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) dari library Scikitlearn dan menambahkan parameter average = 'macro'.

# Daftar Pustaka

![image-2.png](attachment:image-2.png)
