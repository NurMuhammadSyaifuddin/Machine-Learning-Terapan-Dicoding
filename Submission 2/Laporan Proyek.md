# Laporan Proyek Machine Learning - Nur Muhammad Syaifuddin

## Project Overview

![buku](https://d33wubrfki0l68.cloudfront.net/5bf2ba2753b38bba650410fe7c31ba7d51e09b85/dc516/uploads/badung-tambah-koleksi-buku-perpustakaan-sd-senila-800-2018-08-13-105154_0.jpg)

Buku adalah jendela dunia. Dengan buku seseorang dapat menjelajah ke dunia luar tanpa perlu pergi ke dunia luar. Dengan buku seseorang dapat memperoleh pengetahuan yang tiada batas, melintas waktu, dan mengenal seseorang dari seluruh belahan dunia, karena buku merupakan sumber ilmu pengetahuan. Untuk dapat memperoleh ilmu yang ada di dalam buku, seseorang harus membaca buku (Patiung, 2016). Kegiatan membaca buku sangat penting bagi kehidupan manusia, dengan terbiasa membaca buku, maka seseorang akan memiliki cakrawala pengetahuan yang luas (Shofaussamawati, 2014). Namun dengan banyaknya jumlah buku yang tersedia terkadang membuat pembaca kebingungan dalam menentukan buku yang ingin mereka baca. Terkadang dijumpai pembaca yang hanya ingin membaca buku-buku yang dengan reputasi penjualan terbaik. Ada pula pembaca yang hanya ingin membaca buku yang mirip dengan buku-buku yang pernah dibaca sebelumnya. Tidak jarang juga ditemui pembaca yang menentukan buku-buku yang akan dibaca selanjutnya berdasarkan rating dari buku-buku yang telah dilihatnya. Semakin tinggi rating dari buku tersebut, semakin tertarik pula pembaca untuk membacanya. Semakin rendah rating dari buku tersebut, maka pembaca cenderung enggan untuk membacanya (Andrew Hans Ritdrix, 2018).
Berdasarkan permasalahan tersebut, pada proyek ini akan dibuat suatu model sistem rekomendasi menggunakan teknik collaborative filtering untuk merekomendasikan buku-buku yang mungkin akan dibaca oleh pengguna. Collaborative filtering merupakan metode yang digunakan untuk merekomendasikan item berdasarkan penilaian pengguna sebelumnya, dimana attribut yang digunakan bukan konten tetapi user behaviour. Contohnya yaitu merekomendasikan suatu item berdasarkan dari riwayat rating dari user tersebut maupun user lain (Hadi Ati et al., 2016). Dengan adanya sistem rekomendasi ini diharapkan dapat membantu pengguna mendapatkan rekomendasi buku-buku yang sesuai dengan preferensi pengguna di masa lalu, buku-buku yang mungkin disukai, dan belum pernah dibaca oleh pengguna.

## Business Understanding

### Problem Statements

Berdasarkan penjelasan pada project overview, berikut ini merupakan rincian masalah yang perlu diselesaikan di proyek ini:

-   Sistem rekomendasi apa yang baik untuk diterapkan pada kasus ini?
-   Bagaimana cara membuat sistem rekomendasi buku yang mungkin disukai dan belum pernah dibaca oleh pengguna?

### Goals

Tujuan dibuatnya proyek ini adalah sebagai berikut :
-   Membuat sistem rekomendasi buku sesuai dengan preferensi pengguna.
-   Memberikan rekomendasi buku yang mungkin disukai dan belum pernah dibaca oleh pengguna.

### Solution Approach

Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :

-   **Pra-pemrosesan Data**. Pada pra-pemrosesan data dapat dilakukan beberapa tahapan, antara lain :

    -   Menghapus kolom/fitur yang tidak diperlukan.
    -   Mengganti tipe data pada kolom.
    -   Membersihkan data kosong pada kolom.
    -   Melakukan _text cleaning_ pada data.

-   **Persiapan Data**. Pada persiapan data dapat dilakukan beberapa tahapan, antara lain :

    -   Persiapan data untuk model KNN.
        -   Filtering data buku dengan jumlah rating >= threshold (30).
        -   Mengubah format data menjadi pivot tabel.
        -   Mengkonversi value (rating) pada pivot tabel ke dalam scipy sparse matrix.

    -   Persiapan data untuk model Deep Learning.
        -   Melakukan proses encoding fitur user_id dan isbn ke dalam indeks integer.
        -   Pembagian Data untuk Training dan Validasi.

-   **Pembangunan Model**. Pada proyek ini sistem rekomendasi yang dibuat menggunakan teknik _collaborative filtering_ karena sesuai dengan dataset yang akan digunakan. Sehingga sistem rekomendasi dibuat untuk memberikan rekomendasi pada pengguna terhadap buku yang mirip dengan preferensi pengguna di masa lalu. Pada pembangunan model sistem rekomendasi terdapat beberapa pendekatan yang digunakan, antara lain :
    -   **Dengan pendekatan Item-Based dengan algoritma K-Nearest Neighbor.**
        <br> Item-based collaborative filtering merupakan metode rekomendasi yang bekerja berdasarkan adanya kesamaan antara  pemberi rating terhadap item yang dituju. Dari tingkat kesamaan item, kemudian dibagi berdasarkan parameter kebutuhan pelanggan untuk memperoleh nilai kegunaan item. Item yang memiliki nilai tertinggi maka akan dijadikan rekomendasi [[5](https://ejournal.upi.edu/index.php/JATIKOM/article/download/33208/14281)]. Kemudian algoritma yang digunakan pada pendekatan ini yaitu  K-Nearest Neighbor (KNN) karena mudah digunakan dan dapat mengantisipasi jika pengguna kurang paham  dengan apa yang ingin dicari karena metode ini menerapkan prinsip pencarian menggunakan jarak kedekatan (kemiripan data) sampel  dengan  data  yang  ada [[6](https://journals.telkomuniversity.ac.id/tektrika/article/view/1846/1141)]. Kelebihan dan kekurangan algoritma K-Nearest Neighbor adalah sebagai berikut (bersumber dari [[6](https://journals.telkomuniversity.ac.id/tektrika/article/view/1846/1141)]) :
    
        -   Kelebihan :
            -   Keakuratan hasil yang diperoleh lebih dijamin
            -   Untuk Data Training yang besar, hasilnya akan lebih efektif.
        -   Kekurangan :
            -   Berdasarkan perhitungan nilai jarak (Distance Based Learning), tidak jelas atribut mana yang memberikan hasil yang baik dan perhitungan jarak mana yang sebaiknya digunakan,
            -   Peneliti perlu menghitung nilai baru ke semua data yang ada pada Data Training dan menghitung jarak karena nilai komputasinya tinggi
            -   Parameter K perlu ditunjukkan (jumlah tetangga terdekat).
            
    -   **Dengan pendekatan Deep learning atau Neural Network.**
        <br>Deep learning merupakan subbidang machine learning yang algoritmanya terinspirasi dari struktur otak manusia. Struktur tersebut dinamakan Artificial Neural Networks atau disingkat ANN. Pada dasarnya, ia merupakan jaringan saraf yang memiliki tiga atau lebih lapisan ANN. Ia mampu belajar dan beradaptasi terhadap sejumlah besar data serta menyelesaikan berbagai permasalahan yang sulit diselesaikan dengan algoritma machine learning lainnya 
[[7](https://www.dicoding.com/blog/mengenal-deep-learning/)]. Penerapan metode Deep Learning menjadi salah satu metode yang populer untuk sistem rekomendasi. Penggunaan metode Deep Learning pada sistem rekomendasi lebih efisien dan tepat sasaran. Beberapa kelebihan penerapan Deep Learning adalah sebagai berikut (bersumber dari [[6](https://journals.telkomuniversity.ac.id/tektrika/article/view/1846/1141)]) :
        
        -   Dapat memproses unstructured data seperti teks dan gambar.
        -   Dapat mengotomatisasi proses ekstraksi fitur tanpa perlu melakukan proses pelabelan secara manual.
        -   Memberikan hasil akhir yang berkualitas.
        -   Dapat mengurangi biaya operasional.
        -   Dapat melakukan manipulasi data dengan lebih efektif.

## Data Understanding

-   **Informasi Dataset**
    <br> Dataset yang digunakan pada proyek ini yaitu Book-Crossing dataset, informasi lebih lanjut  mengenai dataset tersebut dapat lihat pada tabel berikut:


    | Jenis                   | Keterangan                                                                              |
    | ----------------------- | --------------------------------------------------------------------------------------- |
    | Sumber                  | [Kaggle Dataset : Book-Crossing](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)  |
    | Dataset Owner           | Ruchi Bhatia                                                                            |
    | Lisensi                 | CC0: Public Domain                                                                      |
    | Kategori                | Arts and Entertainment, Online Communities, Literature                                  |
    | Jenis dan Ukuran Berkas | zip (80 MB)                                                                         |

    Setelah melakukan observasi pada dataset yang diunduh pada kaggle, didapatakan informasi sebagai berikut :

    -   Terdapat 1031175 baris (records atau jumlah pengamatan) dalam Book-Crossing dataset.
    -   Terdapat 19 kolom yaitu 'Unnamed: 0', 'user_id', 'location', 'age', 'isbn', 'rating', 'book_title', 'book_author', 'year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l', 'Summary', 'Language', 'Category', 'city', 'state', 'country'.
    -   Terdapat 3 kolom numerik dengan tipe data int64, yaitu: Unnamed: 0, user_id, rating. Ini merupakan fitur numerik. Tetapi untuk kolom Unnamed: 0 merupakan fitur yang tidak diperlukan dan bisa dibuang. 
    -   Terdapat 2 kolom numerik dengan tipe data float64 yaitu: age dan year_of_publication. Ini merupakan fitur numerik.
    -   Terdapat 14 kolom dengan tipe object, yaitu: location, isbn, book_title, book_author, publisher, img_s, img_m, img_l, Summary, Language, Category, city, state, country. Kolom ini merupakan categorical features (fitur non-numerik) dimana kolom ini merupakan target fitur.
    -   Tidak terdapat data duplikat pada dataset. 

     Untuk penjelasan mengenai variabel-variable pada Book-Crossing dataset dapat dilihat pada poin-poin berikut ini:

      - `Unnamed: 0` - index pada data
      - `user_id` - id dari pengguna
      - `location` - lokasi/alamat pengguna
      - `age` - umur pengguna
      - `isbn` - kode ISBN (International Standard Book Number) buku
      - `rating` - rating dari buku
      - `book_title` - judul buku
      - `book_author` - penulis buku
      - `year_of_publication` - tahun terbit buku
      - `publisher` - penerbit buku
      - `img_s` - gambar sampul buku (small)
      - `img_m` - gambar sampul buku (medium)
      - `img_l` - gambar sampul buku (large)
      - `Summary` - ringkasan/sinopsis buku
      - `Language` - bahasa yang digunakan buku
      - `Category` - kategori buku
      - `city` - kota pengguna
      - `state` - negara bagian penguna
      - `country` - negara pengguna

-   **Data Visualization**

    -   **Top 10 dari tahun penerbitan, penulis dan buku.**

        ![image](https://user-images.githubusercontent.com/68520848/186793579-36e66164-ac23-456f-9fe4-4aea4bb2c780.png)

        Dari hasil visualisasi di atas didapatkan informasi bahwa top 10 tahun penerbitan yaitu pada tahun 1995, 1996, 1997, 1994, 1998, 2000, 2003, 1999, 2001 dan 2002. Kemudian tahun 2002 merupakan tahun dengan jumlah buku terbit paling tinggi, dimana jumlah buku yang terbit pada tahun itu sebesar 87.088K.

        ![image](https://user-images.githubusercontent.com/68520848/186793542-74626d61-c10f-4c4c-8a17-142dcb1ffadf.png)

        Dari hasil visualisasi di atas didapatkan informasi bahwa top 10 penulis yaitu Janet Evanovich, Sue Grafton, Danielle Steel, Tom Clancy, Dean R. Knoontz, Marry Higgins Clark, James Patterson, John Grisham, Nora Roberts dan Stephen King. Kemudian Stephen King merupakan penulis dengan jumlah buku paling tinggi, dimana jumlah buku yang ditulis sebanyak 9679 buku.

        ![image](https://user-images.githubusercontent.com/68520848/186793610-7ae9cb99-bd8b-4f99-b83f-cd69e5bb4ce7.png)

        Dari hasil visualisasi di atas didapatkan informasi bahwa top 10 buku yaitu angels demons, the red tent bestselling backlist, divine secrets of the yaya sisterhood a novel, the secret life of beees, bridget joness diary, the nanny diaries a novel, a painted house, the davinci code, the lonely bones a novel dan wild animus. Kemudian wild animus merupakan buku yang paling diminati dengan jumlah pembaca paling tinggi yaitu 2381 pembaca.

    -   **Distribusi rating buku dan umur user.**
    
        ![image](https://user-images.githubusercontent.com/68520848/186793767-c4fcf74b-2c9e-4ab7-b582-590870cf7031.png)

        Dari hasil visualisasi di atas didapatkan informasi bahwa nilai pada kolom rating berada pada rentang 0 - 10. Pada hasil visualisai juga terlihat sebagian besar buku memiliki rating 0.

        ![image](https://user-images.githubusercontent.com/68520848/186793740-bae6ba5e-e44c-4479-a01b-2d4b1a400629.png)

        Dari hasil visualisasi di atas didapatkan informasi bahwa umur pengguna/user berada pada rentang 5 - 99 tahun. Pada hasil visualisai juga terlihat sebagian besar pengguna berada pada umur 34 tahun.

    -   **Wordcloud pada judu, penulis dan penerbit buku.**
        <br><br> Wordcloud kolom penulis (book_author)

        ![image](https://user-images.githubusercontent.com/68520848/186793837-4cd567fd-2c41-49b2-b431-e02c42c854bb.png)

        Wordcloud kolom judul buku (book_title)

        ![image](https://user-images.githubusercontent.com/68520848/186793872-9ed7cb38-b685-4654-b5ce-3020aa173495.png)

        Wordcloud kolom penerbit (publisher)

        ![image](https://user-images.githubusercontent.com/68520848/186793906-4ad2f4f8-51b1-470e-986b-2bac25195ab1.png)

        Dari hasil visualisasi di atas menunjukkan daftar kata-kata yang digunakan dalam dalam kolom book_author, book_title dan publisher, umumnya semakin banyak kata yang digunakan semakin besar ukuran kata tersebut dalam visualisasi. Pada visualisai terlihat bahwa kata-kata yang paling banyak muncul pada kolom book_author yaitu Stephen King dan King Stephen, pada kolom book_title yaitu novels paperback dan mysteries paperback dan pada kolom publisher yaitu Ballantine Books dan Publishing Group.

## Data Preparation

Berikut ini merupakan tahapan-tahapan dalam melakukan persiapan data :

-   **Menghapus kolom/fitur yang tidak diperlukan.** Pada data terdapat kolom/fitur yang tidak diperlukan karena tidak memberikan pengaruh pada proses pembuatan model sistem rekomendasi sehingga bisa dihapus atau dibuang. Kolom-kolom tersebut yaitu kolom 'Unnamed: 0' yang merupakan indeks dara dataset, kemudian kolom 'img_s', 'img_m', 'img_l' yang berisi data sampul gambar dari buku.

-   **Mengganti tipe data pada kolom.** Berdasarkan deskripsi variabel sebelumnya, didapatkan bahwa terdapat 2 kolom yang bertipe data float yaitu kolom 'year_of_publication' dan 'age'. Pada tahap ini kedua kolom tersebut akan diubah menjadi tipe data int, hal ini dilakukan karena tipe data pada kolom tersebut belum sesuai dengan data di kolomnya.

-   **Membersihkan data kosong pada kolom.** Pada Book-Crossing dataset terdapat beberapa kolom yang masih memiliki data kosong yaitu kolom city dengan 14103 data kosong, state dengan 22798 data kosong dan country dengan 35374 data kosong. Kemudian karena jumlah data kosong jauh lebih sedikit dari total dataset yaitu 1031175, maka data kosong tersebut akan dihapus dari data menggunakan fungsi dropna. Setelah membersihkan data kosong tersebut, jumlah data pada dataset berubah menjadi 982279 baris (records atau jumlah pengamatan).

-   **Melakukan _text cleaning_ pada data.** Pada dataset terlihat bahwa data text pada kolom book_title belum seragam dan masih mengandung tanda/karakter yang tidak diperlukan, oleh karena itu dilakukan text cleaning pada kolom tersebut. Text cleaning yang dilakukan terdiri dari membuat text menjadi lowercase, remove text dalam tanda kurung siku, remove links, remove punctuation dan remove angka.

-   **Persiapan data untuk model KNN Item-Based.**
    <br> Pada persiapan data untuk model KNN Item-Based terdiri dari 3 tahapan sebagai berikut :
    
    -   **Filtering data buku dengan jumlah rating >= threshold (30)**.
        <br> Dari data dapat dilihat bahwa hanya sekitar 293.037 dari 982.279 buku yang mendapat rating oleh lebih dari 30 pengguna dan sebagian besar sisanya kurang dikenal dengan sedikit atau tanpa interaksi pengguna yang disebut sparse rating (sparse data). Sparse rating ini kurang dapat diprediksi untuk sebagian besar pengguna dan sangat sensitif terhadap individu yang menyukai buku yang tidak jelas, yang membuat polanya sangat noise. Sebagian besar model membuat rekomendasi berdasarkan pola penilaian pengguna (user rating patterns). Untuk menghilangkan pola bising dan menghindari "memory error" karena kumpulan data besar, maka dilakukan proses filtering rating buku hanya untuk buku populer dimana data buku yang akan digunakan hanya buku-buku yang mendapatkan rating oleh lebih dari 30 pengguna. Setelah memfilter data, jumlah data yang digunakan menjadi 293.037 data dan sudah cukup untuk membuat model rekomendasi.
        
    -   **Mengubah format data menjadi pivot tabel.**
        <br> Sebelum masuk ke pembuatan model rekomendasi menggunakan KNN, terlebih dahulu kita harus mengubah data rating buku menjadi format yang tepat yang dapat digunakan oleh model KNN. Data rating buku akan di reshape ke dalam m x n array, dimana m merupakan jumlah buku dan n merupakan jumlah user, hal tersebut dapat meringkas nilai fitur pada dataframe ke dalam tabel dua dimensi yang rapi (pivot tabel) dengan judul buku (kolom book_title) menjadi indeks tabel, id user (kolom user_id) menjadi kolom tabel dan kolom rating menjadi nilai pada setiap baris tabel. Pada proyek ini, mengubah dataframe ke dalam pivot tabel dengan menggunakan modul [pivot_table](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html) dari pandas. Kemudian selanjutnya kita akan mengisi pengamatan yang hilang (data kosong) dengan nilai nol karena kita akan melakukan operasi aljabar linier (menghitung jarak antar vektor). Berikut merupakan pivot tabel yang dihasilkan :
        
        ![image](https://user-images.githubusercontent.com/68520848/186794005-bad28cf2-00d5-4601-83f9-b4a31fd2be17.png)

    -   **Mengkonversi value (rating) pada pivot tabel ke dalam scipy sparse matrix.**
        <br> Data dalam pivot tabel dapat dikatakan sebagai sparse matrix dengan shape 3602 x 46833. Sparse matrix merupakan matrix yang sebagian besar nilainya adalah nol. Tentu saja kita tidak ingin mengumpankan seluruh data dengan sebagian besar bernilai nol dalam tipe data float32 ke model KNN yang akan dibuat. Oleh karena itu untuk perhitungan yang lebih efisien dan mengurangi memory footprint, kita perlu mengubah nilai pada pivot tabel menjadi scipy sparse matrix dengan menggunakan modul [csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) pada library scipy.

-   **Persiapan data untuk model Deep Learning.**
    <br> Pada persiapan data untuk model Deep Learning.terdiri dari 2 tahapan sebagai berikut :
    
    -   **Melakukan proses encoding fitur user_id dan isbn ke dalam indeks integer.**
        <br> Pada tahap ini akan dilakukan proses encoding yaitu proses mengubah data non-numerik menjadi data numerik agar model dapat memproses data tersebut. Pada proyek ini, proses encoding dilakukan pada fitur user_id dan isbn dengan memanfaatkan fungsi enumerate. Kemudian memetakan user_id dan isbn ke dataframe yang berkaitan.

    -   **Pembagian Data untuk Training dan Validasi.**
        <br> Pada tahap ini kita akan melakukan pembagian data menjadi data training dan validasi. Namun sebelum itu, kita perlu mengacak datanya terlebih dahulu agar distribusinya menjadi random. Kemudian membuat variabel x untuk mencocokkan data user dan buku menjadi satu value, lalu membuat variabel y untuk membuat rating dari hasil. Setelah itu membagi menjadi 80% data train dan 20% data validasi. Setelah melakukan pembagian dataset, didapatkan jumlah sample pada data train yaitu 785823 sampel dan jumlah sample pada data validasi yaitu 196456 sampel.


## Modeling

Pada proyek ini, model yang dibuat merupakan sistem rekomendasi untuk merekomendasikan buku kepada pengguna. Pada proyek ini sistem rekomendasi yang dibuat menggunakan teknik _collaborative filtering_ dengan menggunakan 2 pendekatan yaitu pendekatan Item-Based dengan algoritma K-Nearest Neighbor dan pendekatan Deep learning atau Neural Network.

-   **Dengan pendekatan Item-Based dengan algoritma K-Nearest Neighbor.**
    <br> Untuk membangun model ini, digunakan fungsi [NearestNeighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) dari sklearn dengan parameter metriksnya yakni 'cosine' sehingga algoritma akan menghitung kesamaan cosinus antara vektor rating dan juga parameter algoritma yang digunakan untuk menghitung tetangga terdekat adalah 'brute'. Kemudian fungsi tersebut di inisialisasikan sebagai model_knn yang selanjutnya dilakukan fitting terhadap data yang berupa sparse matrix. Setelah itu dibuat fungsi recomend_book untuk memberikan rekomendasi terhadap suatu judul buku. Hasil rekomendasinya adalah seperti berikut :
    
    ![image](https://user-images.githubusercontent.com/68520848/186794492-85b66511-447e-42f3-a9ed-c40b351a1acc.png)

    Dengan model K-Nearest Neighbor, kita mendapatkan 10 buku hasil rekomendasi terhadap buku dengan judul 'the rescue' dengan distance > 0.80.

-   **Dengan pendekatan Deep learning atau Neural Network.**
    <br> Untuk membangun model ini, digunakan metode Deep Learning atau Neural Network. Model yang dbangun akan menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan buku. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan buku. Kemudian, kita juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Model dengan pendekatan Deep Learning ini dibangun dengan membuat class RecommenderNet dengan [keras Model class](https://keras.io/api/models/model/). Selanjutnya, lakukan proses compile terhadap model. Model yang dibangun menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. Setelah itu lakukan proses training terhadap model.
    
    Untuk mendapatkan rekomendasi resto, pertama kita ambil sampel user secara acak dan definisikan variabel books_not_read yang merupakan daftar buku yang belum pernah dibaca oleh pengguna, daftar books_not_read inilah yang akan menjadi buku yang kita rekomendasikan. Variabel books_not_read diperoleh dengan menggunakan operator bitwise (~) pada variabel books_read_by_user. Sebelumnya, pengguna telah memberi rating pada beberapa buku yang telah mereka baca. Kita menggunakan rating ini untuk membuat rekomendasi buku yang mungkin cocok untuk pengguna. Kemudian, untuk memperoleh rekomendasi buku, gunakan fungsi model.predict() dari library Keras. Hasil rekomendasinya adalah seperti berikut :
    
    ![image](https://user-images.githubusercontent.com/68520848/186796305-aa31b709-ee75-4170-bfb4-37c3c190496c.png)
    
    Dengan pendekatan Deep Learning, kita dapat melihat top 10 buku yang direkomendasikan untuk user dengan id 219951. Dari beberapa buku rekomendasi menyediakan kategori 'Fiction', '9', dan 'Juvenile Fiction' yang sesuai dengan rating user. Kita memperoleh 1 rekomendasi buku dengan kategori 'Fiction', 6 rekomendasi buku dengan kategori '9' dan 3 rekomendasi buku dengan kategori 'Juvenile Fiction'.

## Evaluation

Pada proyek ini, untuk mengukur kinerja model dengan pendekatan Deep Learning untuk sistem rekomendasi digunakan Root Mean Squared Error (RMSE) sebagai metrics evaluationnya. Root Mean Square Error (RMSE) adalah  metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar. Cara Menghitung Root Mean Square Error (RMSE) adalah dengan mengurangi nilai aktual dengan nilai prediksi kemudian dikuadratkan dan dijumlahkan keseluruhan hasilnya kemudian dibagi dengan banyaknya data. Hasil perhitungan tersebut selanjutnya dihitung kembali untuk mencari nilai dari akar kuadrat [[8](https://www.khoiri.com/2020/12/cara-menghitung-root-mean-square-error-rmse.html)]. Berikut merupakan persamaan untuk menghitung RSME :

![rumus rmse](https://user-images.githubusercontent.com/71582007/141730434-094b905b-bd2b-4090-a223-755458fd239b.jpg)

Berikut merupakan visualisai metrik pada proses training terhadap model Deep Learning sebelumnya :

![image](https://user-images.githubusercontent.com/68520848/186796350-02477b92-6456-490e-9535-356223497e18.png)

Pada proses training dapat dilihat model cukup smooth dan model konvergen pada epochs sekitar 30. Dari proses ini, saya memperoleh nilai error akhir sebesar sekitar 0.3407 dan error pada data validasi sebesar 0.3491. Nilai tersebut cukup bagus untuk sistem rekomendasi. 

![image](https://user-images.githubusercontent.com/68520848/186797600-2303392e-5371-45ce-8ccf-738f941d73de.png)

Kemudian setelah dilakukan evaluasi menggunakan seluruh data, model memperoleh nilai error sebesar 0.3702.

## Penutup

Model Machine Learning berupa sistem rekomendasi buku bagi pengguna menggunakan Collaborative Filtering telah selesai dibuat. Setelah diujikan, model ini bekerja cukup baik dalam memberikan 10 rekomendasi teratas terhadap buku berdasarkan preferensi pengguna sebelumnya.

## Daftar Pustaka

Andrew Hans Ritdrix, P. W. W. (2018). Sistem rekomendasi buku menggunakan metode item-based collaborative filtering. Jurnal Masyarakat Informatika, 9, 24–32.

Hadi Ati, S., Saptono, R., & Salamah, U. (2016). Peningkatan Efektivitas Metode User-item based Collaborative Filtering pada Sistem Rekomendasi Wisata Kuliner Kota Solo. Jurnal Teknologi & Informasi ITSmart, 1(1), 01. https://doi.org/10.20961/its.v1i1.574

Patiung, D. (2016). Membaca Sebagai Sumber Pengembangan Intelektual. Al Daulah : Jurnal Hukum Pidana Dan Ketatanegaraan, 5(2), 352–376. https://doi.org/10.24252/ad.v5i2.4854

Shofaussamawati. (2014). Menumbuhkan minat baca dengan pengenalan perpustakaan pada anak sejak dini. Libraria, 2(1), 46–59.
