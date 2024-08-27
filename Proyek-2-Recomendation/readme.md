# Laporan Proyek Machine Learning

<hr>

## Project Overview

Buku adalah jendela dunia. Dengan membaca buku, seseorang dapat memperoleh ilmu yang tiada batas, tanpa ada batasan
waktu, dan mengenal seseorang dari seluruh sisi dunia, karena buku merupakan sumber ilmu pengetahuan. Untuk dapat
memperoleh ilmu yang ada di dalam buku, seseorang harus membaca buku[1].

Membaca buku memiliki peranan yang sangat penting dalam memperluas pengetahuan seseorang. Namun, dengan banyaknya buku
yang tersedia, pembaca sering kali merasa bingung dalam memilih buku yang akan dibaca selanjutnya. Beberapa pembaca
mungkin hanya tertarik membaca buku yang mirip dengan yang pernah mereka baca sebelumnya, sementara yang lain cenderung
memilih buku berdasarkan genre atau jenisnya, dan mungkin merasa kurang tertarik jika buku-buku tersebut sangat berbeda.

Untuk mengatasi masalah ini, proyek ini bertujuan untuk membuat sistem rekomendasi buku menggunakan metode content-based
filtering. Metode ini merekomendasikan buku berdasarkan fitur-fitur dari buku yang sudah dibaca sebelumnya. Misalnya,
seperti halnya merekomendasikan film berdasarkan aktor utamanya. Dengan adanya sistem rekomendasi ini, diharapkan
pembaca dapat dengan mudah menemukan buku-buku yang sesuai dengan minat mereka dan membantu mereka dalam memilih bacaan
selanjutnya.

## Business Understanding

<hr>

### Problem Statements

Berdasarkan penjelasan pada [_project overview_](#project-overview), berikut merupakan rincian masalah yang perlu
diselesaikan di proyek ini:

- Sistem rekomendasi apa yang dapat di terapkan pada studi kasus ini?
- Bagaimana cara membuat sistem rekomendasi buku yang akan merekomendasikan buku berdasarkan jenis/genre dari buku?

### Goals

Tujuan dibuatnya proyek ini adalah sebagai berikut:

- Membuat sistem rekomendasi buku dengan jenis buku sebagai fitur.
- Memberikan rekomendasi buku yang mungkin disukai pengguna.

### Solution Approach

Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya:

- Membuat sebuah model sistem rekomendasi
- Mengevaluasi hasil rekomendasi yang diberikan

## Data Understanding

<hr>

Dataset yang digunakan berasal dari [kaggle datasets](https://kaggle.com/datasets), pada _task_ kali ini penulis
menggunakan data _Book crossing datasets_ yang berisi 2 folder. Folder tersebut adalah `Book reviews` dan
`Books Data with Category Language and Summary`, data yang di gunakan ada pada folder kedua, karena setelah penulis
melihat keseluruhan data, data pada folder kedua sudah di lakukan _preprocessing_ oleh pemilik data. Informasi lebih
lanjut mengenai dataset dapat dijelaskan pada tabel 1 berikut.

Tabel 1. Rangkuman informasi Dataset

| Jenis                 | Keterangan                                                                                          |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| Sumber                | [Book-Crossing: User review ratings](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) |
| Lisensi               | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)                            |
| Kategori              | Arts and Entertainment, Online Communities, Literature                                              |
| Jenis & Ukuran berkas | ZIP (600.34 MB) - 4 berkas `.csv`                                                                   |  

***

Penulis melakukan observasi terhadap dataset yang digunakan dan mendapatkan informasi sebagai beriktu :

- Terdapat 1031175 baris dalam _dataset_
- Terdapat 19 kolom yaitu 'Unnamed: 0', 'user_id', 'location', 'age', 'isbn', 'rating', 'book_title', 'book_author', '
  year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l', 'Summary', 'Language', 'Category', 'city', 'state', '
  country'.

Untuk penjelasan mengenai 19 kolom yaitu sebagai berikut:

- `user_id` : id dari pengguna
- `location` : lokasi/alamat pengguna
- `age` : umur pengguna
- `isbn` : kode ISBN (International Standard Book Number) buku
- `rating` : rating dari buku
- `book_title` : judul buku
- `book_author` : penulis buku
- `year_of_publication` : tahun terbit buku
- `publisher` : penerbit buku
- `img_s` : gambar sampul buku (ukuran kecil)
- `img_m` : gambar sampul buku (ukuran sedang)
- `img_l` : gambar sampul buku (ukuran besar)
- `Summary` : ringkasan/sinopsis buku
- `Language` : bahasa yang digunakan buku
- `Category` : kategori buku
- `city` : kota pengguna
- `state` : negara bagian penguna
- `country` : negara pengguna

***

### Data Exploration

Pada bagian ini akan dijelaskan tentang temuan di dalam data, mulai dari sample data, informasi singkat data dan rating
point distribution.

Tabel 2. Sample Data

| Unnamed: 0 | user_id | location | age                       | isbn | rating     | book_title | book_author         | year_of_publication | publisher               | img_s                                             | img_m                                             | img_l                                             | summary                                           | language | category           | city     | state      | country |  
|------------|---------|----------|---------------------------|------|------------|------------|---------------------|---------------------|-------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|----------|--------------------|----------|------------|---------| 
| 0          | 0       | 2        | stockton, california, usa | 18   | 0195153448 | 0          | Classical Mythology | 2002                | Oxford University Press | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | Provides an introduction to classical myths pl... | en       | ['Social Science'] | stockton | california | usa     |

Tabel 3. informasi singkat mengenai data

| #  | Column       		     | Non-Null Count | Dtype   |
|----|---------------------|----------------|---------|
| 0  | Unnamed: 0  			     | 1031175  		    | int64   |
| 1  | user_id      		     | 1031175  		    | int64   |
| 2  | location  			       | 1031175  		    | object  |
| 3  | age  				           | 1031175  		    | float64 |
| 4  | isbn  				          | 1031175  		    | object  |
| 5  | rating  				        | 1031175  		    | int64   |
| 6  | book_title   		     | 1031175  		    | object  |
| 7  | book_author 			     | 1031175  		    | object  |
| 8  | year_of_publication | 1031175  		    | float64 |
| 9  | publisher 			       | 1031175 		     | object  |
| 10 | img_s				           | 1031175  		    | object  |
| 11 | img_m				           | 1031175  		    | object  |
| 12 | img_l				           | 1031175  		    | object  |
| 13 | Summary				         | 1031175  		    | object  |
| 14 | Language				        | 1031175  		    | object  |
| 15 | Category				        | 1031175  		    | object  |
| 16 | city					           | 1017072  		    | object  |
| 17 | state				           | 1008377  		    | object  |
| 18 | country 				        | 995801   		    | object  |

Tabel 4. Distribusi variabel rating

| rating | count  |
|--------|--------|
| 0      | 647323 |
| 1      | 1481   |
| 2      | 2375   |
| 3      | 5118   |
| 4      | 7617   |
| 5      | 45355  |
| 6      | 31689  |
| 7      | 66404  |
| 8      | 91806  |
| 9      | 60780  |
| 10     | 71227  |

Dari tabel 2 dan 3, dapat dilihat ada beberapa kolom yang tidak akan digunakan dan akan lebih baik jika di hapus. Dari
tabel 4 dapat dilihat bahwa 0 ada rating artinya pengguna pernah membaca buku, tetapi tidak memberikan rating, sehingga
akan lebih baik jika rating 0 di hapus, menyisakan 1 s.d. 10.

## Data Preparation

### Langkah-langkah pra-pemrosesan data

1. Membaca _dataset_ menggunakan _pandas_
2. _Drop_ data kosong (_NaN_ / _NULL_)
3. _Drop_ kolom yang tidak akan digunakan
4. _Drop_ nilai yang _invalid_ pada kolom
5. Pemilihan _Category_
6. TF-IDF
7. _Cosine Similarity_

#### Membaca _dataset_

Pada bagian ini akan digunakan _library pandas_ untuk dapat membaca dan merubah _dataset_ menjadi _DataFrame_, dari
berkas yang diunduh akan ada 2 _file_, diproyek ini digunakan file dengan nama _Books Data with Category Language and
Summary_ yang didalam file akan ditemukan berkas csv yang datanya sudah di proses terlebih dahulu. _Sample dari
_dataset_ ini dapat dilihat pada tabel 2.

#### Drop data kosong

Pada bagian ini kita dapat melihat tabel 3 bahwa kolom `city, state, dan country` jumlahnya tidak sama dengan kolom
lainnya. Untuk proyek ini, akan digunakan fungsi `dropna()`, dengan fungsi ini jika ada data kosong dalam bagian baris
_DataFrame_ maka seluruh baris akan dihapus. Untuk melihat _count_ dari tiap kolom setelah dihapus dapat dilihat di
tabel 5.

Tabel 5. Informasi tentang data setelah _preprocess_

| # | Column      | Non-Null Count | Dtype  |
|---|-------------|----------------|--------|
| 0 | user_id     | 217314         | int64  |
| 1 | rating      | 217314         | int64  |
| 2 | book_title  | 217314         | object |
| 3 | book_author | 217314         | object |
| 4 | publisher   | 217314         | object |
| 5 | Category    | 217314         | object |

#### Drop kolom/nilai

Pada bagian ini akan di hapus kolom yang tidak akan digunakan, tujuan penghapusan ini adalah untuk mengurangi dimensi
yang dibutuhkan, berikut adalah kolom yang
dihapus `'Unnamed: 0','location','isbn', 'img_s','img_m', 'img_l', 'city','age','state','Language','country',
'year_of_publication', 'Summary'`. Selain kolom akan dihapus nilai yang tidak sesuai dengan konteks seperti pada kolom
Category `'9'` dan 0 pada rating. Hasil akhir dapat dilihat pada tabel 5.

### Pemilihan _Category_

Pada bagian ini akan dipilih dari _category_ secara spesifik, pemilihan ini dilakukan dikarenakan sumber daya yang
dimiliki untuk menghitung tidak memadai untuk menggunakan seluruh _category_. _Category_ yang akan dipakai
adalah 25 kategori pertama, yaitu antara lain `Fiction, Juvenile Fiction, Biography Autobiography, Humor, History,
Religion, Body Mind Spirit, Juvenile Nonfiction, Social Science, Business Economics, Family Relationships, Self Help,
Health Fitness, Cooking, Travel, Poetry, True Crime, Psychology, Science, Computers, Literary Criticism, Drama,
Political Science, Philosophy, Comics Graphic Novels`.

### Penggunaan TF-IDF

Metode TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah data kategori buku menjadi fitur yang
dapat digunakan oleh sistem rekomendasi berbasis konten. Proses ini merupakan bagian dari data preparation karena TF-IDF
digunakan untuk mengekstrak fitur penting dari kategori buku yang kemudian akan digunakan dalam proses pencarian
kemiripan.

Pada tahap ini, sistem rekomendasi dibangun berdasarkan _category_ yang dimiliki setiap buku. Teknik _Term
Frequency-Inverse Document Frequency_ (TF-IDF) digunakan untuk menemukan representasi fitur penting dari setiap kategori
buku. Proses ini dimulai dengan membangun matriks TF-IDF untuk masing-masing kategori buku yang ada dalam dataset.

1. **Inisialisasi dan Pelatihan TF-IDF**: Pertama, objek `TfidfVectorizer()` di inisialisasi dan dilatih `.fit()`
   menggunakan kolom
   category dari dataset book_new. Ini menghasilkan fitur-fitur yang relevan untuk setiap kategori buku yang
   diidentifikasi
   dalam dataset. Daftar fitur yang dihasilkan dapat dilihat di bawah ini:
   ```python
    array(['biography_autobiography', 'body_mind_spirit', 'business_economics', 
       'comics_graphic_novels', 'computers', 'cooking', 'drama', 
       'family_relationships', 'fiction', 'health_fitness', 'history', 
       'humor', 'juvenile_fiction', 'juvenile_nonfiction', 'literary_criticism', 
       'philosophy', 'poetry', 'political_science', 'psychology', 'religion', 
       'science', 'self_help', 'social_science', 'travel', 'true_crime'],
      dtype=object)

   ```

2. **Transformasi dan Matriks TF-IDF**: Setelah pelatihan, matriks TF-IDF dibentuk dengan mentransformasikan data
   _category_ dari dataset `book_new`. Hasil transformasi menghasilkan matriks dengan dimensi (55896, 25), yang berarti
   terdapat 55,896 buku dan 25 kategori yang di identifikasi.
3. **Visualisasi Matriks TF-IDF**: Matriks TF-IDF dapat dilihat dalam bentuk matriks padat dengan nilai-nilai yang
   menunjukkan relevansi kategori untuk setiap buku. Misalnya, beberapa baris dalam matriks tersebut mungkin tampak
   seperti berikut:
   ```text
   matrix([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [1., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]])

   ```
4. Contoh Matriks TF-IDF: Tabel berikut menunjukkan sampel dari matriks TF-IDF dengan beberapa buku dan nilai TF-IDF
   untuk kategori tertentu. Nilai 1 menunjukkan bahwa buku tersebut termasuk dalam kategori yang sesuai, sedangkan nilai
   0 menunjukkan sebaliknya.

   Tabel 6. Sample TF-IDF

   | title                                                           | health_fitness | religion | family_relationship | cooking | social_science | drama |
   |-----------------------------------------------------------------|----------------|----------|---------------------|---------|----------------|-------|
   | Brother'S Wife (The Kingsley Baby) (Harlequin Intrigue, No 458) | 0              | 0        | 0                   | 0       | 0              | 0     |
   | One Small Sparrow                                               | 1              | 0        | 0                   | 0       | 0              | 0     |
   | My Utrnost tor His Highest: The Golden Book of Oswald Chambers  | 1              | 0        | 0                   | 0       | 0              | 0     |

   Pada Tabel 6, dapat dilihat bahwa buku dengan judul "One Small Sparrow" dan "My Utmost for His Highest" termasuk
   dalam kategori health_fitness, dengan nilai 1 pada kolom tersebut, sementara kolom kategori lainnya memiliki nilai 0.

   Dengan menggunakan matriks TF-IDF ini, sistem rekomendasi dapat lebih baik dalam memahami dan merekomendasikan buku
   berdasarkan kategori yang relevan.

### _Cosine Similarity_

Setelah data diubah menjadi vektor TF-IDF, kemiripan antara buku dihitung menggunakan Cosine Similarity. Hasil ini
kemudian digunakan untuk mencari buku-buku yang paling mirip satu sama lain berdasarkan kategori mereka.

Pada tahap sebelumnya, telah berhasil mengidentifikasi korelasi antara nama buku dengan kategorinya. Sekarang, akan
dihitung derajat kesamaan (_similarity degree_) antar nama buku dengan menggunakan teknik _cosine similarity_. _Sample_
dari hasil dapat dilihat pada tabel 7.

Tabel 7. _Sample Cosine Similarity_

| title                                  | The Reluctant Suitor | Russia House | One Palestine, Complete: Jews and Arabs Under the British Mandate | The Time Machine and the Invisible Man | Good Intentions |
|----------------------------------------|----------------------|--------------|-------------------------------------------------------------------|----------------------------------------|-----------------|
| Because a Little Bug Went Ka-Choo!     | 0                    | 0            | 0                                                                 | 0                                      | 0               |
| LIE TO ME                              | 1                    | 1            | 0                                                                 | 1                                      | 1               |
| Computers in Your Future (4th Edition) | 0                    | 0            | 0                                                                 | 0                                      | 0               |

Pada Tabel 7, dapat diidentifikasikan kesamaan antara satu nama buku dengan nama buku lainnya. Misalnya, pada tabel
tersebut, buku "_LIE TO ME_" memiliki nilai kesamaan 1 dengan buku "_The Reluctant Suitor_," "_Russia House_," dan "_The
Time
Machine and the Invisible Man._" Ini menunjukkan bahwa buku "_LIE TO ME_" memiliki tingkat kemiripan yang tinggi dengan
ketiga buku tersebut berdasarkan data yang ada.

Di tahap sebelumnya, data sudah di-vektorisasi dan dicari similarity degree. Untuk mengetahui seberapa baik model dalam
memberikan sebuah rekomendasi, dapat dibuat sebuah fungsi yang akan menerima parameter `nama_buku`, `similarity_data`,
`items`, dan `k` dengan definisi masing-masing parameter sebagai berikut:

- `nama_buku`: Nama buku yang akan dicari rekomendasinya.
- `similarity_data`: DataFrame mengenai similarity yang telah dibuat di tahap sebelumnya.
- `items`: Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah `title` dan `category`.
- `k`: Jumlah rekomendasi yang ingin diberikan.

berdasarkan parameter di atas, maka langkah-langkah penjelasan untuk sistem rekomendasi ini adalah sebagai berikut :

1. **Mengambil index kemiripan**
   ```python
   index = similarity_data.loc[:, book_name].to_numpy().argpartition(range(-1, -k, -1))
   ```
    - Mengambil data kemiripan untuk book_name dari similarity_data dan mengubahnya menjadi array NumPy.
    - Menggunakan argpartition untuk menemukan indeks dengan nilai kemiripan tertinggi tanpa mengurutkan seluruh array.
      `range(-1, -k, -1)` digunakan untuk mendapatkan `k` indeks teratas.

2. **Mengambil Buku dengan Kemiripan Terbesar**
   ```python
   closest = similarity_data.columns[index[-1:-(k+2):-1]]
   ```
    - Mengambil nama-nama buku dengan kemiripan terbesar dari indeks yang diperoleh pada langkah sebelumnya.
    - `index[-1:-(k+2):-1]` memilih `k` buku teratas.

3. **Menghapus Buku yang Dicari dari Daftar Rekomendasi**
   ```python
   closest = closest.drop(book_name, errors='ignore')
   ```
    - Menghapus nama buku yang dicari dari daftar hasil rekomendasi untuk memastikan buku tersebut tidak muncul dalam
      hasil.

4. **Menggabungkan, Menyaring Data, dan Mengembalikan**
   ```python
   df = pd.DataFrame(closest).merge(items)
   df.drop_duplicates(keep='first', subset="title", inplace=True)
   return df.head(k)
   ```
    - Mengubah daftar buku yang paling mirip menjadi DataFrame dan menggabungkannya dengan DataFrame `items` untuk
      mendapatkan informasi tambahan seperti kategori.
    - Menghapus duplikat berdasarkan kolom `title` untuk memastikan setiap judul buku hanya muncul sekali.
    - Mengemabalikan hasil yang sudah didapat dengan sejumlah `k`.

Setelah proses diatas selesai maka hasil yang di dapat dari Rekomendasi untuk nama
buku `Macromedia Flash MX for Dummies` dapat di jelaskan pada tabel 8 berikut.

Tabel 8. Hasil Rekomendasi.

| # | title                                             | category  |
|---|---------------------------------------------------|-----------|
| 0 | DocBook: The Definitive Guide (O'Reilly XML)      | Computers |
| 1 | XML in a Nutshell : A Desktop Quick Reference ... | Computers |
| 2 | JavaScript Bible, Gold Edition                    | Computers |
| 3 | Sams Teach Yourself GIMP in 24 Hours              | Computers |
| 4 | Crypto: How the Code Rebels Beat the Governmen... | Computers |
| 5 | ASP in a Nutshell, 2nd Edition                    | Computers |
| 6 | Palm Computing for Dummies                        | Computers |
| 7 | Programming the Perl DBI                          | Computers |
| 8 | Windows 2000: The Complete Reference              | Computers |
| 9 | Using Excel Visual Basic for Applications, Spe... | Computers |

## Evaluasi

Pada tahap ini, Precision digunakan untuk mengevaluasi hasil dari rekomendasi pada Tabel 8. Precision dapat
didefinisikan sebagai berikut:

**Precision** = $\frac{r}{i}\$

- **r** = Total rekomendasi yang relevan
- **i** = Jumlah rekomendasi yang diberikan

Dari hasil rekomendasi di Tabel 8, diketahui bahwa buku *Macromedia Flash MX for Dummies* termasuk dalam kategori *
*Computers**. Dari 10 item yang direkomendasikan, semua 10 item juga termasuk dalam kategori **Computers** (similar).
Artinya, precision sistem adalah \( \frac{10}{10} \) atau **100%**.

Dari keseluruhan poryek yang sudah dikerjakan oleh penulis, maka untuk menjawab pertanyaan
dari [problem statements](#problem-statements) Adalah sebagai menggunakan sistem rekomendasi _Content-Based
Collaborative Filtering_ dengan menggunakan TF-IDF dan _Cosine Similarity_. Dengan tahapan yang sudah dijelaskan pada
Bab-Bab sebelumnya, mulai dari [Data Understanding](#data-understanding) hingga [Modeling](#modeling). Goals pun sudah
dicapai dengan contoh hasil seperti pada tabel. 8 yang sudah di jabarkan sebelumnya oleh penulis. rekomendasi yang di
berikan ini berdasarkan category, jadi sistem rekomendasi ini berdasarkan cateogry yang munkgin di sukai oleh user
berdasarkan buku yang sudah di baca sebelumnya.

## Referensi

[[1]](https://media.neliti.com/media/publications/96720-ID-rumah-baca-jendela-dunia-sebuah-model-pe.pdf) Gresi A.R.,
Alan N., Khasanah B.R., Robby A.S., Priyadi N.P. (2013). Rumah Baca Jendela Dunia, Sebuah Model Perpustakaan Panti
Asuhan. Jurnal Ilmiah Mahasiswa, Vol. 3
No.2. https://media.neliti.com/media/publications/96720-ID-rumah-baca-jendela-dunia-sebuah-model-pe.pdf