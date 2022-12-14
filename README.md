# **Prediksi Conversion Rate Menggunakan Machine Learning**

## **Background**

Perusahaan Leslar memiliki conversion rate yang tinggi diangka 15% pada tahun pertama. Namun,
Perusahaan Leslar memiliki masalah untuk mengetahui pola perilaku dari visitor yang berpotensi
untuk membeli produk perusahaan. Jadi, Perusahaan meminta Tim Data Scientist untuk
membuatkan model yang bisa memprediksi apakah visitor akan membeli produk atau tidak
untuk digunakan di tahun berikutnya.

## **Dataset**

Dataset yang digunakan dalam project ini adalah dataset 'online-shoppers-intention'. Dataset memiliki 12330 records dan 18 kolom. Kolom target yang kami gunakan adalah kolom Revenue.

Dataset dapat diakses [disini](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset?resource=download).

## **Prerequisites**

- Numpy version: `1.21.5`
- Pandas version: `1.4.2`
- Matplotlib version: `3.5.1`
- Seaborn version: `0.11.2`
- Sklearn version: `1.0.2`
- Xgboost version: `0.90`
- Shap version: `0.41.0`

## **Exploratory Data Analysis (EDA)**

Tahap EDA bertujuan untuk menggali insight dan anomali pada data. Adapun hasil EDA yang dilakukan adalah sebagai berikut.

### **Descriptive Statistics**

Analisis statistik deskriptif dilakukan terhadap fitur-fitu numerik (mean, median, min, max, dan kuartil) dan fitur kategorik (number of unique, mode). Hasil analisi statistik deskriptif untuk fitur-fitur numerik adalah sebagai berikut.

1.	Semua fitur numerik (Andiminstrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, dan SpecialDay) memiliki nilai mean dan median yang berbeda. Kemungkinan fitur-fitur tersebut memiliki outlier atau tidak menyebar normal (skewed).
2.	Semua fitur yang disebutkan pada poin 1 juga memiliki nilai maksimum yang cukup jauh dari nilai Q3 yang mengindikasikan bahwa fitur-fitur tersebut memiliki outlier.
3.	Beberapa fitur (Informational, Informational_Duration, PageValues, SpecialDay) terlihat didominasi oleh nilai 0 dan memiliki nilai maksimum yang cukup besar. Indikasi kuat bahwa fitur-fitur tersebut memiliki outlier.

Sedangkan hasil analisi statistik deskriptif untuk fitur-fitur kategorik adalah sebagai berikut.

1.	Kolom target (Revenue) memiliki 2 unique value namun sebagian besar (sekitar 85%) bernilai False. Bisa dipertimbangkan untuk penanganan imbalance.
2.	Beberapa fitur kategorik yaitu Browser, Region, dan TrafficType memiliki terlalu banyak unique value. Beberapa value minor mungkin bisa dikategorikan sebagai Other.
3.	Terdapat fitur-fitur yang didominasi oleh satu unique value saja seperti OperatingSystems, Browser, VisitorType, dan Weekend. Fitur-fitur tersebut mungkin bisa tidak disertakan dalam model.

### **Univariate Analysis**

Analisis univariat bertujuan untuk melihat lebih jelas distribusi data masing-masing fitur. Analisis univariat juga berguna untuk melihat outlier secara lebih jelas. Analisis univariat dilakukan dengan bantuan visualisasi boxplot, displot, dan bar chart. Hasil analisis yang dilakukan adalah sebagai berikut.

#### **Boxplot (fitur numerik)**

Selain untuk melihat bentuk sebaran/kepadatan data, fungsi lain boxplot ialah untuk melihat outliers secara lebih teliti untuk itu, meskipun di kolom Informational, Informational_Duration, PageValues, SpecialDay distribusinya sangat tipis karena sebaran data nya bersifat menumpuk, namun di kolom2 tersebut memiliki nilai outliers yang cukup banyak.

-	Administrative = kebanyakan pengguna mengunjungi page ini kurang dari 5 kali, bahkan cenderung jarang yang mengunjungi (nol kali).
-	Administrative_Duration = pengguna yang banyak menghabiskan waktu mengunjungi page sekitar dibawah 250 detik.
-	Informational= kebanyakan pengguna sangat jarang mengunjungi page ini, jumlah kunjungan sangat dominan di nol (itu sebabnya saat diplot, distribusinya terlihat tipis dan menumpuk di nol).
-	Informational_Duration= durasi pengguna yang mengunjungi page ini sangat dominan di nol detik.
-	ProductRelated = pengguna banyak mengunjungi web sekitar di bawah 50 page
-	ProductRelated_Duration = pengguna yang banyak menghabiskan waktu mengunjungi page sekitar dibawah 5000 detik
-	BounceRates = persentase pengguna banyak mengunjungi web dibawah 0.025
-	ExitRates = persentase pengguna banyak mengunjungi web diantara 0.020 sampai 0.050

#### **Displot (fitur numerik)**

semua distribusinya menghasilkan positive skewed dan terdapat beberapa lonjakan kecil seperti pada 'Informational', 'BounceRates', 'ExitRates', 'SpecialDay' dan juga terdapat long tail pada 'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'PageValues'. Dari plot distribusi, dapat disimpulkan diantara ketiga page di web, pengguna lebih sering mengunjungi page productrelated ketimbang 2 page lainnya. Hal ini bisa dilihat pada range sebaran jumlah kunjungan (x axis) pada page ProductRelated yang sebarannya mencapai ratusan.

#### **Bar Chart (fitur kategorik)**

-	Month: dalam data ini satu tahun hanya terdapat 10 bulan dimana pada bulan januari dan april itu tidak masuk.
Pengguna yang mengunjungi web banyak terdapat pada bulan May dan November dan pengguna yang sepi mengunjungi web terdapat pada bulan February 
-	VisitorType: terdapat 3 tipe pengunjung dimana yang paling mendominasi yaitu pada Returning_visitor yang lebih dari 10000 pngunjung, New_visitor terdapat kurang dari 2000, dan Other yang paling minimum
-	Weekend: pengguna yang mengunjungi dalam weekend terlihat jauh lebih sedikit yaitu sekitar 3000 dibandingkan yang tidak mengunjungi mencapai lebih dari 8000
-	Revenue: pengguna yang Purchase jauh lebih sedikit yaitu sekitar 2000 dibandingkan dengan pengguna yang NotPurchase mencapai lebih dari 10000 
-	OperatingSystems: pada distribusinya terdapat 3 top OperatingSystem yaitu pada tipe 1,2,dan 3
-	Browser: pada distribusinya paling dominan banyak menggunakan tipe 2 
-	Region: pada distribustinya paling banyak menggunakan di tipe Region 1 
-	TrafficType: pada distribusinya paling banyak menggunakan tipe TrafficType 2

Hal yang harus di follow-up untuk data pre-processing dikarenakan pada data ada yang right skewed dan mempunyai long tail maka kita perlu menggunakan log Transformasi agar hasil distribusi mendekati distribusi normal.

### **Multivariate Analysis**

Multivariate analysis lebih bertujuan untuk melihat hubungan-hubungan antar fitur maupun hubungan antara fitur dengan variabel target (Revenue). Berikut adalah hasil analisis korelasi yang diperoleh menggunakan heatmap:

1.	Page value memiliki korelas positif yang tinggi terhadap target. Semakin tinggi page value, semakin tinggi juga kemungkinan visitor untuk purchase.
2.	Bounce rate dan exit rate memiliki korelasi negatif terhadap target yang artinya semakin tinggi bouncerate dan exitrate semakin kecil juga kemungkinan visitor untuk purchase.
3.	Terdapat beberapa fitur yang berpotensi redundan, (bouncerate-exitrate) memiliki nilai korelasi 0.91 dan fitur (Administrative,Informaional danProduct related terhadap Duration ) memiliki nilai korelasi dalam range 0.6 – 0.86.

**Notes** : dikarenakan tipe data target adalah boolean atau bukan numerical, kita belum bisa memvalidasi keterhubungan fitur fitur dengan target dengan lebih jelas.

Selanjutnya dilakukan analysis hubungan fitur categorical terhadap target dengan membuat visual countplot  untuk melihat lebih jelas hubungan target yang merupakan tipe categorical terhadap fitur categorical. Berikut adalah hasil yang didapatkan:
1.	Segmentasi berdasarkan tipe visitor, returning_visitor merupakan tipe yang paling banyak berkunjung. Teteapi, new_visitor memiliki presentasi lebih besar dalam melakukan purchasing jika dilihat berdasarkan jumlahnya pada masing masing tipe.
2.	Ada beberapa tipe traffic yang mendominasi dari jumlah sampai kemngkinan melakukan pembelian.
3.	Operating system dan browser hanya didominasi 2 sampai 3 tipe saja.
4.	Jumlah visitor berdasarkan region sangat berbeda jauh dengan dimana hanya didominasi 3 daerah saja.

Untuk mendapatkan insight lebih mengenai hubungan target dan feature dibutuhkan penanganan outlier atau melakukan transformasi. Kita juga dapet merubah kolom yang merupakan categorical menjadi numerical untuk mendapatkan insight dan hubungan yang lebih jelas.

### **Business Insight**

Dalam menemukan business insight, hal yang perlu dilakukan yaitu menganalisis fitur-fitur  dari dataset. Fitur-fitur yang digunakan dalam menganalisis dataset adalah Month, VisitorType, SpecialDay, dan Revenue. Berikut adalah beberapa hasil yang diperoleh dari analisis fitur-fitur tersebut.

1.	Melihat pertumbuhan penjualan produk per bulan di tahun pertama.
2.	Melihat banyaknya penjualan produk berdasarkan tipe visitor, yaitu new_visitor, returning_visitor, dan other.
3.	Melihat banyaknya penjualan pada hari-hari istimewa.

Dari hasil analisis tersebut diperoleh beberapa business insight sebagai berikut.
<ol>
    <li>Pertumbuhan penjualan produk di tahun pertama masih fluktuatif atau tidak stabil.</li>
        <ul>
            <li>Pada bulan Mei - Jun, penjualan produk mengalami penurunan.</li> 
            <li>Pada bulan Jun - Nov, penjualan produk mengalami peningkatan.</li>
            <li>Pada bulan Nov - Des, penjualan produk mengalami penurunan.</li>
        </ul>
    <li>Sebagian besar penjualan produk per bulan masih berada di bawah rata-rata (rata-rata 191 transaksi per bulan).</li>
    <li>Persentase returning visitor yang membeli produk sebesar 14% lebih rendah dibandingkan new visitor sebesar 25%.</li>
    <li>Penjualan produk pada hari-hari istimewa masih rendah.</li>
</ol>

Berdasarkan business insight di atas, terdapat beberapa rekomendasi yang dapat dilakukan oleh perusahaan untuk meningkatkan penjualan, antara lain: 

<ol>
    <li>Untuk meningkatkan penjualan terhadap new visitor maupun returning visitor, Tim Business dan Tim Marketing dapat melakukan hal-hal berikut: </li>
    <ul>
        <li>melakukan riset terhadap produk apa sajakah yang diinginkan oleh customer dengan menggunakan platform media sosial (ex: Instagram) atau tools riset (ex: Google Analytics).</li> 
        <li>membuat iklan sebagai media promosi perusahaan dengan menggunakan Google Ads atau mengkampanyekan iklan melalui media sosial.</li>
        <li>menyusun strategi content marketing yang menarik perhatian customer dengan menambahkan blog ke website perusahaan. </li>
    </ul>
    <li>Memberikan pelayanan terbaik kepada customer dan membuat program-program loyality customer secara periodik agar returning visitor banyak melakukan pembelian produk.</li>
    <li>Memanfaatkan pemasaran produk di hari-hari istimewa, seperti Valentine's Day, Mother's Day, Christmas & New Year, dengan memberikan promo/diskon atau special offer kepada customer untuk meningkatkan penjualan di hari-hari istimewa.</li>
</ol>

## **Preprocessing**

Tahap preprocessing terdiri dari dua bagian yaitu Data Cleansing dan Feature Engineering.

### **Data Cleansing**

#### **Missing Values dan Duplicated Rows**
Dataset yang kami gunakan tidak memiliki missing value, sehingga tidak perlu penanganan apapun terhadap missing value. Dataset memiliki 125 baris duplikat. Baris-baris tersebut telah dihapus pada tahap ini.

#### **Handling Outliers**
Tahap awal penanganan outlier dimulai dengan mencoba menghilangkan outlier pada fitur-fitur numerik. Namun setelah outlier dihapus, ternyata banyak record untuk target (1) yang hilang. Kami memutuskan untuk mempertahankan outlier karena melihat indikasi outlier-outlier tersebut merupakan pattern pada data.

#### **Feature Encoding**
Proses feature encoding ini bertujuan untuk mengubah fitur-fitur
kategorik menjadi angka (numerik) agar bisa dimengerti oleh model. Fitur-fitur yang diencoding adalah Month, VisitorType, OperatingSystems, Browser, Region, Weekend, Revenue.

#### **Feature Transformation**
Karena fitur-fitur numerik pada dataset sebagian besar tidak berdistribusi normal, pada tahap ini dilakukan transformasi Yeo Johnson (transformasi logaritma tidak dipilih karena semua fotur mengandung 0). Kemudian dilakukan feature scaling menggunakan MinMaxScaler dari Scikit Learn.

#### **Imbalance Handling**
Perbandingan banyaknya records target 0 dengan 1 adalah 80:15, untuk itu dilakukan sedikit over sampling sehingga perbandingannya menjadi 2:1 agar targe 1 tidak dianggap tidak penting oleh model dan juga tidak overfitting.

### **Feature Engneering**
#### **Feature Selection**
Tahap fitur selection bertujuan untuk membuang fitur-fitur yang tidak relevan ataupun yang redundan. Fitur-fitur yan dibuang pada tahap ini adalah ProductRelated_Duration, BounceRates, dan VisitorType.

#### **Feature Extraction**
Feature extraction bertujuan untuk membuat fitur baru dari fitur-fitur yang sudah ada. Namun, dalam project ini kami tidak melakukan feature extraction.

#### **Additional Features**
Berikut rekomendasi fitur-fitur yang jika tersedia kami yakin dapat membantu melakukan prediksi.

- Promotion Day (apakah sesi dilakukan pada saat hari-hari promosi seperti tanggal cantik atau yang lainnya)
- Internet Speed (ini berguna untuk menentukan apakah perlu
menyesuaikan tampilan website untuk visitor dengan internet low speed, karena laman yang lama dimuat cenderung membuat visitor meninggalkan laman dengan cepat)
- ProductReview (berapa sering dan berapa lama visitor melihat laman review product)
- DeviceType (jenis device yang digunakan)

## **Modeling**

Pada tahap ini kami mencoba beberapa algoritma klasifikasi yang mungkin digunakan untuk dataset yang dimiliki. Beberapa
algoritma yang kami coba adalah Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, 
Support Vector Machine, K-Nearest Neighbors, Ada Boost, dan XG Boost.

Dari semua algoritma yang dicoba, dipilih XG Boost sebagai model final yang akan diimplementasikan, model ini dilatih menggunakan hyperparameter default tanpa tuning dengan skor AUC sebesar 91%. Model dengan AUC sebesar itu sudah  cukup untuk digunakan memprediksi apakah customer akan melakukan purchase atau tidak berdasarkan data yang masuk di masa mendatang. Hasil pediksi dari model ini akan digunakan untuk membantu memutuskan strategi marketing yang cocok.

### **Fitur Terpenting**

Dari model XG Boost yang telah dilatih, didapatkan 4 fitur terpenting yang paling berpengaruh dalam model yaitu PageValues, ExitRates, Administrative, dan ProductRelated.

### **Business Insight dari Model**

Berdasarkan fitur-fitur terpenting di atas, kami dapatkan insight sebagai berikut.

1. Jika nilai Page Values semakin tinggi, maka kemungkinan pengunjung untuk purchase juga semakin tinggi.
2. Semakin tinggi exit rate, maka kemungkinan untuk purchase semakin rendah.
3. Semakin sedikit page Administrative yang dikunjungi, maka semakin tinggi kemungkinan pengunjung untuk purchase.
4. Semakin banyak page Product Related yang dikunjungi, maka kemungkinan pengunjung untuk purchase semakin tinggi.

### **Action Items**

Berdasarkan insight yang kami temukan, kami merekomendasikan beberapa action items yang mungkin dapat membantu 
bisnis yaitu sebagai berikut.

1. Karena page dengan value yang tinggi akan leading ke purchase, pilih page dengan values tinggi untuk marketing campaign sesuai dengan target visitor.
2. Lakukan optimasi desain UI/UX untuk menurunkan exit rate dan bounce rate.
3. Kurangi tindakan berlebih yang membuat visitor sering membuka page administrative. Gunakan penempatan pop up page administrative yang sesuai (misal akhir sesi).
4. Karena page Product Related leading ke purchase, maka perlu optimalisasi agar rekomendasi produk yang diberikan akurat sesuai dengan keinginan customer/visitor.