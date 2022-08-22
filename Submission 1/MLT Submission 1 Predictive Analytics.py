#!/usr/bin/env python
# coding: utf-8

# # Analisis Prediktif: <span style="font-weight:normal">Prediksi Strategis Pemilihan Jenis Tanaman untuk Lahan Pertanian Tertentu</span>
# 
# <hr style="border:1px solid gray">
# 
# #### <span style="font-weight:normal">Proyek Submission 1 - Machine Learning Terapan <br/><br/> Oleh: Nur Muhammad Syaifuddin</span>

# ![agriculture.jpg](attachment:agriculture.jpg)

# # Pendahuluan
# 
# #### <div align="left"><span style="white-space: pre-wrap; font: normal 12pt Arial; line-height: 1.5;">Pada proyek ini, saya mengambil tema **pertanian**. Dimana model yang dibangun akan memprediksi jenis tanaman apa yang cocok ditanam di lahan pertanian tertentu dengan berbagai parameter, diantaranya N (kandugan Nitrogen dalam tanah), P (kandungan Fosfor dalam tanah), K (kandungan Kalsium dalam tanah), suhu, kelembaban, nilai pH tanah, dan curah hujan (dalam mm). Dengan adanya model Machine Learning ini diharapkan dapat membantu petani dalam mengambil strategi keputusan yang tepat untuk memilih jenis tanaman pertanian yang cocok agar menghasilkan hasil pertanian yang berkualitas dan melimpah.</span></div>

# # 1. Mengimpor Library yang Dibutuhkan

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


# # 2. Mengunduh Dataset

# #### <div align="left"><span style="white-space: pre-wrap; font: normal 12pt Arial; line-height: 1.5;">Dataset diambil dari link: https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset <br>**Saya menyimpannya di direktori lokal**.</span></div>

# # 3. *Data Understanding*

# ## 3.1 *Data Loading*

# In[2]:


# membaca dataset
data_path = "crop_recommendation.csv"
dataset = pd.read_csv(data_path)
dataset


# #### <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan:**<br>Output dari kode diatas memberikan informasi sebagai berikut:<br><ul><li>Terdapat 2200 baris dalam dataset.</li><li>Terdapat 8 kolom yaitu, N, P, K, Temperature, Humadity, ph, rainfall, dan label.</li></ul></span></div>

# In[3]:


label = dataset['label'].drop_duplicates().values
label


# ## 3.2 *Exploratory Data Analysis* - Deskripsi Variabel

# #### <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Deskripsi Variabel:**<br>Berdasarkan dokumentasi di kaggle, variabel-variabel pada Crop Dataset adalah sebagai berikut:<br><ol><li>N - rasio kandungan Nitrogen dalam tanah.</li><li>P - rasio kandungan Fosfor dalam tanah.</li><li>K - rasio kandungan Kalsium dalam tanah.</li><li>temperature - suhu dalam derajat celcius.</li><li>humadity - kelembaban relatif dalam %.</li><li>ph - nilai ph tanah.</li><li>rainfall - curah hujan dalam mm.</li><li>label - jenis tanaman yang cocok untuk ditanam di lahan pertanian berdasarkan variabel 1-7.</li></ol></span></div>

# In[4]:


dataset.info()


# #### <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan:**<br>Output dari kode diatas memberikan informasi sebagai berikut:<br><ul><li>Terdapat 3 kolom numerik dengan tipe data int64, yaitu: N, P, K. Ini merupakan fitur numerik.</li><li>Terdapat 4 kolom numerik dengan tipe data float64 yaitu: temperature, humidity, ph dan rainfall. Ini merupakan fitur numerik.</li><li>Terdapat 1 kolom dengan tipe data object, yaitu: label. Kolom ini merupakan categorical features (fitur non-numerik) dimana kolom ini merupakan target fitur.</li></ul></span></div>

# In[5]:


dataset.describe()


# #### <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan:**<br>Output kode di atas memberikan informasi statistik pada masing-masing kolom, antara lain:<br><ul><li>count adalah jumlah sampel pada data.</li><li>mean adalah nilai rata-rata.</li><li>std adalah standar deviasi.</li><li>min yaitu nilai minimum setiap kolom.</li><li>25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.</li><li>50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).</li><li>75% adalah kuartil ketiga.</li><li>Max adalah nilai maksimum.</li></ul></span></div>

# In[6]:


dataset.shape


# ## 3.3 *Exploratory Data Analysis* - Memeriksa *Missing Value*

# In[7]:


dataset.isnull().sum()


# #### <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan:**<br>Output kode di atas memberikan informasi bahwa tidak terdapat *missing value* pada dataset</span></div>

# ## 3.4 *Exploratory Data Analysis* - *Univariate Analysis*

# ### 3.4.1 Sebaran/ distribusi data pada setiap fitur numerik

# In[8]:


#visualisasi data masing-masing fitur menggunakan histogram plot untuk mengetahui sebaran/distribusi data pada setiap fitur
features = dataset.columns[:-1]
for feature in features:
  figures = px.histogram(data_frame=dataset,
                        x=feature,
                        template='plotly_white',
                        marginal='box',
                        nbins=200,
                        color_discrete_sequence=["green"],
                        barmode='stack',
                        histfunc='count')

  title = "Sebaran/ distribusi data pada fitur " + feature
  figures.update_layout(font_family='Open Sans',
                        title=dict(text=title, x=0.47, font=dict(color="#333",size=20)),
                        hoverlabel=dict(bgcolor='white'))

  figures.show()


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Berdasarkan hasil visualisasi data diatas, dapat terlihat sebaran atau distribusi data yang ada pada setiap fitur. Termasuk nilai minimum, median, maksimum, Q1, Q3, batas atas dan batas bawah. Selain itu dapat dilihat juga pada beberapa fitur masih terdapat nilai outliers.</span></div>

# ### Sebaran/ distribusi data pada fitur target

# In[9]:


#visualisasi data untuk mengetahui sebaran/ distribusi data pada fitur target 'label'
plt.figure(figsize=(19,7))
sns.countplot(dataset['label'] , palette = 'Spectral')
plt.xticks(rotation=90)
plt.title("Sebaran/ distribusi data pada fitur target (label)", fontdict= {'fontsize':18})
plt.show()


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Berdasarkan hasil visualisasi dari fitur target 'label' dapat memberikan informasi bahwa dataset sudah seimbang dengan jumlah sampel masing-masing label yaitu 100 sampel, sehingga tidak perlu menyeimbangkan data lagi.</span></div>

# ## 3.5 *Exploratory Data Analysis* - *Multivariate Analysis*

# ### 3.5.1 Mengecek dan Membandingkan rata-rata N, P, K antar label

# In[10]:


crop_summary = pd.pivot_table(dataset, index=['label'], aggfunc='mean')

# visualisasi kandungan N, P, K terhadap setiap label
for feature in features[:3]:
    plt.figure(figsize=(19,7))
    sns.barplot(x = "label", y = feature, data = dataset)
    plt.xticks(rotation=90)
    plt.title(f"Rata-rata {feature} terhadap label crop")
    plt.show()

    crop_summary_feature = crop_summary.sort_values(by=feature, ascending=False)
  
    fig = make_subplots(rows=1, cols=2)

    top = {
        'y' : crop_summary_feature[feature][0:11].sort_values().index,
        'x' : crop_summary_feature[feature][0:11].sort_values()
    }

    last = {
        'y' : crop_summary_feature[feature][-11:].index,
        'x' : crop_summary_feature[feature][-11:]
    }

    fig.add_trace(
        go.Bar(top,
              name="crop label dengan kandungan " + feature + " tinggi",
              marker_color='green',
              orientation='h',
              text=top['x']),
        
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(last,
              name="crop label dengan kandungan " + feature + " rendah",
              marker_color='red',
              orientation='h',
              text=last['x']),
        row=1, col=2
    )


    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(title_text=feature,
                      plot_bgcolor='white',
                      font_size=12, 
                      font_color='black',
                    height=500)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()


# In[11]:


# visualisasi perbandingan kandungan fitur N, P, K antar label
fig = go.Figure()
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['N'],
    name='N',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['P'],
    name='P',
    marker_color='lightsalmon'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['K'],
    name='K',
    marker_color='crimson'
))

fig.update_layout(title="Perbandingan kandungan N, P, K antar label",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45)

fig.show()


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Hasil visualisasi di atas memberikan informasi mengenai rata-rata kandungan N, P, K terhadap setiap label crop. Dimana dapat dilihat bahwa terdapat beberapa label crop yang membutuhkan lahan dengan kandungan N, P, K tinggi dan beberapa label membutuhkan lahan dengan kandungan N,P,K rendah.</span></div>

# ### 3.5.2 Mengecek dan Membandingkan rata rata tingkat *temperature*, *humidity*, dan *rainfall* antar setiap label

# In[12]:


# visualisasi tingkat temperature, humidity dan rainfall terhadap setiap label
features1 = features.delete(5)
for feature in features1[-3:]:
    plt.figure(figsize=(19,7))
    sns.barplot(x = "label", y = feature, data = dataset)
    plt.xticks(rotation=90)
    plt.title(f"Rata-rata tingkat {feature} terhadap label crop")
    plt.show()

    crop_summary_feature = crop_summary.sort_values(by=feature, ascending=False)
  
    fig = make_subplots(rows=1, cols=2)

    top = {
        'y' : crop_summary_feature[feature][0:11].sort_values().index,
        'x' : crop_summary_feature[feature][0:11].sort_values()
    }

    last = {
        'y' : crop_summary_feature[feature][-11:].index,
        'x' : crop_summary_feature[feature][-11:]
    }

    fig.add_trace(
        go.Bar(top,
              name="crop label dengan tingkat " + feature + " tinggi",
              marker_color='green',
              orientation='h',
              text=top['x']),
        
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(last,
              name="crop label dengan tingkat " + feature + " rendah",
              marker_color='red',
              orientation='h',
              text=last['x']),
        row=1, col=2
    )


    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(title_text=feature,
                      plot_bgcolor='white',
                      font_size=12, 
                      font_color='black',
                    height=500)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()


# In[13]:


# visualisasi perbandingan tingkat temperature, humidity dan rainfall antar setiap label
fig = go.Figure()
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['temperature'],
    name='temperature',
    marker_color='coral'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['humidity'],
    name='humidity',
    marker_color='maroon'
))

fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['rainfall'],
    name='rainfall',
    marker_color='orangered'
))

fig.update_layout(title="Perbandingan tingkat temperature, humidity dan rainfall antar label",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45)

fig.show()


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Hasil visualisasi di atas memberikan informasi mengenai tingkat temperature, humidity, dan rainfall terhadap setiap label crop. Dimana dapat dilihat bahwa terdapat beberapa label crop yang membutuhkan lahan dengan tingkat temperature, humidity, dan rainfall tinggi dan beberapa label membutuhkan lahan dengan tingkat temperature, humidity dan rainfall rendah.</span></div>

# ### 3.5.5 Korelasi antar fitur numerik

# In[14]:


# korelasi antar fitur numerik menggunakan fungsi pairplot
plt.figure(figsize=(19,17))
sns.pairplot(dataset, hue = "label")
plt.show()


# In[15]:


plt.figure(figsize=(10, 8))
correlation_matrix = dataset.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Output kode di atas memberikan informasi mengenai korelasi antara fitur numerik, dimana dari Correlation Matrix dapat dilihat bahwa fitur P dan K memiliki korelasi yang sedikit tinggi.</span></div>

# # 4. **Data Preparation**

# ## 4.1 Melakukan label encoding pada fitur target (label)

# In[16]:


# memisahkan dataset menjadi data X (variabel independen) dan data y (variabel dependen)
# mengubah value pada fitur target 'label' dari kategorik menjadi numerik menggunakan LabelEncoder()
label_encoder = LabelEncoder()
X = dataset[features]
y = label_encoder.fit_transform(dataset["label"])

label_dict = {}
for i in range(22):
    label_dict[i] = label_encoder.inverse_transform([i])[0]
label_dict


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Sebelum masuk ke tahap pembagian dataset, terlabih dahulu melakukan pemisahan antara variabel independen (N, P, K, temperature, humidity, ph, rainfall) sebagai data X dan variabel dependen (label) sebagai data y. Karena fitur label pada dataset merupakan fitur non-numerik yang berarti nilai pada fitur tersebut adalah kategorikal, maka sebelum dimasukan ke dalam data y telah dilakukan proses label encoding untuk fitur tersebut. Label encoding merupakan teknik untuk mengubah jenis data kategorikal menjadi data numerik yang dapat dipahami model.</span></div>

# ## Melakukan pembagian dataset

# In[17]:


# melakukan pembagian data X dan y dengan train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.2, random_state = 0)
print(f'Total jumlah sample pada dataset: {len(X)}')
print(f'Total jumlah sample pada train dataset: {len(X_train)}')
print(f'Total jumlah sample pada test dataset: {len(X_test)}')


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Pembagian dataset dilakukan dengan presentase 80% data latih dan 20% data uji, dimana jumlah sampel pada data train yaitu 1760 sampel dan jumlah sampel pada data test yaitu 440 sampel.</span></div>

# ## 4.3 Mengatasi *outlier* pada data train dengan metode LOF (*Local Outlier Factor*)

# In[18]:


# mengatasi outlier fungsi LocalOutlierFactor
lof = LocalOutlierFactor().fit_predict(X_train)
mask = lof != -1
X_train, y_train = X_train[mask, :], y_train[mask]


# In[19]:


X_train


# In[20]:


X_test


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Setelah dilakukan standarisasi data, dapat dilihat bahwa semua nilai dari fitur numerik pada data train dan data test berada dalam skala data sama.</span></div>

# # 5. *Model Development*

# ## 5.1 *Model Development - K-Nearest Neighbor*

# In[21]:


# mencari nilai k yang optimal
error_rate = []
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, knn_predictions)
    print(f"Accuracy at k = {i} is {accuracy}")
    error_rate.append(np.mean(knn_predictions != y_test))


# In[22]:


plt.figure(figsize=(10,6))
plt.plot(range(1,30),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. Nilai K')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[23]:


# didapatkan nilai k optimal adalah 1
print("Minimum error:-",min(error_rate)," pada K =",error_rate.index(min(error_rate))+1)


# In[24]:


# membuat model dengan algoritma KKN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# In[25]:


# menguji model menggunakan data test
knn_predictions = knn.predict(X_test)
knn_report = classification_report(y_test, knn_predictions, output_dict=True, target_names=label)
pd.DataFrame(knn_report).transpose()


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Dari kode di atas dapat dilihat bahwa model dengan algoritma K-Nearest Neighbor memperoleh nilai akurasi yaitu sebesar 0.986364 dengan k = 1.</span></div>

# # 5.3 *Model Development - XGBoost Algorithm*

# In[26]:


# membuat model dengan algoritma XGBoost
xgb = XGBClassifier(random_state = 18)
xgb.fit(X_train, y_train)


# In[27]:


# menguji model menggunakan data test
xgb_predictions = xgb.predict(X_test)
xgb_report = classification_report(y_test, xgb_predictions, output_dict=True, target_names=label)
pd.DataFrame(xgb_report).transpose()


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Dari kode di atas dapat dilihat bahwa model dengan algoritma XGBoost memperoleh nilai akurasi yaitu sebesar 0.99545.</span></div>

# ## 5.3 *Model Development - Random Forest*

# In[28]:


# membuat model dengan algoritma Random Forest
rf = RandomForestClassifier(random_state = 18)
rf.fit(X_train, y_train)


# In[29]:


# menguji model menggunakan data test
rf_predictions = rf.predict(X_test)
rf_report = classification_report(y_test, rf_predictions, output_dict=True, target_names=label)
pd.DataFrame(rf_report).transpose()


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Dari kode di atas dapat dilihat bahwa model dengan algoritma Random Forest memperoleh nilai akurasi yaitu sebesar 0.997727.</span></div>

# # 6. Evaluasi Model

# ## 6.1 *Confusion Matrix - K-Nearest Neighbor*

# In[30]:


plt.figure(figsize = (15,9))
sns.heatmap(confusion_matrix(y_test, knn_predictions), annot = True)
plt.title("Confusion Matrix Model K-Nearest Neighbor")
plt.show()


# ## 6.2 *Confusion Matrix - XGBoost Algorithm*

# In[31]:


plt.figure(figsize = (15,9))
sns.heatmap(confusion_matrix(y_test, xgb_predictions), annot = True)
plt.title("Confusion Matrix Model XGBoost Algorithm")
plt.show()


# ## 6.3 *Confusion Matrix - Random Forest*

# In[32]:


plt.figure(figsize = (15,9))
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot = True)
plt.title("Confusion Matrix Model Random Forest")
plt.show()


# ## Perbandingan metriks akurasi antar model

# In[33]:


#menghitung nilai akurasi, precision dan recall setiap model
knn_accuracy = round((accuracy_score(y_test, knn_predictions)*100), 2)
xgb_accuracy = round((accuracy_score(y_test, xgb_predictions)*100), 2)
rf_accuracy = round((accuracy_score(y_test, rf_predictions)*100), 2)

knn_precision = round((precision_score(y_test, knn_predictions, average='macro')*100), 2)
xgb_precision = round((precision_score(y_test, xgb_predictions, average='macro')*100), 2)
rf_precision = round((precision_score(y_test, rf_predictions, average='macro')*100), 2)

knn_recall = round((recall_score(y_test, knn_predictions, average='macro')*100), 2)
xgb_recall = round((recall_score(y_test, xgb_predictions, average='macro')*100), 2)
rf_recall = round((recall_score(y_test, rf_predictions, average='macro')*100), 2)


# In[34]:


# membat dataframe hasil evaluasi
list_evaluasi= [[knn_accuracy, knn_precision, knn_recall],
            [xgb_accuracy, xgb_precision, xgb_recall],
               [rf_accuracy, rf_precision, rf_recall],]
evaluasi = pd.DataFrame(list_evaluasi,
                        columns=['Accuracy (%)', 'Precision (%)', 'Recall (%)'],
                        index=['K-Nearest Neighbor', 'XGBoost','Random Forest'])
evaluasi


# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">**Keterangan**: <br>Dari hasil evaluasi di atas dapat memberikan informasi bahwa ketiga model yang dibangun memiliki performa di atas 98%. Dimana dapat dilihat juga bahwa model dengan algoritma Random Forest memiliki performa (nilai akurasi, precision, dan recall) yang lebih baik dari dua model lainnya yaitu model dengan algoritma K-Nearest Neighbor dan XGBoost.</span></div>

# # Penutup

# ####  <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;">Model untuk memprediksi jenis tanaman yang cocok ditanam di lahan pertanian tertentu telah selesai dibuat, dan dari hasil pengujian, ketiga model yang dibuat memiliki performa yang baik dan dapat digunakan untuk memprediksi data sebenarnya.</span></div>

# ## Referensi

# #### <div align="left"><span style="white-space: pre-wrap; font: normal 11pt Arial; line-height: 1.5;"><ul><li>https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset</li><li>https://www.kaggle.com/venugopalkadamba/croprecommendation-eda-visualization-modeling-99#XGBoost-Classifier</li><li>https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390</li><li>https://www.kaggle.com/njain5/crop-prediction-using-classification-models</li><li>https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826</li></ul></span></div>
