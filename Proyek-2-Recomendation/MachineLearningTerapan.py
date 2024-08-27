#%% md
# # Book Recommendation System
# Content-based Collaborative filtering using
# - Author
# - Title
# - Publisher
# - Category
# 
# All four point above are the features for the ML Model
# 
# <hr>
# 
# Make sure to install `opendatasets' on virtual environment, if you dont have just install on notebook with this code cell `!pip install opendatasets`
#%%
!pip install opendatasets
#%%
import os
import re
import nltk
import requests
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import opendatasets as od
from nltk.corpus import stopwords
nltk.download("stopwords")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
#%%
od.download('https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset')
#%%
books = pd.read_csv('/content/bookcrossing-dataset/Books Data with Category Language and Summary/Preprocessed_data.csv')
books.head()
#%%
books.info()
#%%
books.isna().sum()
#%%
print(books['Category'].unique, '\n')
#%%
books.Category.value_counts()
#%%
print(sorted(books['rating'].unique()), '\n')
books['rating'].value_counts()
#%%
books['publisher'].value_counts()
#%% md
# <hr>
# 
# # Dataset Information
# the information gathered from the dataset are
# - colum information
#     - user_id - id dari pengguna
#     - location - lokasi/alamat pengguna
#     - age - umur pengguna
#     - isbn - kode ISBN (International Standard Book Number) buku
#     - rating - rating dari buku
#     - book_title - judul buku
#     - book_author - penulis buku
#     - year_of_publication - tahun terbit buku
#     - publisher - penerbit buku
#     - img_s - gambar sampul buku (small)
#     - img_m - gambar sampul buku (medium)
#     - img_l - gambar sampul buku (large)
#     - Summary - ringkasan/sinopsis buku
#     - Language - bahasa yang digunakan buku
#     - Category - kategori buku
#     - city - kota pengguna
#     - state - negara bagian penguna
#     - country - negara pengguna
#     
# - Null / NaN / Missing Values
#     - 1 on book author
#     -  14103 on City
#     - 22798 on State
#     - 35374 on Country
#      
# - Rating points range are from 0 to 10
# - "9" as Category has 406102 on books
# 
# <hr>
#%% md
# # Data Preprocessing
# 
# - Drop NaN / Null / Missing values
# - Drop unused columns
# - drop the "9" category
# - drop the data that has 0 rating
# - Apply the regex for changing the non-alphanumeric with whitespace, then cut the whitespace at front and end of the string
# 
#%%
data = books.copy()
data.dropna(inplace=True, how='any', axis=0)
data.reset_index(drop=True, inplace=True)
data.drop(columns = ['Unnamed: 0','location','isbn',
                   'img_s','img_m', 'img_l', 'city','age',
                   'state','Language','country',
                   'year_of_publication', 'Summary'],axis=1,inplace = True)
data.drop(index=data[data.Category == '9'].index, inplace=True)
data.drop(index=data[data.rating == 0].index, inplace=True)
data.Category = data.Category.apply(lambda x: re.sub('[\W_]+', ' ', x).strip())
#%%
data.head()
#%%
data.info()
#%%
data.isnull().sum()
#%%
data['Category'].value_counts()
#%% md
# the total category are 4013 categories, that's to much to handle, so if you have PyCharm, it can show first 10 / 20 entries <br>
# or you can run the code cell below to show first 25 entries
#%%
i = 1
for idx, name in enumerate(data['Category'].value_counts().index.tolist()):
    if i > 25:
        break
    print(i)
    print('Name :', name)
    print('Counts :', data['Category'].value_counts().iloc[idx])

    print('---'*8)
    i+=1
#%% md
# ## Picking Category
# 
# Pada bagian ini akan dipilih dari category secara spesifik, pemilihan ini dilakukan dikarenakan sumber daya yang dimiliki untuk menghitung tidak memadai untuk menggunakan seluruh category. Category yang akan dipakai adalah 25 category pertama dari data yang sudah di olah
#%%
cate_list = data.Category.value_counts().index.tolist()
print(cate_list[:25])
print(len(cate_list[:25]))
#%%
df = data[data.Category.isin(cate_list[:25])]
df.info()
#%%
df.Category.nunique()
#%%
prep = df.copy()
prep.sort_values('book_title')
#%%
prep = prep.drop_duplicates('book_title')
prep.info()
#%%
prep.head()
#%%
prep['Category'] = prep['Category'].str.replace(' ', '_')
prep.head(10)
#%%
prep.info()
#%%
book_new = pd.DataFrame({
    'title': prep['book_title'].tolist(),
    'author': prep['book_author'].tolist(),
    'category': prep['Category'].tolist(),
    'publisher': prep['publisher'].tolist()
})
book_new.head()
#%% md
# # Modeling
# 
# ## TF-IDF
# tahapan ini akan membangun sistem rekomendasi berdasarkan category yang dimiliki buku. Teknik ini digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap category buku.
# 
# `.fit()` berfungsi untuk melakukan perhitungan terhadap category buku
# `.fit_transform()` berguna untuk men-translate hasil perhitungan tersebut kedalam kolom
#%%
tfidf = TfidfVectorizer()
tfidf.fit(book_new['category'])
tfidf.get_feature_names_out()
#%%
tfidf_matrix = tfidf.fit_transform(book_new['category'])
tfidf_matrix.shape
#%%
# untuk mendatarkan matrix dan memiliki koneksi ke fitur lain
tfidf_matrix.todense()
#%% md
# Membuat dataframe untuk melihat tf-idf matrix, Kolom diisi dengan category book, Baris di isi dengan nama book
#%%
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=book_new.title
).sample(5, axis=1).sample(10, axis=0)
#%% md
# ## Cosine Similarity
# Menghitung cosine similarity pada matrix tf-idf
#%%
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
#%% md
# Make a new dataframe from cosine similarity with row and columns from book_title
#%%
cosine_sim_df = pd.DataFrame(cosine_sim, index=book_new['title'], columns=book_new['title'])
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(3, axis=0)
#%%
def recommendation(book_name, similarity_data=cosine_sim_df, items=book_new[['title', 'category']], k=5):
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,book_name].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop nama_resto agar nama resto yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(book_name, errors='ignore')
    df = pd.DataFrame(closest).merge(items)
    df.drop_duplicates(keep='first', subset="title", inplace=True)
    return df.head(k)
#%%
book_new.head()
#%%
book_new[book_new['title'].eq("Macromedia Flash MX for Dummies")]
#%%
recommendation("Macromedia Flash MX for Dummies", k=10)
#%%
