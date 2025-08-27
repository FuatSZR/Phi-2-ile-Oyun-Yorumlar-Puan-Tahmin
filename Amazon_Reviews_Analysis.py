
# **Proje: Yapay Zeka ile Puan Tahmini ve Değerlendirilmesi**
Bu proje, Amazon'un video oyunu yorumları veri setini kullanarak, bir metin incelemesinin sentimentini (duygu durumunu) analiz etmeyi ve 5 üzerinden bir puan tahmini yapmayı amaçlamaktadır. Yapay zekanın tahmin yaparken hangi kelimelere önem verdiği analizlerle bulunmaya çalışılmıştır.


## **Kullanılan Araçlar ve Veri Seti**

***Veri Seti:***

Amazon-Reviews-2023

Video Oyunları alt kümesinden 10.000 örnekle yapılmıştır

**Veri Seti Linki:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)


 ***Model:***

Microsoft Phi-2 (Hafif ve güçlü bir dil modeli)

***Temel Kütüphaneler:***

Pandas, NLTK, Hugging Face Transformers, Matplotlib

## **Proje Akışı**

**Veri Toplama ve Ön İşleme:** Ham yorum metinleri temizlenir ve modele hazır hale getirilir.

**Model Tahmini:** Phi-2 modeli, her yoruma bir puan (1-5) atamak için kullanılır.

**Analiz ve Değerlendirme:** Modelin tahminleri, gerçek puanlarla karşılaştırılarak modelin doğruluğu ve hataları incelenir.

**Sonuçlar:** Hatalı tahminlerin nedenlerini ortaya çıkarmak için kelime analizi yapılır ve bulguları görselleştirilir.

## Veri Seti Yüklenmesi ve Ön İşleme
"""

!pip install transformers accelerate bitsandbytes datasets

!pip install datasets==3.6.0

!huggingface-cli login

import pandas as pd
import re
import nltk

import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import bitsandbytes

try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    print(f"NLTK indirme hatası: {e}")

# Veri setini yükle
print("Veri seti yükleniyor...")
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Video_Games", split="full[:10000]")
df = pd.DataFrame(dataset)

df = pd.DataFrame(dataset)
df = df[df['verified_purchase'] == True]
df.head()

df = df.drop(columns=['title', 'images','asin','parent_asin','user_id','timestamp','helpful_vote','verified_purchase'])
df.head()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['text_cleaned'] = df['text'].apply(preprocess_text)

df.head()

"""## Modeli Hazırlama Ve Tahminin Gerçekleştirilmesi"""

model_id = "microsoft/phi-2"

try:
    print(f"{model_id} modeli sıkıştırılmış olarak yükleniyor...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model başarıyla GPU belleğine sıkıştırılarak yüklendi.")

except Exception as e:
    print(f"Hata: Model yüklenirken bir sorun oluştu. Detaylar: {e}")
    torch.cuda.empty_cache()
    gc.collect()
    exit()

# Güvenli analiz döngüsü
for index, row in df.iterrows():
    try:
        # Daha kısa ve daha kesin bir prompt formatı
        prompt = f"Based on the following review, what is the rating from 1 to 5? Review: {row['text_cleaned']}\nRating:"

        inputs = tokenizer(prompt, return_tensors="pt")

        generated_output = model.generate(
            inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda"),
            max_new_tokens=1,
            do_sample=False,
        )

        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

        sentiment = generated_text.split("Rating: ")[-1].strip()
        try:
            predicted_rating = int(sentiment)
            df.loc[index, 'predicted_rating'] = predicted_rating

        except ValueError:
            print(f"Uyarı: {index}. yorum için sayısal bir puan alınamadı. Çıktı: '{rating_str}'")
            df.loc[index, 'predicted_rating'] = None

        #print(f"Original Text: {row['text_cleaned']}")
        print(f"Model Rate: {sentiment}")
        print(f"Original Rate: {row['rating']}")
        print("-" * 50)

    except Exception as e:
        print(f"Error: {index}. problem occurs while comment processed. Details: {e}")
        continue
df.to_csv('analiz_sonuclari.csv', index=False)

df.drop('predicted_rating', axis=1, inplace=True)
df.head()

"""## **Karışıklık Matrisi İle Sonuçlara Genel Bakış**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    df = pd.read_csv('./drive/MyDrive/Amazon-Reviews-2023-raw_review_Video_Games/analiz_sonuclari.csv')
    print("DataFrame başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: 'analiz_sonuclari.csv' dosyası bulunamadı.")
    exit()


df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['predicted_rating', 'rating'], inplace=True)

df['predicted_rating'] = df['predicted_rating'].astype(int)
df['rating'] = df['rating'].astype(int)


cm = confusion_matrix(df['rating'], df['predicted_rating'], labels=sorted(df['rating'].unique()))


cm_df = pd.DataFrame(cm, index=sorted(df['rating'].unique()), columns=sorted(df['rating'].unique()))
cm_df.index.name = 'Gerçek Puan'
cm_df.columns.name = 'Tahmin Edilen Puan'


plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=.5)
plt.title('Puan Tahmin Karışıklık Matrisi', fontsize=16)
plt.show()


print("Sayısal Karışıklık Matrisi:")
print(cm_df)

"""## Model Hatalarının Analizi ile Sorunların Tespiti"""

nltk.download()

nltk.download('punkt')

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print(f"NLTK indirme hatası: {e}")

try:
    df = pd.read_csv('./drive/MyDrive/Amazon-Reviews-2023-raw_review_Video_Games/analiz_sonuclari.csv')
    print("DataFrame başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: 'analiz_sonuclari.csv' dosyası bulunamadı.")
    exit()

"""### **Gerçek Puanı 5 Olup, Yapay Zekanın 3 ve Altında Tahmin Ettiği Yorumlar**

Veri setinde 5 puan verildiği durumun fazlalığından dolayı modelin yanlış tahminlerine yüzeysel bir bakış
"""

df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['predicted_rating', 'rating'], inplace=True)

model_yanlis_tahminler = df[(df['rating'] == 5) & (df['predicted_rating'] <= 3)]

print(f"Gerçek Puanı 5 Olup, Yapay Zekanın 3 ve Altında Tahmin Ettiği Yorum Sayısı: {len(model_yanlis_tahminler)}")
print("-" * 50)

print(model_yanlis_tahminler[['text', 'rating', 'predicted_rating']].head(10))

# Veri tiplerini doğru şekilde ayarla
df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['predicted_rating', 'rating'], inplace=True)

# Gerçek puanı 5 olup, yapay zekanın 3 veya daha az tahmin ettiği yorumları bul
model_yanlis_tahminler = df[(df['rating'] == 5) & (df['predicted_rating'] <= 3)]

# Hatalı tahminlerin temizlenmiş metinlerini birleştir
all_wrong_reviews_text = ' '.join(model_yanlis_tahminler['text_cleaned'].tolist())

# Metni kelimelere ayır (tokenize et)
words = word_tokenize(all_wrong_reviews_text)

# Kelime sıklığını say
word_counts = Counter(words)

# En sık geçen 10 kelimeyi bul
top_30_words = word_counts.most_common(30)

print("Hatalı Tahminlerdeki En Sık Kullanılan 30 Kelime:")
for word, count in top_30_words:
    print(f"'{word}': {count} defa")

"""**Gözlem**

Modelimiz 'like', 'love' gibi kelimelere rağmen 3 ve altında tahminlerde bulunmuş.



**Geliştirme Önerisi**

Buradan yola çıkarak modelimizin hangi durumlarda 5 yerine ara değerler olan 4 ve 3 gibi puanlar verdiği bulunabilir.

### **Yapay Zekanın 1 Verdiği Yorumların Gerçek Puanları**
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from datasets import load_dataset

required_nltk_data = ['punkt', 'stopwords', 'wordnet']
for item in required_nltk_data:
    try:
        nltk.download(item, quiet=True)
    except Exception as e:
        print(f"Hata: {item} indirilirken bir sorun oluştu: {e}")

try:
    df = pd.read_csv('./drive/MyDrive/Amazon-Reviews-2023-raw_review_Video_Games/analiz_sonuclari.csv')
    print("DataFrame başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: 'analiz_sonuclari.csv' dosyası bulunamadı.")
    exit()

df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['predicted_rating', 'rating'], inplace=True)


model_1_puan_verenler = df[df['predicted_rating'] == 1]


print("Modelin 1 Puan Verdiği Yorumların Gerçek Puan Dağılımı:")
gercek_puan_dagilimi = model_1_puan_verenler['rating'].value_counts().sort_index()

print(gercek_puan_dagilimi)
print("-" * 50)

print("Modelin 1 Puan Verdiği Yorumlardan Örnekler:")
print(model_1_puan_verenler[['text', 'rating', 'predicted_rating']].sample(5))

df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['predicted_rating', 'rating'], inplace=True)

"""**Gözlem**

Modelimiz çoğunlukla doğru tahmin yapsa da, gerçek puanı 5 ve 2 olan bir çok yoruma 1 puan verdiğini görüyoruz. Buradan bu puanları daha detaylı inceliyoruz.

#### **Gerçek Puanı 5, Tahmini 1 Olan Yorumlar**
"""

modelin_yanildigi_yorumlars = df[(df['predicted_rating'] == 1) & (df['rating'] == 5)]

all_wrong_reviews_text = ' '.join(modelin_yanildigi_yorumlars['text_cleaned'].tolist())

words = word_tokenize(all_wrong_reviews_text)

word_counts = Counter(words)

top_20_words = word_counts.most_common(20)

print(f"Gerçek Puanı 5, Tahmini 1 Olan Yorum Sayısı: {len(modelin_yanildigi_yorumlars)}")
print("-" * 50)
print("Bu Yorumlarda En Sık Kullanılan 20 Kelime:")
for word, count in top_20_words:
    print(f"'{word}': {count} defa")

r5_p1_df = pd.DataFrame(top_20_words, columns=['word', 'count'])

print(r5_p1_df)

"""#### **Gerçek Puanı 2, Tahmini 1 Olan Yorumlar**"""

modelin_yanildigi_yorumlars = df[(df['predicted_rating'] == 1) & (df['rating'] == 2)]

all_wrong_reviews_text = ' '.join(modelin_yanildigi_yorumlars['text_cleaned'].tolist())

words = word_tokenize(all_wrong_reviews_text)

word_counts = Counter(words)

top_20_words = word_counts.most_common(20)

print(f"Gerçek Puanı 2, Tahmini 1 Olan Yorum Sayısı: {len(modelin_yanildigi_yorumlars)}")
print("-" * 50)
print("Bu Yorumlarda En Sık Kullanılan 20 Kelime:")
for word, count in top_20_words:
    print(f"'{word}': {count} defa")

r2_p1_df = pd.DataFrame(top_20_words, columns=['word', 'count'])

print(r2_p1_df)

"""#### **Gerçek Puanı 1, Tahmini 1 Olan Yorumlar**"""

modelin_yanildigi_yorumlars = df[(df['predicted_rating'] == 1) & (df['rating'] == 1)]

all_wrong_reviews_text = ' '.join(modelin_yanildigi_yorumlars['text_cleaned'].tolist())

words = word_tokenize(all_wrong_reviews_text)

word_counts = Counter(words)

top_20_words = word_counts.most_common(20)

print(f"Gerçek Puanı 1, Tahmini 1 Olan Yorum Sayısı: {len(modelin_yanildigi_yorumlars)}")
print("-" * 50)
print("Bu Yorumlarda En Sık Kullanılan 20 Kelime:")
for word, count in top_20_words:
    print(f"'{word}': {count} defa")

r1_p1_df = pd.DataFrame(top_20_words, columns=['word', 'count'])

print(r1_p1_df)

"""### **Tahmini 1 Puan Olan Yorumların Analizi**

Modelimizin 1 puan verdiği yorumların, yazanları tarafından 5 ve 2 puan verilmesi durumlarında en sık kullanılan 20 kelime üzerinden bir analiz yapıldı.
"""

import matplotlib.pyplot as plt
import seaborn as sns

all_words = list(set(r1_p1_df['word']).union(r2_p1_df['word']).union(r5_p1_df['word']))

merged_df = pd.DataFrame(all_words, columns=['word'])
merged_df = merged_df.merge(r1_p1_df, on='word', how='left', suffixes=('', '_r1')).rename(columns={'count': 'r1_p1_count'}).fillna(0)
merged_df = merged_df.merge(r2_p1_df, on='word', how='left', suffixes=('', '_r2')).rename(columns={'count': 'r2_p1_count'}).fillna(0)
merged_df = merged_df.merge(r5_p1_df, on='word', how='left', suffixes=('', '_r5')).rename(columns={'count': 'r5_p1_count'}).fillna(0)

df_plot = merged_df.melt(id_vars='word', var_name='Rating Category', value_name='Frequency')
df_plot['Rating Category'] = df_plot['Rating Category'].map({
    'r1_p1_count': 'Gerçek Puan 1',
    'r2_p1_count': 'Gerçek Puan 2',
    'r5_p1_count': 'Gerçek Puan 5'
})
plt.figure(figsize=(18, 10))
sns.barplot(data=df_plot, x='word', y='Frequency', hue='Rating Category')
plt.title('Kelime Frekanslarının Gerçek Puanlara Göre Dağılımı', fontsize=18)
plt.xlabel('Kelime', fontsize=14)
plt.ylabel('Frekans', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Yorumun Gerçek Puanı')
plt.tight_layout()
plt.show()

"""**Gözlem:**

"battery", "use", "cant", "money", "amazon" ve "product" kelimelerinin geçtiği metinlerde tam olarak doğru tahmin yapılmış.

"like", "dont", "great", "doesnt", "love" gibi analizde etkili rol oynayabilecek kelimelerde modelin "yanlışlıkla 1 puan verme" durumu oldukça fazla. Bu durum dikkate alınarak promptta güncelleme yapılabilir, model değişimine veya optimizasyonuna gidilebilir.

**Geliştirme Önerisi**

Kelimeler detaylıca incelenerek başka bağlantılar bulunabilir.

Modelin diğer puan durumları incelenebilir ve bu sonuçla karşılaştırılabilir.
"""

