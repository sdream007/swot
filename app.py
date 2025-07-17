import streamlit as st
import pandas as pd
import re
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Önceden hazırlanmış labeled dataset ---
@st.cache_data
def load_existing_data():
    df = pd.read_csv("swot_labeled_examples.csv")
    return df

df_existing = load_existing_data()
texts_existing = df_existing['Text'].values
labels_existing = df_existing['Label'].values

# TF-IDF ve model ilk kurulumu (Streamlit yeniden başlatılana kadar cache ile kalır)
@st.cache_resource
def train_initial_model(texts, labels):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
    model.fit(X, labels)
    return vectorizer, model

vectorizer, model = train_initial_model(texts_existing, labels_existing)

# Basit otomatik SWOT etiketleme fonksiyonu
def otomatik_swot_etiketle(cumle):
    c = cumle.lower()
    if any(k in c for k in ["güçlü", "avantaj", "başarı", "yarar", "fırsat", "teşvik", "potansiyel"]):
        return "Opportunity"
    if any(k in c for k in ["eksik", "yetersiz", "risk", "tehdit", "engel"]):
        return "Threat"
    # Daha da geliştirilebilir
    return "Uncertain"

def yeni_dosyayi_oku_ve_etiketle(dosya):
    metin = dosya.read().decode("utf-8")
    cumleler = re.split(r'(?<=[.!?])\s+', metin)
    veri = []
    for c in cumleler:
        label = otomatik_swot_etiketle(c)
        veri.append({"Text": c.strip(), "Label": label})
    return pd.DataFrame(veri)

def modeli_online_egit_yeni_veri(df_yeni):
    global model, vectorizer
    metinler = df_yeni['Text'].values
    etiketler = df_yeni['Label'].values
    X_yeni = vectorizer.transform(metinler)
    model.partial_fit(X_yeni, etiketler, classes=model.classes_)
    st.success(f"{len(metinler)} cümle ile model güncellendi.")

st.title("Otomatik SWOT Etiketleme ve Model Güncelleme")

uploaded_files = st.file_uploader("Yeni txt dosyalarını seçin", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### Dosya: {uploaded_file.name}")
        yeni_df = yeni_dosyayi_oku_ve_etiketle(uploaded_file)
        st.dataframe(yeni_df)
        modeli_online_egit_yeni_veri(yeni_df)
