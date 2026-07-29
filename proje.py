# %% [markdown]
# # Müşteri Kaybı (Churn) Tahmini — Telco Customer Churn
#
# Her şirketin kâbusu, müşterinin sessizce kalkıp gitmesidir. Biz gitmeden
# önce "bu müşteri bizi terk etmek üzere" diyen bir erken uyarı sistemi
# kuracağız. Elimizde bir telekom şirketinin ~7000 müşterisinin sözleşme
# bilgileri, kullandığı hizmetler ve fatura verisi var; hedefimiz bu
# müşterilerden hangisinin şirketten ayrılacağını (churn) önceden kestirmek.
#
# **Not (yöntem netliği):** Projenin ana fikrinde "doğrusal regresyon +
# karar ağaçları" geçiyor; ancak burada hedef değişken ikili (kalacak /
# ayrılacak), yani bir **sınıflandırma** problemi. Bu yüzden "doğrusal
# model" ailesini temsilen doğrusal regresyon değil, sınıflandırmanın
# doğrusal karşılığı olan **Lojistik Regresyon** kullanılıyor. Buna ek
# olarak bir de **XGBoost** ile güçlü bir topluluk (ensemble) modeli
# eğitip üç modeli karşılaştırıyoruz.

# %%
import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from scipy.stats import loguniform, randint, uniform

from xgboost import XGBClassifier
import shap

# jupyter nbconvert --execute notebook içinde __file__ tanımlı DEĞİL;
# script olarak çalıştırıldığında ise var. İkisinde de çalışsın diye:
PROJE_KOK = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
VERI_YOLU = PROJE_KOK / "veri" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
GORSEL_KOK = PROJE_KOK / "gorseller"
GORSEL_KOK.mkdir(exist_ok=True)

RASSAL_TOHUM = 42

# --- Görsel tema: dataviz iskeletinden alınan, doğrulanmış renk paleti ---
RENK_MAVI = "#2a78d6"      # kategorik slot 1 -> "Kaldı" / Hayır
RENK_KIRMIZI = "#e34948"   # kategorik slot 8 -> "Ayrıldı" / Evet (churn = risk)
RENK_TURUNCU = "#eb6834"   # kategorik slot 2 -> 2. sözleşme kategorisi
RENK_AQUA = "#1baf7a"      # kategorik slot 3 -> 3. sözleşme kategorisi
RENK_GRI_NOTR = "#f0efec"  # diverging orta nokta (korelasyon ısı haritası)
YUZEY = "#fcfcfb"
METIN_ANA = "#0b0b0b"
METIN_SOLUK = "#52514e"

PLOTLY_SABLON = dict(
    layout=go.Layout(
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=METIN_ANA, size=13),
        paper_bgcolor=YUZEY,
        plot_bgcolor=YUZEY,
        colorway=[RENK_MAVI, RENK_KIRMIZI, RENK_TURUNCU, RENK_AQUA],
        xaxis=dict(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7"),
        yaxis=dict(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

# --- Görsellerde okunabilirlik için Türkçe etiketleme yardımcıları ---
# NOT: Bu sözlük SADECE görsele YAZILAN metni etkiler. Veri setindeki /
# koddaki gerçek sütun adları (df["tenure"], df["MonthlyCharges"] vb.)
# İngilizce/orijinal haliyle kalır — sadece eksende, karar ağacı
# düğümlerinde ve SHAP grafiğinde göründüklerinde yanlarına parantez
# içinde kısa bir Türkçe açıklama ekleniyor (ör. "tenure (üyelik süresi, ay)").
SUTUN_ACIKLAMA_TR = {
    "gender": "cinsiyet",
    "SeniorCitizen": "65 yaş üstü",
    "Partner": "partner var mı",
    "Dependents": "bakmakla yükümlü",
    "tenure": "üyelik süresi, ay",
    "PhoneService": "telefon hizmeti",
    "MultipleLines": "çoklu hat",
    "InternetService": "internet hizmeti",
    "OnlineSecurity": "çevrimiçi güvenlik",
    "OnlineBackup": "çevrimiçi yedek",
    "DeviceProtection": "cihaz koruma",
    "TechSupport": "teknik destek",
    "StreamingTV": "TV yayını",
    "StreamingMovies": "film yayını",
    "Contract": "sözleşme",
    "PaperlessBilling": "e-fatura",
    "PaymentMethod": "ödeme yöntemi",
    "MonthlyCharges": "aylık ücret",
    "TotalCharges": "toplam ücret",
    "Churn": "müşteri kaybı",
}


def sutun_etiket_tr(ozellik_adi: str) -> str:
    """Sütun/özellik adının yanına parantez içinde Türkçe açıklama ekler.

    Örn: "tenure" -> "tenure (üyelik süresi, ay)"
         "Contract_Two year" -> "Contract_Two year (sözleşme)"
    Gerçek (İngilizce) sütun adı değişmez; sadece görsele yazılan etiket
    okunaklı hale getirilir. Eşleşme yoksa adı olduğu gibi döndürür.
    """
    if ozellik_adi in SUTUN_ACIKLAMA_TR:
        return f"{ozellik_adi} ({SUTUN_ACIKLAMA_TR[ozellik_adi]})"
    on_ek = ozellik_adi.split("_", 1)[0]
    if on_ek in SUTUN_ACIKLAMA_TR:
        return f"{ozellik_adi} ({SUTUN_ACIKLAMA_TR[on_ek]})"
    return ozellik_adi


# SHAP kütüphanesinin bazı grafik tiplerinde kullandığı varsayılan İngilizce
# iç etiketler (waterfall grafiğinde şu an fiilen okunmuyor ama başka bir
# SHAP grafiği eklenirse de Türkçe kalsın diye baştan çeviriyoruz).
shap.plots._labels.labels.update({
    "MAIN_EFFECT": "Ana etkinin SHAP değeri\n%s",
    "INTERACTION_VALUE": "SHAP etkileşim değeri",
    "INTERACTION_EFFECT": "SHAP etkileşim değeri\n%s ve %s",
    "VALUE": "SHAP değeri (model çıktısına etkisi)",
    "GLOBAL_VALUE": "ortalama(|SHAP değeri|) (model çıktısına ortalama mutlak etki)",
    "VALUE_FOR": "SHAP değeri\n%s",
    "PLOT_FOR": "%s için SHAP grafiği",
    "FEATURE": "Özellik %s",
    "FEATURE_VALUE": "Özellik değeri",
    "FEATURE_VALUE_LOW": "Düşük",
    "FEATURE_VALUE_HIGH": "Yüksek",
    "JOINT_VALUE": "Ortak SHAP değeri",
    "MODEL_OUTPUT": "Model çıktı değeri",
})

print(f"Proje kök dizini: {PROJE_KOK}")
print(f"Veri yolu: {VERI_YOLU}")

# %% [markdown]
# ## 1. Veri Yükleme

# %%
df = pd.read_csv(VERI_YOLU)
print("Veri seti boyutu:", df.shape)
df.head()

# %% [markdown]
# ## 2. Keşifsel Veri Analizi (EDA)
#
# Önce hedef değişkenin (Churn) dağılımına bakıyoruz — dengesiz mi, değil
# mi bunu bilmeden metrik seçemeyiz.

# %%
churn_sayim = df["Churn"].value_counts()
churn_oran = df["Churn"].value_counts(normalize=True) * 100
print("Churn dağılımı (adet):")
print(churn_sayim)
print("\nChurn dağılımı (%):")
print(churn_oran.round(2))

toplam_musteri = len(df)
ayrilan_musteri = int(churn_sayim.get("Yes", 0))
print(f"\nToplam müşteri: {toplam_musteri}, ayrılan: {ayrilan_musteri} "
      f"(%{100*ayrilan_musteri/toplam_musteri:.1f})")

# %%
print("Eksik değer sayımı:")
print(df.isnull().sum().sum(), "adet klasik NaN (TotalCharges boşlukları henüz metin halinde, aşağıda ele alınacak)")

bos_totalcharges = df[df["TotalCharges"].str.strip() == ""]
print(f"\nTotalCharges'ta boş string sayısı: {len(bos_totalcharges)}")
print("Bu satırların tenure (üyelik süresi) değerleri:", sorted(bos_totalcharges["tenure"].unique()))

# %% [markdown]
# **Gözlem:** TotalCharges sütunu metin (str) olarak okunmuş ve 11 satırda
# boş string var. Bu 11 satırın tamamında `tenure = 0`, yani bunlar henüz
# ilk faturasını almamış yepyeni müşteriler. Mantıklı varsayım: toplam
# ödemeleri 0 kabul edilir (aşağıdaki temizleme adımında uygulanıyor).

# %% [markdown]
# ### 2.1 Görsel 1 — Sözleşme tipine göre churn oranı

# %%
sozlesme_churn = (
    df.groupby(["Contract", "Churn"], observed=True).size().reset_index(name="adet")
)
sozlesme_toplam = df.groupby("Contract", observed=True).size().rename("toplam")
sozlesme_churn = sozlesme_churn.merge(sozlesme_toplam, on="Contract")
sozlesme_churn["oran"] = 100 * sozlesme_churn["adet"] / sozlesme_churn["toplam"]

sozlesme_sira = ["Month-to-month", "One year", "Two year"]
sozlesme_ad_tr = {"Month-to-month": "Aylık", "One year": "1 Yıllık", "Two year": "2 Yıllık"}
sozlesme_churn["Sözleşme"] = sozlesme_churn["Contract"].map(sozlesme_ad_tr)
sozlesme_churn["Churn_tr"] = sozlesme_churn["Churn"].map({"No": "Kaldı", "Yes": "Ayrıldı"})

fig1 = px.bar(
    sozlesme_churn,
    x="Sözleşme", y="oran", color="Churn_tr", barmode="group",
    category_orders={"Sözleşme": [sozlesme_ad_tr[s] for s in sozlesme_sira], "Churn_tr": ["Kaldı", "Ayrıldı"]},
    color_discrete_map={"Kaldı": RENK_MAVI, "Ayrıldı": RENK_KIRMIZI},
    text=sozlesme_churn["oran"].round(1).astype(str) + "%",
    labels={"oran": "Oran (%)", "Sözleşme": "Sözleşme Tipi", "Churn_tr": "Durum"},
    title="Sözleşme Tipine Göre Churn Oranı",
)
fig1.update_traces(textposition="outside")
# px.bar, text= için verilen ham seriye varsayılan olarak İngilizce "text"
# etiketini veriyor (hover'da "text=42.7%" görünüyordu) — Türkçeleştir.
for iz in fig1.data:
    if iz.hovertemplate and "text=%{text}" in iz.hovertemplate:
        iz.hovertemplate = iz.hovertemplate.replace("text=%{text}", "Etiket=%{text}")
fig1.update_layout(PLOTLY_SABLON["layout"], yaxis_range=[0, 105])
fig1.write_image(GORSEL_KOK / "01_sozlesme_tipi_churn_orani.png", width=900, height=550, scale=2)
fig1.write_html(GORSEL_KOK / "01_sozlesme_tipi_churn_orani.html")
print("Kaydedildi: 01_sozlesme_tipi_churn_orani.png / .html")
sozlesme_churn[["Sözleşme", "Churn_tr", "adet", "oran"]]

# %% [markdown]
# ### 2.2 Görsel 2 — Üyelik süresi (tenure) × churn dağılımı

# %%
df["Churn_tr"] = df["Churn"].map({"No": "Kaldı", "Yes": "Ayrıldı"})

fig2 = px.violin(
    df, x="Churn_tr", y="tenure", color="Churn_tr", box=True, points=False,
    category_orders={"Churn_tr": ["Kaldı", "Ayrıldı"]},
    color_discrete_map={"Kaldı": RENK_MAVI, "Ayrıldı": RENK_KIRMIZI},
    labels={"tenure": "Üyelik Süresi (ay)", "Churn_tr": "Durum"},
    title="Üyelik Süresi (Tenure) × Churn Dağılımı",
)
fig2.update_layout(PLOTLY_SABLON["layout"], showlegend=False)
fig2.write_image(GORSEL_KOK / "02_tenure_churn_violin.png", width=900, height=550, scale=2)
fig2.write_html(GORSEL_KOK / "02_tenure_churn_violin.html")
print("Kaydedildi: 02_tenure_churn_violin.png / .html")
df.groupby("Churn_tr", observed=True)["tenure"].describe()[["count", "mean", "50%", "std"]]

# %% [markdown]
# **Gözlem:** Ayrılan müşterilerin üyelik süresi medyanı, kalanlara göre
# belirgin biçimde daha düşük — churn büyük ölçüde bir "erken dönem"
# problemi.

# %% [markdown]
# ## 3. Veri Temizleme ve Ön İşleme

# %%
veri = df.drop(columns=["Churn_tr"]).copy()

# TotalCharges: metin -> sayı. Boşluklar tenure=0 (yeni müşteri, henüz
# faturalanmamış) olduğu için 0 kabul ediyoruz.
veri["TotalCharges"] = pd.to_numeric(veri["TotalCharges"].str.strip(), errors="coerce")
print("Dönüşüm sonrası NaN sayısı:", veri["TotalCharges"].isna().sum())
veri["TotalCharges"] = veri["TotalCharges"].fillna(0.0)
print("fillna(0) sonrası NaN sayısı:", veri["TotalCharges"].isna().sum())

# Hedef değişken
veri["Churn"] = veri["Churn"].map({"No": 0, "Yes": 1}).astype(int)

# İkili (2 kategorili) metinsel sütunlar -> 0/1
ikili_sutunlar = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
veri["gender"] = veri["gender"].map({"Male": 1, "Female": 0})
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    veri[c] = veri[c].map({"Yes": 1, "No": 0})

# Çok kategorili sütunlar (one-hot ile pipeline içinde kodlanacak)
cok_kategorili_sutunlar = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

sayisal_sutunlar = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + ikili_sutunlar

kimlik = veri["customerID"]
X = veri.drop(columns=["customerID", "Churn"])
y = veri["Churn"]

print("\nÖzellik matrisi boyutu:", X.shape)
print("Sayısal (ölçeklenecek) sütun sayısı:", len(sayisal_sutunlar))
print("Çok kategorili (one-hot) sütun sayısı:", len(cok_kategorili_sutunlar))
assert len(sayisal_sutunlar) + len(cok_kategorili_sutunlar) == X.shape[1], "Sütun sayısı tutmuyor!"

# %% [markdown]
# ### 3.1 Görsel 3 — Korelasyon ısı haritası
#
# Tam one-hot genişletilmiş matris (30+ sütun) okunaksız olacağı için ısı
# haritasını sayısal + ikili kodlanmış sütunlar üzerinde çiziyoruz.

# %%
korelasyon_sutunlari = sayisal_sutunlar + ["Churn"]
korelasyon = veri[korelasyon_sutunlari].corr()

# Görselde eksen etiketi olarak sütun adının yanına Türkçe açıklama eklenen
# ayrı bir kopya (hesaplamalar/sonraki adımlar orijinal `korelasyon`'u kullanır).
korelasyon_gorsel = korelasyon.rename(index=sutun_etiket_tr, columns=sutun_etiket_tr)

# Diverging özel renk haritası: kırmızı (negatif) <-> gri (0) <-> mavi (pozitif)
diverging_cmap = mcolors.LinearSegmentedColormap.from_list(
    "mavi_kirmizi_diverging", [RENK_KIRMIZI, RENK_GRI_NOTR, RENK_MAVI]
)

plt.figure(figsize=(12, 9))
sns.heatmap(
    korelasyon_gorsel, annot=True, fmt=".2f", cmap=diverging_cmap, center=0,
    vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Korelasyon Katsayısı"},
)
plt.title("Korelasyon Isı Haritası (Sayısal + İkili Kodlanmış Değişkenler)", fontsize=13)
plt.tight_layout()
plt.savefig(GORSEL_KOK / "03_korelasyon_isi_haritasi.png", dpi=150)
plt.close()
print("Kaydedildi: 03_korelasyon_isi_haritasi.png")

print("\nChurn ile en yüksek mutlak korelasyonlar:")
print(korelasyon["Churn"].drop("Churn").abs().sort_values(ascending=False).head(5))

# %% [markdown]
# ## 4. Eğitim / Test Ayrımı

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RASSAL_TOHUM, stratify=y
)
print(f"Eğitim seti: {X_train.shape[0]} satır, Test seti: {X_test.shape[0]} satır")
print(f"Eğitimde churn oranı: %{100*y_train.mean():.2f}, testte: %{100*y_test.mean():.2f}")

# %% [markdown]
# ## 5. Ön İşleme Pipeline'ı ve Model Eğitimi
#
# Üç model de aynı `ColumnTransformer` üzerinden geçiyor: sayısal/ikili
# sütunlar `StandardScaler` ile ölçekleniyor (ağaç tabanlı modeller için
# zararsız, lojistik regresyon için gerekli), çok kategorili sütunlar
# `OneHotEncoder` ile kodlanıyor. Churn sınıfı dengesiz olduğu
# (%26.5 ayrılan) için `class_weight="balanced"` kullanılıyor; XGBoost'ta
# karşılığı `scale_pos_weight`.

# %%
onisleme = ColumnTransformer(
    transformers=[
        ("sayisal", StandardScaler(), sayisal_sutunlar),
        ("kategorik", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cok_kategorili_sutunlar),
    ]
)

pos_agirlik = (y_train == 0).sum() / (y_train == 1).sum()
print(f"XGBoost scale_pos_weight (negatif/pozitif oranı): {pos_agirlik:.3f}")

cv_semasi = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASSAL_TOHUM)

modeller = {}
arama_sonuclari = {}

# --- 5.1 Lojistik Regresyon ---
lr_pipeline = Pipeline([
    ("onisleme", onisleme),
    ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=RASSAL_TOHUM)),
])
lr_param_dagilimi = {
    "clf__C": loguniform(1e-3, 1e2),
    "clf__penalty": ["l1", "l2"],
}
lr_arama = RandomizedSearchCV(
    lr_pipeline, lr_param_dagilimi, n_iter=20, cv=cv_semasi,
    scoring="roc_auc", n_jobs=2, random_state=RASSAL_TOHUM, verbose=0,
)
lr_arama.fit(X_train, y_train)
modeller["Lojistik Regresyon"] = lr_arama.best_estimator_
arama_sonuclari["Lojistik Regresyon"] = lr_arama.best_params_
print("Lojistik Regresyon en iyi parametreler:", lr_arama.best_params_)
print(f"CV ROC-AUC: {lr_arama.best_score_:.4f}")

# %%
# --- 5.2 Karar Ağacı ---
dt_pipeline = Pipeline([
    ("onisleme", onisleme),
    ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RASSAL_TOHUM)),
])
dt_param_dagilimi = {
    "clf__max_depth": randint(3, 16),
    "clf__min_samples_split": randint(2, 40),
    "clf__min_samples_leaf": randint(1, 20),
    "clf__criterion": ["gini", "entropy"],
}
dt_arama = RandomizedSearchCV(
    dt_pipeline, dt_param_dagilimi, n_iter=20, cv=cv_semasi,
    scoring="roc_auc", n_jobs=2, random_state=RASSAL_TOHUM, verbose=0,
)
dt_arama.fit(X_train, y_train)
modeller["Karar Ağacı"] = dt_arama.best_estimator_
arama_sonuclari["Karar Ağacı"] = dt_arama.best_params_
print("Karar Ağacı en iyi parametreler:", dt_arama.best_params_)
print(f"CV ROC-AUC: {dt_arama.best_score_:.4f}")

# %%
# --- 5.3 XGBoost ---
xgb_pipeline = Pipeline([
    ("onisleme", onisleme),
    ("clf", XGBClassifier(
        eval_metric="logloss", random_state=RASSAL_TOHUM,
        scale_pos_weight=pos_agirlik, n_jobs=1,
    )),
])
xgb_param_dagilimi = {
    "clf__n_estimators": randint(100, 320),
    "clf__max_depth": randint(2, 8),
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
}
xgb_arama = RandomizedSearchCV(
    xgb_pipeline, xgb_param_dagilimi, n_iter=20, cv=cv_semasi,
    scoring="roc_auc", n_jobs=2, random_state=RASSAL_TOHUM, verbose=0,
)
xgb_arama.fit(X_train, y_train)
modeller["XGBoost"] = xgb_arama.best_estimator_
arama_sonuclari["XGBoost"] = xgb_arama.best_params_
print("XGBoost en iyi parametreler:", xgb_arama.best_params_)
print(f"CV ROC-AUC: {xgb_arama.best_score_:.4f}")

# %% [markdown]
# ## 6. Test Setinde Değerlendirme ve Karşılaştırma Tablosu
#
# Dengesiz sınıf olduğu için accuracy tek başına yanıltıcı olabilir — bu
# yüzden precision, recall, F1 ve ROC-AUC birlikte raporlanıyor.

# %%
sonuc_satirlari = []
for isim, model in modeller.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    sonuc_satirlari.append({
        "Model": isim,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    })

karsilastirma_df = pd.DataFrame(sonuc_satirlari).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
    karsilastirma_df[c] = karsilastirma_df[c].round(4)

karsilastirma_yolu = PROJE_KOK / "model_karsilastirma_tablosu.csv"
karsilastirma_df.to_csv(karsilastirma_yolu, index=False)
print(f"Kaydedildi: {karsilastirma_yolu.name}")
print(karsilastirma_df.to_string(index=False))

en_iyi_model_adi = karsilastirma_df.iloc[0]["Model"]
en_iyi_model = modeller[en_iyi_model_adi]
print(f"\nROC-AUC'a göre en iyi model: {en_iyi_model_adi}")

# %% [markdown]
# ## 7. En İyi Model İçin ROC Eğrisi ve Karmaşıklık Matrisi

# %%
y_pred_best = en_iyi_model.predict(X_test)
y_proba_best = en_iyi_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_auc_deger = roc_auc_score(y_test, y_proba_best)
cm = confusion_matrix(y_test, y_pred_best)

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=(f"ROC Eğrisi ({en_iyi_model_adi})", "Karmaşıklık Matrisi"),
    specs=[[{"type": "xy"}, {"type": "heatmap"}]],
)
fig3.add_trace(
    go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc_deger:.3f})",
               line=dict(color=RENK_MAVI, width=3),
               hovertemplate="Yanlış Pozitif Oranı: %{x:.3f}<br>Doğru Pozitif Oranı: %{y:.3f}<extra></extra>"),
    row=1, col=1,
)
fig3.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Rastgele Tahmin",
               line=dict(color="#c3c2b7", width=2, dash="dash"),
               hovertemplate="Yanlış Pozitif Oranı: %{x:.3f}<br>Doğru Pozitif Oranı: %{y:.3f}<extra></extra>"),
    row=1, col=1,
)
cm_etiket = [["Kaldı (Gerçek)", "Ayrıldı (Gerçek)"]]
fig3.add_trace(
    go.Heatmap(
        z=cm[::-1], x=["Kaldı (Tahmin)", "Ayrıldı (Tahmin)"], y=["Ayrıldı (Gerçek)", "Kaldı (Gerçek)"],
        colorscale=[[0, YUZEY], [1, RENK_MAVI]], showscale=False,
        text=cm[::-1], texttemplate="%{text}", textfont=dict(size=18, color=METIN_ANA),
        hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Müşteri Sayısı: %{z}<extra></extra>",
    ),
    row=1, col=2,
)
fig3.update_xaxes(title_text="Yanlış Pozitif Oranı", row=1, col=1)
fig3.update_yaxes(title_text="Doğru Pozitif Oranı", row=1, col=1)
fig3.update_layout(PLOTLY_SABLON["layout"], title=f"En İyi Model Değerlendirmesi — {en_iyi_model_adi}", showlegend=True)
fig3.write_image(GORSEL_KOK / "06_roc_karmasiklik_matrisi.png", width=1100, height=520, scale=2)
fig3.write_html(GORSEL_KOK / "06_roc_karmasiklik_matrisi.html")
print("Kaydedildi: 06_roc_karmasiklik_matrisi.png / .html")
print("Karmaşıklık matrisi (satır=gerçek, sütun=tahmin, sıra: [Kaldı, Ayrıldı]):")
print(cm)

# %% [markdown]
# ## 8. Karar Ağacının Görselleştirilmesi
#
# Not: Ayarlanan (tuned) karar ağacı üretim modelidir ama derinliği
# (max_depth ~10+) görsel olarak okunamaz. Bu yüzden SADECE görselleştirme
# amacıyla aynı veri üzerinde `max_depth=3` sınırlı, ayrı bir "gösterim
# ağacı" eğitiliyor — metrik tablosundaki sayılar bu sınırlı ağaca değil,
# yukarıdaki tuned modele aittir.

# %%
X_train_islenmis = onisleme.fit_transform(X_train, y_train)
ozellik_isimleri = onisleme.get_feature_names_out()
ozellik_isimleri = [f.replace("sayisal__", "").replace("kategorik__", "") for f in ozellik_isimleri]
# Karar ağacı düğümlerinde ve SHAP grafiğinde görünecek etiketlere, gerçek
# sütun adını bozmadan yanına parantez içinde Türkçe açıklama ekliyoruz.
ozellik_isimleri = [sutun_etiket_tr(f) for f in ozellik_isimleri]

gosterim_agaci = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RASSAL_TOHUM)
gosterim_agaci.fit(X_train_islenmis, y_train)

agac_gorseli_yolu = GORSEL_KOK / "04_karar_agaci_gorseli.png"
dtreeviz_calisti = False
try:
    import dtreeviz
    viz_model = dtreeviz.model(
        gosterim_agaci, X_train=X_train_islenmis, y_train=y_train.values,
        feature_names=ozellik_isimleri, target_name="Churn",
        class_names=["Kaldı", "Ayrıldı"],
    )
    v = viz_model.view(scale=1.2)
    v.save(str(agac_gorseli_yolu.with_suffix(".svg")))
    # dtreeviz svg üretiyor; PNG karşılığını da matplotlib fallback ile üretelim ki
    # README/görsel envanteri PNG bekleyen akışla tutarlı kalsın.
    dtreeviz_calisti = True
    print(f"dtreeviz çalıştı, kaydedildi: {agac_gorseli_yolu.with_suffix('.svg').name}")
except Exception as e:
    print(f"dtreeviz başarısız oldu ({type(e).__name__}: {e}), sklearn plot_tree'ye düşülüyor.")
    # dtreeviz, dot çalıştırmayı denerken uzantısız bir ara .dot dosyası bırakabiliyor; temizle.
    yaris_dosyasi = agac_gorseli_yolu.with_suffix("")
    if yaris_dosyasi.exists() and yaris_dosyasi.is_file():
        yaris_dosyasi.unlink()

if not dtreeviz_calisti:
    plt.figure(figsize=(26, 13))
    agac_anotasyonlari = plot_tree(
        gosterim_agaci, feature_names=ozellik_isimleri, class_names=["Kaldı", "Ayrıldı"],
        filled=True, rounded=True, fontsize=8, proportion=False,
        impurity=True,
    )
    # sklearn.tree.plot_tree düğüm kutularına ve dal etiketlerine sabit
    # İngilizce metin gömüyor (samples/value/class/True/False) — bunların
    # yerine görsele yazılan metni Türkçeleştiriyoruz. "gini" (Gini indeksi)
    # teknik/isim terimi olduğu için ROC/AUC/XGBoost gibi orijinal bırakıldı.
    for ann in agac_anotasyonlari:
        metin = ann.get_text()
        if metin.strip() == "True":
            ann.set_text(metin.replace("True", "Evet"))
        elif metin.strip() == "False":
            ann.set_text(metin.replace("False", "Hayır"))
        else:
            metin_tr = (
                metin.replace("samples", "örnek")
                .replace("value", "değer")
                .replace("class", "sınıf")
            )
            ann.set_text(metin_tr)
    plt.title("Karar Ağacı (Gösterim Amaçlı, max_depth=3)", fontsize=14)
    plt.tight_layout()
    plt.savefig(agac_gorseli_yolu, dpi=150)
    plt.close()
    print(f"Kaydedildi: {agac_gorseli_yolu.name} (sklearn plot_tree fallback)")

# %% [markdown]
# ## 9. SHAP ile Model Yorumlama — Tek Bir Müşteri Neden Gidiyor?

# %%
en_iyi_pipeline = modeller[en_iyi_model_adi]
en_iyi_clf = en_iyi_pipeline.named_steps["clf"]
X_test_islenmis = onisleme.transform(X_test)
X_test_islenmis_df = pd.DataFrame(X_test_islenmis, columns=ozellik_isimleri, index=X_test.index)

if en_iyi_model_adi in ("Karar Ağacı", "XGBoost"):
    aciklayici = shap.TreeExplainer(en_iyi_clf)
    shap_degerleri = aciklayici(X_test_islenmis_df)
else:
    arkaplan = shap.sample(pd.DataFrame(X_train_islenmis, columns=ozellik_isimleri), 100, random_state=RASSAL_TOHUM)
    aciklayici = shap.LinearExplainer(en_iyi_clf, arkaplan)
    shap_degerleri = aciklayici(X_test_islenmis_df)

# İkili sınıflandırmada bazı explainer'lar (n_ornek, n_ozellik, 2) şeklinde döner;
# pozitif sınıfı (Ayrıldı=1) seçiyoruz.
if len(shap_degerleri.values.shape) == 3:
    shap_degerleri_secili = shap_degerleri[:, :, 1]
else:
    shap_degerleri_secili = shap_degerleri

# En yüksek "ayrılma" olasılığına sahip test müşterisini seçelim (en öğretici örnek)
tum_olasiliklar = en_iyi_pipeline.predict_proba(X_test)[:, 1]
en_riskli_konum = int(np.argmax(tum_olasiliklar))
en_riskli_musteri_id = kimlik.loc[X_test.index[en_riskli_konum]]
print(f"Seçilen müşteri: {en_riskli_musteri_id}, tahmini ayrılma olasılığı: %{100*tum_olasiliklar[en_riskli_konum]:.1f}")
print(f"Gerçek durum: {'Ayrıldı' if y_test.iloc[en_riskli_konum] == 1 else 'Kaldı'}")

plt.figure()
shap_eksen = shap.plots.waterfall(shap_degerleri_secili[en_riskli_konum], max_display=12, show=False)
# shap.plots.waterfall, gösterilmeyen özellikleri "N other features" diye sabit
# İngilizce metinle grupluyor (kütüphane kodunda hardcoded, labels sözlüğünden
# okumuyor). Bu metin plt.yticks() ile kurulan bir
# FuncFormatter(Axis._format_with_dict, {konum: metin}) sözlüğünden geliyor;
# Text nesnesine .set_text() ile dokunmak KALICI OLMUYOR — tight_layout/savefig
# formatter'dan metni yeniden üretip üstüne yazıyor. Bu yüzden formatter'ın
# kendi sözlüğünü (gerçek kaynağı) Türkçeleştiriyoruz.
eksen_formatlayici = shap_eksen.yaxis.get_major_formatter()
try:
    etiket_sozlugu = eksen_formatlayici.func.args[0]
    for konum in list(etiket_sozlugu.keys()):
        metin = etiket_sozlugu[konum]
        eslesme = re.match(r"^(\d+) other features$", metin)
        if eslesme:
            etiket_sozlugu[konum] = f"{eslesme.group(1)} diğer özellik"
except (AttributeError, IndexError, TypeError) as e:
    print(f"UYARI: SHAP 'other features' etiketi Türkçeleştirilemedi ({type(e).__name__}: {e})")
plt.title(f"SHAP Waterfall — Müşteri {en_riskli_musteri_id} ({en_iyi_model_adi})", fontsize=11)
plt.tight_layout()
plt.savefig(GORSEL_KOK / "05_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Kaydedildi: 05_shap_waterfall.png")

# %% [markdown]
# ## 10. Sonuç ve Yorum
#
# Üç model de aynı veri, aynı ön işleme ve aynı çapraz doğrulama şeması
# (5 katlı, RandomizedSearchCV ile 20 kombinasyon denemesi) üzerinden
# eğitildi. Karşılaştırma tablosu ve görseller yukarıda; ayrıntılı yorum
# README.md içinde.

# %%
print("Proje betiği tamamlandı.")
print("Üretilen görseller:", sorted(p.name for p in GORSEL_KOK.iterdir()))
