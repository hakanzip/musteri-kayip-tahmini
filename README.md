# Müşteri Kaybı (Churn) Tahmini

Her şirketin kâbusu, müşterinin sessizce kalkıp gitmesidir. Biz gitmeden önce "bu müşteri bizi terk etmek üzere" diyen bir erken uyarı kuracağız.

## Veri seti

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, `blastchar/telco-customer-churn`) kullanıldı: bir telekom şirketinin 7.043 müşterisi, 21 sütun (demografi, aldığı hizmetler, sözleşme tipi, fatura tutarları ve hedef değişken `Churn`).

Churn dağılımı dengesiz: 1.869 müşteri (%26,5) ayrılmış, 5.174 müşteri (%73,5) kalmış. Bu yüzden değerlendirmede accuracy tek başına kullanılmadı, aşağıdaki Sonuçlar bölümüne bakın.

Not (veri temizliği): `TotalCharges` sütunu ham veride metin olarak gelmiş ve 11 satırda boş string içeriyordu. İncelendiğinde bu 11 satırın tamamının `tenure = 0` olduğu, yani henüz ilk faturasını almamış yeni müşteriler olduğu görüldü. Bu yüzden boş değerler 0 kabul edildi.

## Yöntem notu

Projenin ana fikrinde "doğrusal regresyon + karar ağaçları" geçiyor, ama hedef değişken ikili (kalacak / ayrılacak), yani bir sınıflandırma problemi. Bu yüzden "doğrusal model" ailesini temsilen doğrusal regresyon değil, sınıflandırmanın doğrusal karşılığı olan Lojistik Regresyon kullanıldı. Buna ek olarak üçüncü bir model olarak XGBoost eğitildi.

## Modelleme

İkili kategorik sütunlar (cinsiyet, partner, bağımlı, telefon hizmeti, kağıtsız fatura) 0/1'e çevrildi; 10 çok kategorili sütun (internet servisi, sözleşme tipi, ödeme yöntemi vb.) one-hot ile kodlandı; sayısal sütunlar (tenure, aylık/toplam ücret, kıdemli vatandaş) `StandardScaler` ile ölçeklendi.

Veri %80/%20 oranında, sınıf dağılımı korunacak şekilde (stratified) ayrıldı: 5.634 eğitim, 1.409 test satırı. Üç model, Lojistik Regresyon, Karar Ağacı ve XGBoost, aynı `RandomizedSearchCV` düzeniyle ayarlandı (20 kombinasyon denemesi, 5 katlı çapraz doğrulama, skor ROC-AUC).

Dengesiz sınıf için Lojistik Regresyon ve Karar Ağacı'nda `class_weight="balanced"` kullanıldı; XGBoost'ta bunun karşılığı, eğitim setinden hesaplanan `scale_pos_weight ≈ 2,77` (kalan/ayrılan oranı) oldu.

## Sonuçlar

Test setinde (1.409 müşteri), ROC-AUC'a göre sıralı:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0,745 | 0,512 | 0,816 | 0,629 | 0,846 |
| Lojistik Regresyon | 0,737 | 0,503 | 0,783 | 0,613 | 0,841 |
| Karar Ağacı | 0,712 | 0,475 | 0,805 | 0,597 | 0,830 |

En iyi model XGBoost oldu, ama fark Lojistik Regresyon'a göre küçük (ROC-AUC farkı sadece 0,005). Bu veri setinde ilişkiler büyük ölçüde doğrusal, bu yüzden basit model de güçlü modele yakın performans veriyor. Üç modelde de recall precision'dan yüksek: amaç "ayrılacak müşteriyi kaçırmamak" olduğu için class_weight/scale_pos_weight ile model bilerek biraz daha fazla yanlış alarma razı edildi.

Görsellere bakınca iki şey öne çıkıyor. Birincisi, sözleşme tipi gerçekten belirleyici: aylık sözleşmeli müşterilerin %42,7'si ayrılmış, bu oran 1 yıllıkta %11,3'e, 2 yıllıkta %2,8'e düşüyor (`gorseller/01_sozlesme_tipi_churn_orani.png`). İkincisi, churn büyük ölçüde bir erken dönem sorunu: ayrılan müşterilerin üyelik süresi (tenure) medyanı yaklaşık 10 ay iken kalanlarınki 38 ay (`gorseller/02_tenure_churn_violin.png`). Kayıp ilk yıl içinde yoğunlaşıyor, sadakat çalışmalarının en çok bu pencerede karşılık bulacağı söylenebilir.

## Görseller (`gorseller/`)

1. `01_sozlesme_tipi_churn_orani.png` / `.html` — sözleşme tipine göre churn oranı (Plotly gruplu bar)
2. `02_tenure_churn_violin.png` / `.html` — üyelik süresi × churn dağılımı (Plotly violin)
3. `03_korelasyon_isi_haritasi.png` — sayısal/ikili değişkenler arası korelasyon (seaborn)
4. `04_karar_agaci_gorseli.png` — karar ağacının görsel hali (yalnızca gösterim amaçlı, `max_depth=3` sınırlı ayrı bir ağaç; tuned modelin derinliği görsel olarak okunamayacak kadar fazla)
5. `05_shap_waterfall.png` — SHAP waterfall: müşteri `5178-LMXOP` için modelin %93,4 ayrılma olasılığı tahmininin nedenleri (kısa tenure, fiber optik internet, elektronik çek ödemesi en büyük itici faktörler)
6. `06_roc_karmasiklik_matrisi.png` / `.html` — en iyi model (XGBoost) için ROC eğrisi + karmaşıklık matrisi

Not: dtreeviz kütüphanesi bu makinede kurulu, ama sistemde Graphviz'in `dot` çalıştırılabilir dosyası olmadığı için (`ExecutableNotFound`) karar ağacı görseli otomatik olarak `sklearn.tree.plot_tree` ile üretildi (kod içinde try/except fallback).

## Notebook

`proje.ipynb`, önce `proje.py` script olarak yazılıp test edildi, `jupytext` ile notebook'a çevrildi, sonra `jupyter nbconvert --execute` ile gerçekten baştan sona çalıştırıldı. 17 kod hücresinin tamamında çıktı üretildiği kontrol edildi.

## Kullanılan kütüphaneler

- [pandas](https://pandas.pydata.org/docs/) — veri işleme
- [NumPy](https://numpy.org/doc/) — sayısal işlemler
- [scikit-learn](https://scikit-learn.org/stable/) — ön işleme, Lojistik Regresyon, Karar Ağacı, `RandomizedSearchCV`, metrikler
- [XGBoost](https://xgboost.readthedocs.io/) — gradyan artırmalı ağaç modeli
- [SHAP](https://shap.readthedocs.io/) — model yorumlama (waterfall grafiği)
- [dtreeviz](https://github.com/parrt/dtreeviz) — karar ağacı görselleştirme (bu ortamda Graphviz eksikliği nedeniyle fallback'e düştü)
- [Plotly](https://plotly.com/python/) — interaktif grafikler
- [seaborn](https://seaborn.pydata.org/) — korelasyon ısı haritası
- [Matplotlib](https://matplotlib.org/stable/) — statik grafikler
- [SciPy](https://docs.scipy.org/doc/scipy/) — `RandomizedSearchCV` parametre dağılımları (`loguniform`, `randint`, `uniform`)
- [Jupytext](https://jupytext.readthedocs.io/) — script ↔ notebook dönüşümü
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) — veri seti indirme

## Dosya yapısı

```
04_musteri_churn/
├── proje.py                        # ana script (kaynak), jupytext ile notebook'a çevrildi
├── proje.ipynb                     # çalıştırılmış, çıktıları doğrulanmış notebook
├── model_karsilastirma_tablosu.csv # 3 modelin test seti metrikleri
├── requirements.txt
├── README.md
├── veri/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── gorseller/
    ├── 01_sozlesme_tipi_churn_orani.png / .html
    ├── 02_tenure_churn_violin.png / .html
    ├── 03_korelasyon_isi_haritasi.png
    ├── 04_karar_agaci_gorseli.png
    ├── 05_shap_waterfall.png
    └── 06_roc_karmasiklik_matrisi.png / .html
```
