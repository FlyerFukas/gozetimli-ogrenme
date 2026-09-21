# Sınıflandırma metrikleri

Kaynak ders notu: `9-Logistic_Regression_Performance_Metrics.pdf`.
Kod: [`gozetimli/metrikler/siniflandirma.py`](../../src/gozetimli/metrikler/siniflandirma.py) ·
Test: [`testler/test_siniflandirma_metrikleri.py`](../../testler/test_siniflandirma_metrikleri.py)

Hangi metriği ne zaman seçeceğiniz için [metrik-secim-rehberi.md](metrik-secim-rehberi.md).
Bu dosya tanımları ve davranışları anlatır.

---

## 1. Karmaşıklık matrisi — her şeyin kaynağı

Aşağıdaki metriklerin tamamı tek bir tablodan türer:

|  | Tahmin: Pozitif | Tahmin: Negatif |
|---|---|---|
| **Gerçek: Pozitif** | TP (True Positive) | FN (False Negative) |
| **Gerçek: Negatif** | FP (False Positive) | TN (True Negative) |

```python
from gozetimli.metrikler.siniflandirma import karmasiklik_matrisi, ikili_bilesenler

karmasiklik_matrisi(y_gercek, y_tahmin)   # [[TN, FP], [FN, TP]]
tn, fp, fn, tp = ikili_bilesenler(y_gercek, y_tahmin)
```

**Satır = gerçek, sütun = tahmin.** Köşegen doğru tahminlerdir. Bu sırayı
karıştırmak (özellikle başka bir kütüphaneden gelen matrisleri okurken) precision
ile recall'u yer değiştirir — bu, sessiz ve maliyetli bir hatadır.

İsimlendirme mantığı: ikinci kelime **modelin ne dediği**, birincisi **doğru
söyleyip söylemediği**. "False Positive" = model pozitif dedi, yanıldı.

### Normalleştirme

```python
karmasiklik_matrisi(y, t, normalize="gercek")   # satır toplamı 1, köşegen = recall
karmasiklik_matrisi(y, t, normalize="tahmin")   # sütun toplamı 1, köşegen = precision
karmasiklik_matrisi(y, t, normalize="tumu")     # toplam 1
```

Rapora **ham sayıları** koyun. Yüzdeler destek (support) bilgisini gizler;
"%100 recall" 3 örnekten geliyorsa bunu bilmek gerekir.

---

## 2. Accuracy (doğruluk)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

En sezgisel metrik ve en çok yanıltan metrik. Ders notunun uyarısı yerinde:
*"dengesiz veri kümelerinde yanıltıcı olabilir."*

**Somut örnek:** 100 hastanın 2'sinde nadir bir hastalık var. "Kimse hasta
değil" diyen model:
- accuracy = %98
- recall = 0
- F1 = 0
- dengeli doğruluk = 0.5 (yani şans seviyesi)

```python
from gozetimli.metrikler.siniflandirma import dogruluk, dengeli_dogruluk
dogruluk(y, t)          # 0.98  — gurur verici ve yalan
dengeli_dogruluk(y, t)  # 0.50  — gerçek
```

---

## 3. Precision ve Recall

```
Precision = TP / (TP + FP)    "pozitif dediklerimin kaçı gerçekten pozitif?"
Recall    = TP / (TP + FN)    "gerçek pozitiflerin kaçını yakaladım?"
```

İkisi arasında **kaçınılmaz bir ödünleşme** vardır: eşiği düşürürseniz daha çok
pozitif tahmin edersiniz, recall artar ama precision düşer. Eşiği yükseltirseniz
tersi olur.

| Alan | Hangisi kritik | Neden |
|---|---|---|
| Spam filtresi | Precision | Gerçek postayı spam'e atmak kullanıcıyı kaybettirir |
| Hastalık teşhisi | Recall | Hastayı sağlıklı ilan etmek geri dönülemez |
| Dolandırıcılık tespiti | Duruma göre | Blok maliyeti vs. kayıp maliyeti |
| Arama motoru ilk sayfa | Precision | Kullanıcı 10 sonuca bakar |
| Hukuki delil taraması | Recall | Kaçan belge davayı kaybettirir |

**Özgüllük (specificity)** = TN / (TN + FP), ROC eğrisinin x ekseninin
tamamlayanıdır (FPR = 1 - özgüllük). Tıpta recall "duyarlılık", specificity
"seçicilik" olarak geçer.

---

## 4. F1 ve F-beta

```
F1    = 2 · P · R / (P + R)
F-beta = (1 + β²) · P · R / (β² · P + R)
```

F1, precision ve recall'un **harmonik ortalamasıdır** — aritmetik değil. Fark
önemli: P = 1.00, R = 0.01 olan bir model
- aritmetik ortalamada 0.505 (kabul edilebilir görünür)
- F1'de 0.0198 (gerçeği söyler)

Harmonik ortalama küçük değere yakın durur; iki metrikten biri çökmüşse F1 de
çöker. F1'in varlık sebebi budur.

```python
from gozetimli.metrikler.siniflandirma import f1_skoru, f_beta_skoru
f1_skoru(y, t)
f_beta_skoru(y, t, beta=2.0)    # recall 2 kat önemli
f_beta_skoru(y, t, beta=0.5)    # precision 2 kat önemli
```

**Uyarı — makro F1:** sınıf başına F1'lerin ortalamasıdır. Makro P ile makro
R'den F1 formülüyle yeniden hesaplamak farklı (ve yanlış) bir sayı verir.

---

## 5. Çok sınıflı ortalama stratejileri

| Strateji | Nasıl | Ne zaman |
|---|---|---|
| `"makro"` | Sınıf skorlarının düz ortalaması | Her sınıf eşit önemli; azınlık sınıfı önemli |
| `"agirlikli"` | Destekle ağırlıklı ortalama | Gerçek dağılım yansısın isteniyor |
| `"mikro"` | TP/FP/FN havuzlanır | Tek etiketli çok sınıflıda = accuracy |
| `None` | Sınıf başına dizi | Ayrıntılı inceleme |

```python
kesinlik(y, t, ortalama="makro")       # azınlık sınıfını gizlemez
kesinlik(y, t, ortalama="agirlikli")   # büyük sınıflar baskın
kesinlik(y, t, ortalama=None)          # [sınıf0, sınıf1, sınıf2]
```

**Makro ile ağırlıklı arasındaki fark neyi ele verir?** Makro belirgin şekilde
düşükse model küçük sınıflarda başarısızdır ama ağırlıklı ortalama bunu
gizliyordur. İkisini yan yana raporlayın.

---

## 6. MCC ve Cohen's Kappa — tek sayıda dürüstlük

**Matthews korelasyon katsayısı**, karmaşıklık matrisinin dört hücresini birden
kullanır ve [-1, +1] aralığında değer alır:

```
MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

- +1 kusursuz, 0 şanstan farksız, -1 tam ters tahmin
- F1'den farkı: **TN'yi de hesaba katar**. F1, doğru reddedilen negatifleri
  görmezden gelir; dengesiz veride bu bilgi kaybettirir.
- Dengesiz veride tek sayı raporlanacaksa en dürüst seçenek MCC'dir.

**Cohen's kappa**, "sınıf dağılımını bilen rastgele tahminci"ye göre ne kadar
iyi olunduğunu ölçer:

```
κ = (p_gözlenen - p_şans) / (1 - p_şans)
```

Kaba yorum: < 0.20 zayıf, 0.40-0.60 orta, > 0.80 güçlü uyum. Bu eşikler
gelenekseldir, kanun değildir.

---

## 7. Log-loss — olasılığın kalitesi

```
L = -(1/n) Σ Σ y_ik · log(p_ik)
```

Lojistik regresyonun optimize ettiği kayıp fonksiyonudur (ders notu bölüm 5).
Sert etiketi değil **olasılığı** cezalandırır:

| Gerçek | Tahmin | Katkı |
|---|---|---|
| 1 | 0.99 | 0.01 — neredeyse bedava |
| 1 | 0.51 | 0.67 |
| 1 | 0.01 | 4.61 — çok pahalı |

Aynı accuracy'ye sahip iki model çok farklı log-loss alabilir. Modelin çıktısı
bir eşikten geçirilip atılacaksa log-loss önemsizdir; ama olasılık bir karara
girdi olacaksa (fiyatlama, beklenen değer hesabı, risk skoru) **kalibrasyon
accuracy'den önemlidir** ve log-loss onu ölçer.

Ders notunun MSE yerine log-loss kullanılması gerekçesi de doğrudur: MSE ile
lojistik regresyonun kayıp yüzeyi konveks olmaz, gradient descent yerel
minimumlara takılabilir.

---

## 8. ROC ve PR eğrileri — eşikten bağımsız değerlendirme

Şimdiye kadarki metriklerin hepsi **bir eşik seçildikten sonra** hesaplanır.
Eğriler tüm eşikleri birden değerlendirir.

**ROC:** her eşik için (FPR, TPR) çifti.
**ROC-AUC yorumu:** rastgele bir pozitif örneğe, rastgele bir negatiften daha
yüksek skor verme olasılığı. 0.5 = yazı tura.

```python
from gozetimli.metrikler.siniflandirma import roc_egrisi, roc_auc
fpr, tpr, esikler = roc_egrisi(y, olasiliklar)
roc_auc(y, olasiliklar)
```

AUC yalnızca **sıralamaya** bakar; skorları 100 ile çarpmak ya da logaritmasını
almak AUC'yi değiştirmez.

**PR eğrisi:** her eşik için (recall, precision) çifti. Çok dengesiz veride
ROC'dan daha ayırt edicidir çünkü taban çizgisi pozitif sınıf oranıdır.

```python
from gozetimli.metrikler.siniflandirma import kesinlik_duyarlilik_egrisi, ortalama_kesinlik
kesinlikler, duyarliliklar, esikler = kesinlik_duyarlilik_egrisi(y, olasiliklar)
ortalama_kesinlik(y, olasiliklar)     # AP = Σ (R_n - R_{n-1}) · P_n
```

AP, trapez değil **basamak** toplamıyla hesaplanır; trapez PR eğrisinde iyimser
sapma yapar. Bu implementasyon scikit-learn ile birebir aynı sözleşmeyi
kullanır (eşikler artan, recall azalan) ve
[`test_sklearn_uyumu.py`](../../testler/test_sklearn_uyumu.py) bunu her koşuda
doğrular.

---

## 9. Eşik seçimi

```python
from gozetimli.metrikler.siniflandirma import esik_tara

esik, skor, tum_esikler, tum_skorlar = esik_tara(y_dogrulama, olasiliklar, metrik="f1")
esik, _, _, _ = esik_tara(y_dogrulama, olasiliklar, metrik="youden")
esik, _, _, _ = esik_tara(y_dogrulama, olasiliklar,
                          metrik="kesinlik", en_az_duyarlilik=0.90)
```

`metrik` seçenekleri: `"f1"`, `"f2"`, `"f0.5"`, `"youden"` (TPR - FPR),
`"kesinlik"`, `"duyarlilik"`.

Eşik **doğrulama** setinde seçilir. Test setinde seçilen eşik test skorunu
şişirir.

---

## 10. Hızlı referans

```python
from gozetimli.metrikler.siniflandirma import siniflandirma_raporu

print(siniflandirma_raporu(y_test, tahminler, metin=True))
```

```
             sinif   kesinlik  duyarlilik       f1   destek
                 0     0.8571      0.9231   0.8889       13
                 1     0.8000      0.6667   0.7273        6

          dogruluk                          0.8421       19
    makro ortalama     0.8286      0.7949   0.8081       19
agirlikli ortalama     0.8391      0.8421   0.8379       19
```

Sınıf 1'in recall'u (0.6667) accuracy'nin (0.8421) belirgin altında: model
azınlık sınıfının üçte birini kaçırıyor. Makro F1 (0.8081) ile ağırlıklı F1
(0.8379) arasındaki fark da aynı şeyi söylüyor. Tek başına accuracy bunların
hiçbirini göstermezdi.

| Fonksiyon | Aralık | Yüksek = iyi |
|---|---|---|
| `dogruluk` | [0, 1] | evet |
| `kesinlik`, `duyarlilik`, `ozgulluk` | [0, 1] | evet |
| `f1_skoru`, `f_beta_skoru` | [0, 1] | evet |
| `dengeli_dogruluk` | [0, 1] | evet (0.5 = şans) |
| `matthews_korelasyonu` | [-1, 1] | evet (0 = şans) |
| `cohen_kappa` | [-1, 1] | evet (0 = şans) |
| `log_kaybi` | [0, ∞) | **hayır — düşük iyi** |
| `roc_auc` | [0, 1] | evet (0.5 = şans) |
| `ortalama_kesinlik` | [0, 1] | evet (taban = pozitif oranı) |
