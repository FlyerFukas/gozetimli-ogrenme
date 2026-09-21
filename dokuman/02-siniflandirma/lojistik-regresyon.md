# Lojistik regresyon

Kaynak ders notları: `8-Logistic_Regression.pdf`,
`9-Logistic_Regression_Performance_Metrics.pdf`

---

## 1. Neden doğrusal regresyon sınıflandırmada yetmez?

Ders notundaki iki gerekçe doğru ve tamamlanmaya değer:

1. **Çıktı aralığı.** Doğrusal regresyon çıktısı (-∞, +∞) aralığındadır.
   "Olasılık 1.7" veya "olasılık -0.3" anlamsızdır.
2. **Aykırı değer duyarlılığı.** Sınıfından çok uzakta duran tek bir nokta,
   kareli hatayı düşürmek için doğrunun eğimini değiştirir ve karar sınırını
   kaydırır. Sınıflandırmada bu noktanın "ne kadar uzakta" olduğu önemsizdir;
   önemli olan hangi tarafta olduğudur.

Üçüncü bir gerekçe daha var: **kayıp fonksiyonu.** MSE ile lojistik model
eğitilirse maliyet yüzeyi konveks olmaz, gradient descent yerel minimumlara
takılabilir. Log-loss bu sorunu çözer.

---

## 2. Sigmoid ve hipotez

```
σ(z) = 1 / (1 + e⁻ᶻ)
h(x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
```

Sigmoid, (-∞, +∞) aralığını (0, 1)'e sıkıştırır. z = 0'da tam 0.5 verir, S
biçimlidir.

```python
from gozetimli.boosting import sigmoid, log_odds
sigmoid(0.0)      # 0.5
log_odds(0.75)    # 1.0986 , sigmoid'in tersi
```

Bu implementasyon taşma güvenlidir: naif `1/(1+exp(-z))` formülü z = -1000'de
`exp(1000)` hesaplayıp taşar.

### Katsayıların yorumu

Lojistik regresyonun en güçlü yanı yorumlanabilirliğidir:

```
log(p / (1-p)) = β₀ + β₁x₁ + ... + βₙxₙ
```

Sol taraf **log-odds**'tur. Yani: x₁ bir birim arttığında log-odds β₁ kadar
artar, **odds** ise e^β₁ **katına** çıkar.

| β₁ | e^β₁ | Yorum |
|---|---|---|
| 0.69 | 2.0 | Bir birim artış, odds'u 2 katına çıkarır |
| 0 | 1.0 | Etkisi yok |
| -0.69 | 0.5 | Odds'u yarıya düşürür |

Bu yüzden lojistik regresyon, kredi skorlama ve klinik risk modellerinde hâlâ
tercih edilir: her katsayı bir cümleyle açıklanabilir.

---

## 3. Karar kuralı: 0.5 bir varsayımdır

Ders notundaki kural `ŷ = 1 eğer h(x) ≥ 0.5` doğrudur ama **koşulludur.**
0.5 yalnızca şu iki koşulda optimaldir:

1. Sınıflar dengelidir
2. FP ile FN aynı maliyettedir

İkisi de çoğu gerçek problemde yanlıştır.

```python
from gozetimli.metrikler.siniflandirma import esik_tara
esik, skor, _, _ = esik_tara(y_dogrulama, olasiliklar, metrik="f1")
```

Eşik **doğrulama** setinde seçilir. Ayrıntı:
[metrik-secim-rehberi.md § 5](../04-degerlendirme/metrik-secim-rehberi.md).

---

## 4. Log-loss (çapraz entropi)

```
J(β) = -(1/m) Σ [ y⁽ⁱ⁾ log(h(x⁽ⁱ⁾)) + (1 - y⁽ⁱ⁾) log(1 - h(x⁽ⁱ⁾)) ]
```

İki terimden biri her örnekte sıfırlanır: y = 1 ise `-log(h)`, y = 0 ise
`-log(1-h)` kalır. Doğru sınıfa verilen olasılık 1'e yakınsa ceza ~0, 0'a
yakınsa ceza patlar.

Bu fonksiyon **konvekstir:** tek bir global minimumu vardır ve gradient
descent ona ulaşır.

```python
from gozetimli.metrikler.siniflandirma import log_kaybi
log_kaybi(y, olasiliklar)
```

**Ne zaman önemser:** modelin çıktısı bir eşikten geçirilip atılacaksa log-loss
önemsizdir. Ama olasılık bir karara girdi olacaksa (beklenen değer hesabı, risk
tabanlı fiyatlama, sıralama) **kalibrasyon accuracy'den önemlidir.**

---

## 5. Çok sınıflı stratejiler

### One-vs-Rest (OvR)

K sınıf için K ikili sınıflandırıcı. Her biri "ben mi, diğerleri mi?" sorusunu
çözer; en yüksek olasılığı veren kazanır.

- **Model sayısı:** K
- **Artı:** ucuz, yorumlanabilir
- **Eksi:** her modelin eğitim verisi dengesizdir (1 sınıfa karşı K-1 sınıf);
  olasılıklar toplamı 1 etmez, normalleştirme gerekir

### One-vs-One (OvO)

Her sınıf çifti için bir model: K(K-1)/2 model. Oylama ile karar verilir.

- **Model sayısı:** K = 10 için 45 model
- **Artı:** her model yalnızca iki sınıfın verisiyle eğitilir: daha dengeli ve
  daha küçük eğitim setleri
- **Eksi:** model sayısı kareyle büyür

### Softmax (multinomial)

Üçüncü bir yol daha var (ders notunda yok): tek bir model, K çıktı,
normalleştirme softmax ile:

```
p(y=k|x) = e^(zₖ) / Σⱼ e^(zⱼ)
```

Olasılıklar doğrudan toplam 1 eder. scikit-learn'ün `LogisticRegression`
varsayılanı budur. Sınıflar birbirini dışlıyorsa (tek etiket) softmax genelde
OvR'den iyidir.

---

## 6. Uygulama notları

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from gozetimli.dogrulama.bolme import TabakaliKKat
from gozetimli.dogrulama.capraz import capraz_dogrula
from gozetimli.metrikler.siniflandirma import f1_skoru, roc_auc, log_kaybi

boru = Pipeline([
    ("olcek", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

sonuc = capraz_dogrula(boru, X, y, bolucu=TabakaliKKat(5, karistir=True, tohum=0),
                       metrikler={"f1": f1_skoru, "auc": roc_auc},
                       olasilik_metrikleri=("auc",))
```

| Konu | Not |
|---|---|
| **Ölçekleme** | Düzenlileştirme varsayılan olarak açıktır (L2, C=1.0), bu yüzden ölçekleme şart |
| **`C` parametresi** | Düzenlileştirmenin **tersidir**: küçük C = güçlü ceza |
| **Dengesiz sınıf** | `class_weight="balanced"` azınlık sınıfına ağırlık verir |
| **Yakınsama uyarısı** | `max_iter` artırın veya ölçeklemeyi kontrol edin |
| **Tam ayrılabilirlik** | Sınıflar kusursuz ayrılıyorsa katsayılar sonsuza gider; düzenlileştirme bunu engeller |

---

## 7. Ne zaman lojistik regresyon, ne zaman değil?

**Kullanın:**
- Yorumlanabilirlik zorunluysa (regülasyon, kredi, klinik)
- Kalibre olasılık gerekiyorsa
- Baz çizgisi kurarken: her zaman ilk denenecek model
- Özellik sayısı örnek sayısına yakınsa (düzenlileştirme ile)

**Kullanmayın:**
- İlişki güçlü şekilde doğrusal değilse (etkileşim/polinom terim eklemeden)
- Karmaşık etkileşimler varsa: ağaç tabanlı modeller daha iyi
- Özellikler arasında yüksek çoklu bağlantı varsa (katsayılar kararsızlaşır;
  Ridge cezası şart)

---

## İlgili

- [Sınıflandırma metrikleri](../04-degerlendirme/siniflandirma-metrikleri.md)
- [Metrik seçim rehberi](../04-degerlendirme/metrik-secim-rehberi.md)
- [Polinom ve düzenlileştirme](../01-regresyon/polinom-ve-duzenlilestirme.md)
