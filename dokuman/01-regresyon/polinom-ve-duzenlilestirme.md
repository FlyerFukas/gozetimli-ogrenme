# Polinom regresyon ve düzenlileştirme (Ridge / Lasso / Elastic Net)

Kaynak ders notları: `5-Polynomial_Regression.pdf`,
`6-Ridge_Lasso_Elastic_Net_Regression.pdf`

---

## 1. Polinom regresyon — doğrusal modelin esnetilmesi

Doğrusal regresyon `ŷ = β₀ + β₁x` bir doğru çizer. Veri eğrisel bir desen
izliyorsa doğru asla yeterli olmaz. Polinom regresyon girdiyi yüksek dereceli
terimlerle genişletir:

```
ŷ = β₀ + β₁x + β₂x² + ... + βₙxⁿ
```

**Kritik nokta:** bu hâlâ **doğrusal bir modeldir.** Doğrusallık x'e göre değil,
**katsayılara (β) göre** tanımlanır. x², x³ sütunlarını hesaplayıp sıradan en
küçük kareler uygularsınız; çözüm yöntemi değişmez.

### Derece seçimi bir ödünleşmedir

| Derece | Davranış |
|---|---|
| 0 | Sabit tahmin (ortalama) |
| 1 | Basit doğrusal regresyon |
| 2–3 | Çoğu eğrisel ilişki için yeterli |
| Yüksek | Eğitim verisine kusursuz uyum, test verisinde felaket |

Ders notunun sonuç bölümü doğru: *"Düşük dereceler genellikle yeterlidir"* ve
*"Modelin başarısı doğrulama verisi ile test edilmelidir."*

Derece bir **hiperparametredir**; çapraz doğrulama ile seçilir:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from gozetimli.dogrulama.bolme import KKat
from gozetimli.dogrulama.capraz import capraz_dogrula
from gozetimli.metrikler.regresyon import rmse, r2

for derece in (1, 2, 3, 5, 9):
    boru = Pipeline([
        ("poli", PolynomialFeatures(degree=derece, include_bias=False)),
        ("olcek", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    sonuc = capraz_dogrula(boru, X, y, bolucu=KKat(5, karistir=True, tohum=0),
                           metrikler={"rmse": rmse, "r2": r2}, egitim_skoru=True)
    print(derece, sonuc.ozet()["rmse"])
```

**Ölçeklemeyi atlamayın:** x = 100 iken x⁵ = 10¹⁰'dur. Ölçeklenmemiş polinom
terimleri sayısal olarak kararsızdır ve düzenlileştirmeyi anlamsızlaştırır.

### Özellik patlaması

`PolynomialFeatures` etkileşim terimlerini de üretir. p özellik, d derece için
terim sayısı C(p+d, d)'dir:

| p | d | Terim sayısı |
|---|---|---|
| 5 | 2 | 20 |
| 10 | 3 | 285 |
| 20 | 3 | 1 770 |

Bu yüzden polinom genişletme neredeyse her zaman düzenlileştirme ile birlikte
kullanılır.

---

## 2. Aşırı öğrenme (overfitting)

Ders notundaki tarif tam olarak doğru:

| | Aşırı öğrenen model |
|---|---|
| Eğitim doğruluğu | Yüksek (hatta %100) |
| Test doğruluğu | Düşük |
| Bias (yanlılık) | Düşük |
| Varyans | Yüksek |

Model, verinin altındaki **ilişkiyi** değil, o örneklem parçasındaki
**gürültüyü** öğrenmiştir. Teşhis için öğrenme eğrisi:
[capraz-dogrulama.md § 8](../04-degerlendirme/capraz-dogrulama.md).

Düzenlileştirmenin fikri: katsayıların büyümesini **maliyet fonksiyonunda
cezalandırmak.** Büyük katsayı = keskin tepkiler = gürültüye uyum.

---

## 3. Ridge (L2)

```
J(β) = (1/2m) Σ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + λ Σ βⱼ²
```

Cezaya katsayıların **karesi** girer.

| λ | Etki |
|---|---|
| 0 | Klasik doğrusal regresyon |
| Artan | Katsayılar küçülür, model yumuşar |
| → ∞ | Katsayılar sıfıra yaklaşır ama **tam sıfır olmaz** |

**Ne zaman:** özellikler arasında çoklu doğrusal bağlantı (multicollinearity)
varsa. İki özellik yüksek korelasyonluysa sıradan en küçük kareler
katsayıları kararsız olur (biri +1000, öbürü -1000 gibi); Ridge ikisini de
küçültüp aralarında paylaştırır.

**Ne zaman değil:** özellik sayısı çok fazlaysa ve sadeleştirme isteniyorsa.
Ridge hiçbir özelliği elemez.

---

## 4. Lasso (L1)

```
J(β) = (1/2m) Σ (ŷᵢ - yᵢ)² + λ Σ |βⱼ|
```

Cezaya katsayıların **mutlak değeri** girer. Tek harf farkı, davranışı
tamamen değiştirir: **bazı katsayılar tam olarak sıfır olur.**

**Neden sıfırlıyor?** L1 cezasının geometrisi köşelidir (elmas biçimli kısıt
bölgesi). Kayıp fonksiyonunun eş-yükselti eğrileri bu bölgeye genellikle bir
**köşede** teğet olur; köşede en az bir koordinat sıfırdır. L2'nin kısıt
bölgesi (daire) köşesizdir, bu yüzden tam sıfır çıkmaz.

Sonuç: Lasso aynı anda hem düzenlileştirme hem **otomatik özellik seçimi**
yapar. Ders notunun uyarısı da yerinde: *"çok fazla değişkeni sıfırlayabilir"* —
zayıf ama gerçek etkileri olan özellikler de elenebilir.

**Korelasyonlu özelliklerde davranışı:** iki özellik neredeyse aynıysa Lasso
birini seçip diğerini sıfırlar. Hangisini seçeceği veriye ve rastgeleliğe
bağlıdır; bu, yorumlanabilirlik açısından kırılganlık demektir.

---

## 5. Elastic Net (L1 + L2)

```
J(β) = (1/2m) Σ (ŷᵢ - yᵢ)² + λ₁ Σ βⱼ² + λ₂ Σ |βⱼ|
```

İkisini birleştirir. Korelasyonlu özellik gruplarında Lasso'nun keyfî seçim
sorununu çözer: **grubu birlikte** tutar veya birlikte eler ("grouping effect").

scikit-learn'de iki parametreyle ifade edilir: `alpha` (toplam ceza gücü) ve
`l1_ratio` (L1'in payı; 0 = saf Ridge, 1 = saf Lasso).

---

## 6. Karşılaştırma

| Özellik | Ridge (L2) | Lasso (L1) | Elastic Net |
|---|---|---|---|
| Ceza | β² | \|β\| | ikisi |
| Aşırı öğrenmeyi azaltır | evet | evet | evet |
| Değişken seçimi | **hayır** | evet | evet |
| Katsayıyı tam sıfırlar | hayır | evet | evet |
| Korelasyonlu grup | hepsini küçültür | birini seçer | grubu birlikte tutar |
| p > n durumu | çalışır | en fazla n özellik seçer | çalışır |

---

## 7. Uygulamada üç kural

**1. Ölçekleme zorunludur.** Ceza terimi katsayının büyüklüğüne bakar;
katsayının büyüklüğü özelliğin birimine bağlıdır. Metre yerine santimetre
kullanmak, o özelliğin cezasını 100 kat değiştirir. Her zaman `StandardScaler`
ile birlikte, Pipeline içinde kullanın.

**2. λ (alpha) bir hiperparametredir.** Doğrulama verisiyle seçilir. Logaritmik
bir ızgara makuldür: `[0.001, 0.01, 0.1, 1, 10, 100]`.

**3. Seçim ve değerlendirme ayrı olmalı.** λ'yı CV ile seçip aynı CV skorunu
raporlamak iyimserdir; nested CV veya ayrı test seti gerekir.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from gozetimli.dogrulama.bolme import KKat
from gozetimli.dogrulama.capraz import ic_ice_capraz_dogrula
from gozetimli.metrikler.regresyon import rmse

def kur(alpha):
    return Pipeline([("olcek", StandardScaler()), ("model", Ridge(alpha=alpha))])

skorlar, secilen = ic_ice_capraz_dogrula(
    lambda **p: kur(**p), X, y,
    dis_bolucu=KKat(5, karistir=True, tohum=0),
    ic_bolucu=KKat(4, karistir=True, tohum=0),
    izgara=[{"alpha": a} for a in (0.001, 0.01, 0.1, 1, 10, 100)],
    metrik=rmse, buyuk_iyi=False)
```

---

## İlgili

- [Regresyon metrikleri](../04-degerlendirme/regresyon-metrikleri.md)
- [Çapraz doğrulama](../04-degerlendirme/capraz-dogrulama.md)
