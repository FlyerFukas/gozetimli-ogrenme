# SVM, KNN ve Naive Bayes

Kaynak ders notları: `10-Support_Vector_Machines.pdf`, `12-KNN_Algorithm.pdf`,
`11-Naive_Bayes_Theorem_ML_Algorithm.pdf`

Üç algoritma tek dosyada toplandı çünkü ortak bir yönleri var: **hepsi
ölçeklemeye ve özellik mühendisliğine doğrusal modellerden daha duyarlıdır.**

---

# Bölüm 1: Support Vector Machines

## 1.1 wᵀx + b = 0 ne demek?

Ders notunun bu bölümü değerli çünkü gösterim değişikliğini açıklıyor.
`y = β₀ + β₁x` ile `wᵀx + b = 0` aynı şeyi söyler, ama ikincisi boyuttan
bağımsızdır:

- 2 boyutta bir **doğru**
- 3 boyutta bir **düzlem**
- n boyutta bir **hiperdüzlem**

`w` ağırlık vektörü (hiperdüzlemin normali), `b` sabit terim.

## 1.2 Margin: SVM'in ayırt edici fikri

Sonsuz sayıda ayırıcı hiperdüzlem varken SVM **marjini en büyük olanı** seçer:

```
wᵀx + b = +1  ve  wᵀx + b = -1        (marjinal düzlemler)
Margin = 2 / ||w||
```

Marjini büyütmek = `||w||`'yi küçültmek. Optimizasyon problemi:

```
min (1/2)||w||²   koşul:  y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1
```

**Neden en geniş marjin?** Sezgisel gerekçe: karar sınırı verilerden ne kadar
uzaksa, yeni bir örnek küçük bir gürültüyle yanlış tarafa düşme riski o kadar
azdır. İstatistiksel öğrenme kuramında bunun genelleme sınırlarıyla bağı vardır.

**Destek vektörleri:** yalnızca marjin üzerindeki (ve ihlal eden) noktalar
çözümü belirler. Diğer tüm noktalar silinse model değişmez. Bu, SVM'i bellek
açısından verimli yapar ama aykırı değerlere karşı hassaslaştırır.

## 1.3 Hard margin vs soft margin

Gerçek veri kusursuz ayrılmaz. Soft margin, ihlallere izin verip cezalandırır:

```
min (1/2)||w||² + C Σ ξᵢ      koşul: y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
```

`ξᵢ` (slack) ihlal miktarı, `C` ceza katsayısı.

| C | Davranış |
|---|---|
| Küçük | Geniş marjin, çok ihlal toleransı: daha basit model, aşırı öğrenme riski düşük |
| Büyük | Dar marjin, az tolerans: eğitim verisine sıkı uyum, aşırı öğrenme riski yüksek |

`C`, düzenlileştirmenin **tersidir:** lojistik regresyondaki `C` ile aynı mantık.

Bu formülasyonun kayıp fonksiyonu **hinge loss**'tur: `max(0, 1 - y·f(x))`.
Log-loss'tan farkı: doğru tarafta ve marjin dışında olan noktalara **sıfır**
ceza verir. Lojistik regresyon her noktaya küçük de olsa ceza verir.

## 1.4 SVR: regresyon için SVM

SVR mantığı ters çevirir: marjin içinde **kalan** tahminleri ödüllendirir.

**ε-duyarsız kayıp:**
```
L(y, ŷ) = 0                  eğer |y - ŷ| ≤ ε
        = |y - ŷ| - ε        aksi halde
```

Tahmin eğrisinin etrafında ε genişliğinde bir "tüp" vardır; tüpün içindeki
sapmalar **görmezden gelinir.**

| | Doğrusal regresyon | SVR |
|---|---|---|
| Kayıp | Her noktanın kare hatası | Yalnızca tüp dışı, doğrusal |
| Aykırı değer | Çok duyarlı | Dayanıklı |
| Sonuç | Tüm noktalara uyum arar | "Yeterince iyi"yi kabul eder |

## 1.5 Kernel trick

Doğrusal ayrılamayan veri için: veriyi daha yüksek boyuta taşı, orada ayır.
Kernel trick bu dönüşümü **açıkça hesaplamadan** yapar, yalnızca iç çarpımlar
gerektiği için `K(x, x')` fonksiyonu yeterlidir.

| Kernel | Formül | Ne zaman |
|---|---|---|
| Lineer | `xᵀx'` | Yüksek boyut, metin, p > n |
| Polinomial | `(γxᵀx' + r)^d` | Etkileşim terimleri önemliyse |
| RBF (Gaussian) | `exp(-γ‖x - x'‖²)` | **Varsayılan.** Yerel, esnek |
| Sigmoid | `tanh(γxᵀx' + r)` | Nadiren; genelde RBF daha iyi |

**RBF'in γ parametresi:** tek bir örneğin etkisinin ne kadar uzağa yayıldığını
belirler. Büyük γ = dar etki alanı = çok esnek = aşırı öğrenme. `C` ile
birlikte ızgara aramasıyla seçilir.

## 1.6 Uygulama notları

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

boru = Pipeline([("olcek", StandardScaler()),
                 ("model", SVC(kernel="rbf", C=1.0, gamma="scale"))])
```

- **Ölçekleme zorunludur.** Mesafe/iç çarpım tabanlı olduğu için ölçeklenmemiş
  veride büyük ölçekli özellik kerneli tamamen ele geçirir.
- **Olasılık çıktısı pahalıdır.** `probability=True` içeride Platt ölçekleme
  için ek CV çalıştırır; gerekmiyorsa açmayın.
- **Ölçeklenebilirlik:** eğitim maliyeti kabaca O(n²)-O(n³). ~50 000 örnekten
  sonra `LinearSVC` veya `SGDClassifier` düşünün.

---

# Bölüm 2: K-Nearest Neighbors

## 2.1 Fikir

Eğitim aşaması yoktur; veri saklanır (lazy learning). Tahmin anında:

1. Yeni noktaya en yakın k komşu bulunur
2. **Sınıflandırma:** komşuların çoğunluk etiketi
3. **Regresyon:** komşuların hedef ortalaması

Ders notundaki 6 satırlık örnek (boy/kilo → sporcu) tam olarak bunu gösterir ve
[`testler/test_ders_ornekleri.py`](../../testler/test_ders_ornekleri.py)
içinde doğrulanmıştır.

## 2.2 Mesafe ölçüleri

```
Öklid    : d = √Σ(xᵢ - yᵢ)²
Manhattan: d = Σ|xᵢ - yᵢ|
```

Minkowski ailesinin p=2 ve p=1 halleridir. Yüksek boyutta Manhattan bazen daha
kararlıdır.

## 2.3 Ölçekleme: KNN'in en kritik konusu

Ders notu doğru söylüyor: KNN *"özellikler benzer ölçekteyse"* çalışır.

**Neden bu kadar kritik:** mesafe hesabı ölçeğin karesiyle büyür. Gelir (30 000
-90 000) ve yaş (20 - 60) aynı mesafede kullanılırsa, gelir farkı yaş farkını
tamamen bastırır, model fiilen tek özellikle çalışır.

Test dosyasındaki `test_olcekleme_tahmini_degistirebilir`, ölçeklemenin k-NN
tahminini **ters çevirdiği** somut bir örnek içeriyor.

Her zaman Pipeline içinde ölçekleyin.

## 2.4 k seçimi

| k | Davranış |
|---|---|
| 1 | Sıfır bias, maksimum varyans. Gürültülü etikete tam uyum |
| Küçük (3-5) | Esnek, yerel |
| Büyük | Düzgün sınır, yüksek bias; k = n ise sabit tahmin |

İkili sınıflandırmada **tek sayı** seçin (beraberlik olmasın). CV ile seçilir.

## 2.5 Ölçeklenebilirlik: KD-Tree ve Ball Tree

Naif KNN her sorguda tüm eğitim verisiyle mesafe hesaplar: O(n·p).

- **KD-Tree:** veriyi eksenlere göre özyinelemeli böler. Düşük boyutta (p < 20)
  çok hızlı.
- **Ball Tree:** veriyi hiyerarşik kürelere ayırır. Yüksek boyutta KD-Tree'den
  iyi.

**Boyutun laneti:** boyut arttıkça tüm noktalar birbirine eşit uzaklıkta
görünmeye başlar, "en yakın komşu" kavramı anlamını yitirir. Bu, ağaç
yapılarının da çözemediği yapısal bir sınırdır. p > 20-30 ise önce boyut
indirgeme (PCA) veya başka bir algoritma düşünün.

## 2.6 Uygulama

```python
from sklearn.neighbors import KNeighborsClassifier
Pipeline([("olcek", StandardScaler()),
          ("model", KNeighborsClassifier(n_neighbors=5, weights="distance"))])
```

`weights="distance"` yakın komşulara daha çok oy verir, genellikle uniform'dan
iyidir.

---

# Bölüm 3: Naive Bayes

## 3.1 Bayes teoreminden sınıflandırıcıya

```
P(y|x₁...xₙ) = P(y) · P(x₁|y) · ... · P(xₙ|y) / P(x₁...xₙ)
```

Payda tüm sınıflar için aynı olduğundan karar verirken atılır:

```
ŷ = argmax_y  P(y) · Π P(xᵢ|y)
```

**"Naive" (saf) nereden geliyor?** Özelliklerin sınıf verildiğinde birbirinden
bağımsız olduğu varsayımından. Bu varsayım neredeyse her zaman yanlıştır
("ücretsiz" ve "kazandınız" kelimeleri spam içinde birlikte geçme eğilimindedir).

**Yine de neden çalışıyor?** Çünkü sınıflandırma için olasılıkların **doğru**
olması gerekmez; yalnızca **doğru sıralanması** gerekir. Bağımlılık her iki
sınıfın skorunu benzer yönde bozarsa argmax değişmez. Bu yüzden Naive Bayes
kalibrasyonu kötü (olasılıkları uç değerlere yığılır) ama sınıflandırması iyi
bir modeldir, olasılık çıktısını doğrudan kullanmayın.

## 3.2 Sıfır olasılık sorunu ve Laplace düzeltmesi

Ders notundaki örnekte P(Free=Yes|Ham) = 0 çıkıyor ve tüm çarpımı sıfırlıyor.
Tek bir görülmemiş kombinasyon, diğer tüm kanıtları siliyor.

**Laplace (add-one) düzeltmesi:**
```
P(xᵢ|y) = (sayım + α) / (toplam + α · olası_değer_sayısı)
```

α = 1 ile: `(0+1)/(2+2) = 0.25`. scikit-learn'de `alpha` parametresi; metin
sınıflandırmada 0.1-1.0 arası tipik.

## 3.3 Üç varyant

| Varyant | Veri tipi | Kullanım |
|---|---|---|
| **Bernoulli** | İkili (0/1) | Kelime var/yok, tıklandı/tıklanmadı |
| **Multinomial** | Sayım / frekans | Metin sınıflandırma (bag-of-words, TF-IDF) |
| **Gaussian** | Sürekli | Yaş, boy, kilo, sensör ölçümü |

Gaussian varyantta her özellik için sınıf başına μ ve σ hesaplanır
(**ddof=0**, popülasyon), olasılık normal dağılım yoğunluğundan gelir:

```
P(xᵢ|y) = 1/√(2πσ²) · exp(-(xᵢ - μ)²/(2σ²))
```

Ders notundaki obezite örneğinin tüm ara değerleri
[`test_ders_ornekleri.py::TestNaiveBayes`](../../testler/test_ders_ornekleri.py)
içinde doğrulanmıştır.

**Karışık özellik tipi varsa:** çoğunluk ikiliyse Bernoulli, çoğunluk
sürekliyse Gaussian; ya da kategorikleri sayısallaştırıp Gaussian. En temiz
çözüm, her tip için ayrı NB eğitip log-olasılıkları toplamaktır.

## 3.4 Artı ve eksileri

**Artı:**
- Çok hızlı (tek geçiş), az bellek
- Yüksek boyutta iyi çalışır: metin sınıflandırmanın klasik baz çizgisidir
- Küçük veriyle bile makul sonuç verir
- Eksik veriyle çalışabilir (o özellik çarpımdan düşer)

**Eksi:**
- Bağımsızlık varsayımı gerçekçi değil
- Olasılıkları kalibre değil (0 veya 1'e yığılır)
- Sürekli veride dağılım varsayımı gerekir
- Özellikler güçlü korelasyonluysa o kanıt birden çok kez sayılır

---

## Üçünün karşılaştırması

| | SVM | KNN | Naive Bayes |
|---|---|---|---|
| Eğitim maliyeti | Yüksek O(n²-n³) | **Sıfır** | **Çok düşük** |
| Tahmin maliyeti | Düşük | **Yüksek** | Çok düşük |
| Ölçekleme gerekli | **Evet** | **Evet** | Hayır |
| Yüksek boyut | İyi (lineer kernel) | **Kötü** | **İyi** |
| Yorumlanabilirlik | Düşük | Orta ("bu komşulara benziyor") | Orta |
| Olasılık kalitesi | Ek kalibrasyon gerekir | Orta | **Kötü** |
| Aykırı değer | Hassas (destek vektörü olur) | Hassas | Dayanıklı |

---

## İlgili

- [Metrik seçim rehberi](../04-degerlendirme/metrik-secim-rehberi.md)
- [Çapraz doğrulama](../04-degerlendirme/capraz-dogrulama.md)
