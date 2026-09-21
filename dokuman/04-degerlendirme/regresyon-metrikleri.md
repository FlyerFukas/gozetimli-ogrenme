# Regresyon metrikleri

Kod: [`gozetimli/metrikler/regresyon.py`](../../src/gozetimli/metrikler/regresyon.py) ·
Test: [`testler/test_regresyon_metrikleri.py`](../../testler/test_regresyon_metrikleri.py)

> **Kaynak notu.** Elimizdeki ders notlarında (5–19 numaralı PDF'ler) regresyon
> metrikleri ayrı bir başlık olarak işlenmiyor. MSE yalnızca doğrusal/Ridge/Lasso
> maliyet fonksiyonunun içinde, varyans ise karar ağacı bölme ölçütü olarak
> geçiyor. Klasörde 1–4 numaralı PDF'ler eksik; bu konuların orada anlatılmış
> olması muhtemel. Buradaki tanımlar standart literatürden alınmış ve
> scikit-learn'e karşı doğrulanmıştır. Ayrıntı: [KAYNAK-NOTLARI.md](../KAYNAK-NOTLARI.md).

Hangi metriği ne zaman seçeceğiniz için [metrik-secim-rehberi.md](metrik-secim-rehberi.md).

---

## 1. MSE, RMSE, MAE

```
MSE  = (1/n) Σ (y - ŷ)²
RMSE = √MSE
MAE  = (1/n) Σ |y - ŷ|
```

**MSE**, doğrusal regresyonun ve karar ağacı regresörünün varsayılan kaybıdır.
Ders notundaki maliyet fonksiyonu J(θ) = (1/2m) Σ (h(x) - y)² bunun 1/2 katıdır
(türev alırken sadeleşsin diye).

Birimi hedefin **karesidir** — "MSE 2500" cümlesi kimseye bir şey anlatmaz.
Raporlarken **RMSE** kullanın: hedefle aynı birimdedir, "ortalama 50 TL sapıyor"
denebilir.

**MAE**, hatayı karesine almadığı için aykırı değerlere dayanıklıdır.

**RMSE ≥ MAE her zaman doğrudur** (Jensen eşitsizliği); eşitlik yalnızca tüm
hatalar aynı büyüklükteyse olur. **Aradaki fark bir teşhis aracıdır:**

| Durum | Anlamı |
|---|---|
| RMSE ≈ MAE | Hatalar birbirine yakın büyüklükte, homojen |
| RMSE ≫ MAE (2 kat+) | Birkaç büyük hata ortalamayı taşıyor — o örneklere bakın |

```python
from gozetimli.metrikler.regresyon import mse, rmse, mae
mse(y, tahmin); rmse(y, tahmin); mae(y, tahmin)
```

### MSE mi MAE mi? Bu bir merkezi eğilim seçimidir

Sabit bir sayı tahmin etmek zorunda olsaydınız:

- **MSE**'yi minimize eden sayı **ortalamadır**
- **MAE**'yi minimize eden sayı **medyandır**

Bu yüzden MSE ile eğitilen model aykırı değerlere doğru çekilir. Karar ağacı
regresörünün yaprakta ortalama döndürmesi de bundandır: varyans azaltımı
ölçütü, ağırlıklı MSE azalışıyla aynı bölmeyi seçer.

### Dayanıklı alternatifler

```python
from gozetimli.metrikler.regresyon import medyan_mutlak_hata, maksimum_hata
medyan_mutlak_hata(y, tahmin)   # aykırı değerlere en dayanıklı
maksimum_hata(y, tahmin)        # en kötü tek örnek — SLA garantisi gerekiyorsa
```

100 örnekten birinde 100 birimlik hata varsa: MAE = 1, MSE = 100, medyan
mutlak hata = 0. Üçü de doğru, üçü farklı soruya cevap veriyor.

---

## 2. R² ve düzeltilmiş R²

```
R² = 1 - SS_kalıntı / SS_toplam = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

**Yorum:** "Modelim, her şeye ortalamayı söyleyen modelden ne kadar iyi?"

| R² | Anlamı |
|---|---|
| 1.0 | Kusursuz |
| 0.0 | Ortalamayla aynı — model hiçbir şey öğrenmemiş |
| < 0 | Ortalamadan **kötü** — test setinde bu gerçek bir uyarıdır |

Negatif R² şaşırtıcı gelebilir ama mümkündür ve önemlidir: modeliniz test
setinde, "her zaman eğitim ortalamasını söyle" stratejisinden daha kötü demektir.

**R²'nin tuzağı:** eğitim verisinde özellik eklendikçe **hiçbir zaman düşmez.**
Tamamen rastgele bir sütun bile eğitim R²'sini yükseltir. Bu yüzden farklı
özellik sayılı modelleri karşılaştırırken:

```
R²_düzeltilmiş = 1 - (1 - R²) · (n - 1) / (n - p - 1)
```

```python
from gozetimli.metrikler.regresyon import r2, duzeltilmis_r2
r2(y, tahmin)
duzeltilmis_r2(y, tahmin, ozellik_sayisi=X.shape[1])
```

Düzeltilmiş R², her yeni özelliğin bedelini ödetir: özellik gerçekten bilgi
katmıyorsa **düşer.**

### Açıklanan varyans ile farkı

```
Açıklanan varyans = 1 - Var(y - ŷ) / Var(y)
```

R²'den farkı: kalıntıların **ortalamasını** cezalandırmaz. Model sistematik
olarak sabit bir miktar sapıyorsa (yanlılık/bias), R² düşer ama açıklanan
varyans yüksek kalır.

**İkisi arasındaki fark, modelin yanlılığını ele verir.** Açıklanan varyans
belirgin şekilde R²'den büyükse modelinizde düzeltilebilir bir kayma vardır.

---

## 3. Yüzde tabanlı hatalar: MAPE ve sMAPE

```
MAPE  = (1/n) Σ |y - ŷ| / |y|
sMAPE = (1/n) Σ |y - ŷ| / ((|y| + |ŷ|)/2)
```

MAPE oran döndürür (0.12 = %12) ve iş birimlerinin sevdiği metriktir. **Üç
tuzağı vardır:**

1. **y = 0'da tanımsız.** Talep tahmininde sıfır satışlı günler varsa çöker.
   (Bu implementasyon epsilon ile korur ama sayı anlamsızlaşır.)
2. **Asimetriktir.** Düşük tahminin cezası en fazla 1.0'dır (tahmin 0 olsa
   bile); yüksek tahminin cezası sınırsızdır. Model sistematik olarak düşük
   tahmin etmeye itilir — talep tahmininde bu, stoksuz kalmak demektir.
3. **Küçük gerçek değerlerde patlar.** y = 1 iken ŷ = 2 vermek %100 hata; oysa
   mutlak hata 1 birim.

**sMAPE** paydayı simetrik yapar ve [0, 2] aralığında sınırlıdır; MAPE
sınırsızdır.

```python
from gozetimli.metrikler.regresyon import mape, smape
mape(y, tahmin)    # 0.12 = %12
smape(y, tahmin)
```

`regresyon_raporu`, hedefte sıfır varsa MAPE'yi otomatik olarak atlar.

---

## 4. Logaritmik hatalar: MSLE ve RMSLE

```
MSLE = (1/n) Σ (log(1+y) - log(1+ŷ))²
```

Mutlak hatayı değil **oransal** hatayı cezalandırır. Hedef üstel büyüyorsa
(gelir, nüfus, ziyaret sayısı, satış hacmi) doğru seçimdir:

| Gerçek → Tahmin | MSE katkısı | MSLE katkısı |
|---|---|---|
| 100 → 200 | 10 000 | ≈ 0.47 |
| 10 000 → 20 000 | 100 000 000 | ≈ 0.48 |

Aynı oran hatası (2 kat), MSLE'de neredeyse aynı cezayı alır. Sınır değeri
log(2)² ≈ 0.4805'tir.

Negatif değer kabul etmez. `log1p`'teki +1 sıfır hedefi mümkün kılar ama küçük
değerlerde oransal denkliği hafifçe bozar (bkz. test dosyası).

---

## 5. Hızlı rapor

```python
from gozetimli.metrikler.regresyon import regresyon_raporu
regresyon_raporu(y_test, tahminler, ozellik_sayisi=X.shape[1])
```

`y = [10, 20, 30, 40]`, `ŷ = [11, 19, 32, 38]` için gerçek çıktı:

```python
{'mse': 2.5, 'rmse': 1.5811, 'mae': 1.5, 'medyan_mutlak_hata': 1.5,
 'maksimum_hata': 2.0, 'r2': 0.98, 'aciklanan_varyans': 0.98,
 'mape': 0.0667, 'smape': 0.0656, 'duzeltilmis_r2': 0.97}
```

Burada `r2` ile `aciklanan_varyans` eşit, çünkü kalıntıların ortalaması sıfır
(+1, -1, +2, -2) — yani modelde sistematik kayma yok.

---

## 6. Metriğin ötesi: kalıntı grafiği

Hiçbir tek sayı, kalıntıları (y - ŷ) tahmine karşı çizen grafiğin yerini
tutmaz. Aranacak desenler:

| Desen | Teşhis |
|---|---|
| Rastgele dağılım, 0 etrafında | İyi — modelde kalan yapı yok |
| Huni şekli (yelpaze) | Değişen varyans — hedefin log'unu almayı deneyin |
| Eğri (U veya ters U) | Doğrusal olmayan ilişki kaçırılmış — polinom terim ekleyin |
| Sistematik kayma | Yanlılık — kesme terimi veya eksik özellik |
| Uç noktalarda kümelenme | Hedef kırpılmış/sansürlü olabilir |

---

## 7. Hızlı referans

| Fonksiyon | Birim | Aralık | Aykırıya dayanıklı | Yüksek = iyi |
|---|---|---|---|---|
| `mse` | hedef² | [0, ∞) | hayır | hayır |
| `rmse` | hedef | [0, ∞) | hayır | hayır |
| `mae` | hedef | [0, ∞) | orta | hayır |
| `medyan_mutlak_hata` | hedef | [0, ∞) | **evet** | hayır |
| `maksimum_hata` | hedef | [0, ∞) | hayır (tanımı gereği) | hayır |
| `mape` | oran | [0, ∞) | hayır | hayır |
| `smape` | oran | [0, 2] | orta | hayır |
| `msle` / `rmsle` | log | [0, ∞) | evet (oransal) | hayır |
| `r2` | — | (-∞, 1] | hayır | **evet** |
| `duzeltilmis_r2` | — | (-∞, 1] | hayır | **evet** |
| `aciklanan_varyans` | — | (-∞, 1] | hayır | **evet** |
