# Çapraz doğrulama: model nasıl test edilir?

Kaynak ders notu: `7-Types_of_Cross_Validation.pdf`.
Kod: [`gozetimli/dogrulama/`](../../src/gozetimli/dogrulama/) ·
Test: [`testler/test_bolme.py`](../../testler/test_bolme.py), [`testler/test_capraz.py`](../../testler/test_capraz.py)

---

## 1. Neden tek bir train/test bölmesi yetmez?

Tek bölmede raporladığınız skor, **o bölmenin şansını** da içerir. Küçük
veride bu şans birkaç puan oynatabilir, yani iki model arasındaki "fark"
tamamen bölme kaynaklı olabilir.

Çapraz doğrulama aynı veriyi k farklı şekilde bölüp k skor üretir. Size
**ortalamayı ve standart sapmayı** verir. Standart sapma olmadan raporlanan bir
CV ortalaması, bilginin yarısını atmaktır.

```python
from gozetimli.dogrulama.bolme import TabakaliKKat
from gozetimli.dogrulama.capraz import capraz_dogrula
from gozetimli.metrikler.siniflandirma import dogruluk, f1_skoru

sonuc = capraz_dogrula(model, X, y,
                       bolucu=TabakaliKKat(5, karistir=True, tohum=42),
                       metrikler={"dogruluk": dogruluk, "f1": f1_skoru},
                       egitim_skoru=True)
print(sonuc.ozet())
```

`ozet()` her metrik için `ortalama`, `std`, `min`, `max`, kat kat skorlar ve
(`egitim_skoru=True` ise) `egitim_ortalama` ile `asiri_ogrenme_farki` döndürür.

---

## 2. Üçlü ayrım: eğitim / doğrulama / test

Ders notundaki şemanın kuralı basit ama ihlali yaygındır:

| Set | Ne için | Kaç kez kullanılır |
|---|---|---|
| **Eğitim** | Model parametrelerini öğrenmek | Sınırsız |
| **Doğrulama** | Hiperparametre, eşik, model seçimi | Çok kez |
| **Test** | Nihai performans raporu | **Bir kez, en sonda** |

```python
from gozetimli.dogrulama.bolme import egitim_test_bol

X_gecici, X_test, y_gecici, y_test = egitim_test_bol(X, y, test_orani=0.2,
                                                      tohum=42, tabaka=y)
X_egitim, X_dog, y_egitim, y_dog = egitim_test_bol(X_gecici, y_gecici,
                                                    test_orani=0.25, tohum=42,
                                                    tabaka=y_gecici)
# 60 / 20 / 20
```

**Test setine her bakış onu biraz daha doğrulama setine dönüştürür.** "Test
skoru düşük çıktı, modeli değiştireyim" dediğiniz anda test seti artık
doğrulama setidir ve raporladığınız sayı iyimserdir.

---

## 3. Beş temel strateji (ders notundaki sırayla)

### K-Fold

Veri k eşit parçaya bölünür, her turda biri test olur. Her örnek tam olarak bir
kez test edilir.

```python
from gozetimli.dogrulama.bolme import KKat
KKat(5, karistir=True, tohum=42)
```

- **Artısı:** dengeli, hesaplama maliyeti makul (k eğitim).
- **k seçimi:** 5 ve 10 gelenekseldir. k büyüdükçe eğitim setleri büyür (yanlılık
  azalır) ama katlar birbirine benzer (varyans tahmini bozulur) ve maliyet artar.
- **`karistir` neden önemli:** veri sınıfa göre sıralı geldiyse (önce tüm 0'lar,
  sonra tüm 1'ler) karıştırmamak katları tek sınıflı yapar. Bu sessiz ve
  ölümcül bir hatadır.
- **Zaman serisinde kullanmayın.**

### Stratified K-Fold: sınıflandırmanın varsayılanı

Her katta sınıf oranları korunur. Ders notu: *"Katmanlardaki etiket oranlarını
(ör. %60 1 ve %40 0) her katmanda korumaya çalışır."*

```python
from gozetimli.dogrulama.bolme import TabakaliKKat
TabakaliKKat(5, karistir=True, tohum=42)
```

Neden varsayılan? %4 pozitifli 50 örneklik bir veride düz K-Fold, hiç pozitif
içermeyen test katları üretir; o katta recall tanımsızdır ve ortalama anlamını
yitirir. Test dosyasındaki `test_duz_kfold_azinlik_sinifini_kaybedebilir` bunu
ölçerek gösterir.

**Sınıflandırmada `KKat` yerine neredeyse her zaman `TabakaliKKat` kullanın.**

### Leave-One-Out (LOOCV)

Her turda tek bir gözlem test edilir, n tur yapılır.

```python
from gozetimli.dogrulama.bolme import BirDisarida
BirDisarida()
```

Ders notunun dezavantaj listesi doğru, ama bir tanesi eklenmeli:

- Büyük veride çok yavaş (n kez eğitim).
- **Kat başına skor ikili olur**: tek örnekte accuracy ya 0 ya 1'dir. Ortalama
  anlamlıdır ama standart sapma yorumlanamaz.
- Eğitim setleri neredeyse birbirinin aynısıdır, bu yüzden skorlar yüksek
  korelasyonludur, varyans tahmini güvenilmez.

n < ~100 olan küçük veri setleri dışında K-Fold tercih edilir.

### Leave-P-Out

Her turda p gözlem test edilir, **tüm** kombinasyonlar denenir.

```python
from gozetimli.dogrulama.bolme import PDisarida
PDisarida(2)
```

Kombinatorik patlama gerçektir:

| n | p | Tur sayısı |
|---|---|---|
| 20 | 2 | 190 |
| 20 | 5 | 15 504 |
| 50 | 5 | 2 118 760 |

Bu implementasyon `guvenlik_siniri` (varsayılan 100 000) ile erken hata verir;
bilinçli olarak yükseltilebilir. Pratikte nadiren gerekir.

### Time Series CV: sıra bozulamaz

Ders notu: *"Geçmiş veriler eğitimde, ileri tarihli veriler doğrulamada
kullanılır. Veri sırası mutlaka korunmalıdır."*

```python
from gozetimli.dogrulama.bolme import ZamanSerisiBolme
ZamanSerisiBolme(5)                                    # genişleyen pencere
ZamanSerisiBolme(5, pencere="kayan")                   # sabit pencere, eski veri düşer
ZamanSerisiBolme(5, bosluk=7)                          # 7 adımlık tampon
```

- **`pencere="genisleyen"`**: eğitim seti her turda büyür. Varsayılan.
- **`pencere="kayan"`**: eğitim uzunluğu sabit kalır, eski veri düşer. Rejim
  değişimi olan serilerde daha gerçekçi.
- **`bosluk`**: eğitim sonu ile test başı arasında atlanan örnek sayısı. 7 gün
  sonrasını tahmin ediyorsanız `bosluk=6` koyun, yoksa eğitim penceresi test
  hedefine sızar.

**Not:** Bu bölücü ilk bloğu asla test etmez (ilk katın eğitim verisi olması
gerekir). Bu yüzden out-of-fold tahmin tüm veriyi kapsamaz ve `capraz_tahmin`
bu bölücüyle açık bir hata verir.

---

## 4. Ders notunda olmayan ama gereken iki strateji

### GrupKKat: sızıntının en sık kaynağı

Aynı gruba ait satırlar asla eğitim ve testte birlikte bulunmaz.

```python
from gozetimli.dogrulama.bolme import GrupKKat
GrupKKat(5)
# capraz_dogrula(..., bolucu=GrupKKat(5), gruplar=hasta_kimlikleri)
```

Ne zaman gerekir: aynı hastanın 5 ölçümü, aynı mağazanın 300 günü, aynı
kullanıcının 40 tıklaması, aynı belgeden çıkarılmış 20 cümle.

Satır bazlı bölerseniz model "bu hastayı zaten gördüm" der. Test dosyasındaki
`test_grup_sizintisi_skoru_sisirir` bu farkı ölçüyor: aynı veride satır bazlı
CV %98+, grup bazlı CV %80'in altında sonuç veriyor. Aradaki 20 puan tamamen
sahtedir.

### TekrarliKKat: bölme şansını ortalamak

```python
from gozetimli.dogrulama.bolme import TekrarliKKat
TekrarliKKat(5, tekrar=3, tohum=42, tabakali=True)   # 15 bölme
```

Küçük veride tek bir 5-kat bölmenin skoru, o rastgele bölmenin şansını içerir.
5x2 veya 10x5 tekrarlı CV, ortalamanın etrafındaki belirsizliği görmenin en
ucuz yoludur.

---

## 5. Hangi bölücü? Karar akışı

```
Veri zamana bağlı mı (sıra anlamlı mı)?
├── EVET → ZamanSerisiBolme
│           tahmin ufku h > 1 ise bosluk=h-1
└── HAYIR
    ├── Satırlar gruplanıyor mu (aynı kişi/mağaza/belge)?
    │   └── EVET → GrupKKat  (gruplar= parametresi zorunlu)
    └── HAYIR
        ├── Sınıflandırma mı?
        │   ├── EVET → TabakaliKKat(5 veya 10, karistir=True, tohum=...)
        │   └── HAYIR → KKat(5 veya 10, karistir=True, tohum=...)
        └── Veri çok mu küçük (n < 100)?
            └── EVET → BirDisarida veya TekrarliKKat
```

---

## 6. İç içe (nested) çapraz doğrulama

**Problem:** Hiperparametreyi CV ile seçip sonra aynı CV skorunu "modelin
performansı" diye raporlamak, seçim sürecinin bilgisini skora sızdırır. Çok
aday denendiğinde iyimser sapma birkaç puana çıkar.

**Çözüm:** İki döngü.

```python
from gozetimli.dogrulama.capraz import ic_ice_capraz_dogrula

dis_skorlar, secilen = ic_ice_capraz_dogrula(
    lambda **p: DecisionTreeClassifier(random_state=0, **p), X, y,
    dis_bolucu=TabakaliKKat(5),      # skor buradan
    ic_bolucu=TabakaliKKat(4),       # seçim buradan
    izgara=[{"max_depth": d} for d in (1, 3, 5, None)],
    metrik=dogruluk)
```

- **Dış döngü:** veriyi eğitim/test diye böler: **raporlanacak skor budur.**
- **İç döngü:** yalnızca dış eğitim parçasında en iyi hiperparametreyi seçer.

Test dosyasındaki `test_nested_cv_iyimser_sapmayi_onler`, sinyalsiz veride
(rastgele etiket) ızgara aramasının şişirdiği skoru ve nested CV'nin ~0.5'te
kaldığını gösteriyor.

**Ek fayda:** `secilen` listesi kat kat seçilen parametreleri döndürür.
Katlar arasında çok farklı parametreler seçiliyorsa, modeliniz o
hiperparametreye karşı kararsızdır, bu, tek başına değerli bir bulgudur.

---

## 7. Out-of-fold tahmin

Her örnek için, o örneği **görmemiş** bir modelden gelen tahmin:

```python
from gozetimli.dogrulama.capraz import capraz_tahmin
oof = capraz_tahmin(model, X, y, bolucu=TabakaliKKat(5))
olasiliklar = capraz_tahmin(model, X, y, bolucu=TabakaliKKat(5), olasilik=True)
```

Kullanım alanları:
- Dürüst bir karmaşıklık matrisi / ROC eğrisi çizmek
- Eşik seçimi için tüm veriyi kullanmak (test setini harcamadan)
- Stacking için ikinci seviye modelin girdisini üretmek

**Uyarı:** dönen tahminler tek bir modele ait değildir; k farklı modelin
birleşimidir. Bu yüzden OOF skoru, kat ortalamasından biraz farklı çıkabilir.

---

## 8. Öğrenme eğrisi: bias mı varyans mı?

```python
from gozetimli.dogrulama.capraz import ogrenme_egrisi
boyutlar, egitim, dogrulama = ogrenme_egrisi(
    model, X, y, bolucu=TabakaliKKat(5), metrik=dogruluk,
    oranlar=(0.1, 0.25, 0.5, 0.75, 1.0))
```

Teşhis tablosu:

| Desen | Teşhis | Ne yapmalı |
|---|---|---|
| İki eğri de düşük, birbirine yakın | **Yüksek bias** (eksik öğrenme) | Daha karmaşık model, daha iyi özellik. **Veri eklemek işe yaramaz.** |
| Eğitim yüksek, doğrulama düşük, boşluk kalıcı | **Yüksek varyans** (aşırı öğrenme) | Düzenlileştirme (Ridge/Lasso), budama, daha çok veri |
| Boşluk veri arttıkça kapanıyor | İyi yolda | Veri eklemek gerçekten işe yarar |

Ders notundaki Ridge/Lasso bölümünün "düşük bias, yüksek varyans" tarifi tam
olarak ikinci satırdır.

---

## 9. En kritik kural: her şey döngünün İÇİNDE

Ölçekleme, eksik değer doldurma, özellik seçimi, kodlama, **öğrenen her adım**
çapraz doğrulama döngüsünün içinde olmalıdır.

```python
# YANLIŞ: test katının bilgisi eğitime sızar
X_olcekli = StandardScaler().fit_transform(X)
capraz_dogrula(model, X_olcekli, y, ...)

# DOĞRU: her katta yeniden öğrenilir
from sklearn.pipeline import Pipeline
boru = Pipeline([("olcek", StandardScaler()), ("model", model)])
capraz_dogrula(boru, X, y, ...)
```

Bu konunun tamamı ve ölçülmüş örnekleri: [veri-sizintisi.md](veri-sizintisi.md).
