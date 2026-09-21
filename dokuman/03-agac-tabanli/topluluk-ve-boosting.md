# Topluluk öğrenme: Bagging, Random Forest ve Boosting ailesi

Kaynak ders notları: `14-Ensemble_Algorithms.pdf`, `15-Random_Forest_Algorithms.pdf`,
`16-Adaboost_Algorithm.pdf`, `17-Gradient_Boosting_Algorithm.pdf`,
`18-XGBoost_Algorithm.pdf`, `19-LightGbm_Algorithm.pdf`

Kod: [`gozetimli/boosting.py`](../../src/gozetimli/boosting.py) ·
Test: [`testler/test_boosting.py`](../../testler/test_boosting.py),
[`testler/test_ders_ornekleri.py`](../../testler/test_ders_ornekleri.py)

> ⚠️ XGBoost ders notunda **bir aritmetik hatası** ve **bir yanlış gerekçe**
> tespit edildi. Ayrıntı: [KAYNAK-NOTLARI.md](../KAYNAK-NOTLARI.md).

---

## 1. İki temel strateji

| | **Bagging** | **Boosting** |
|---|---|---|
| Eğitim | **Paralel:** modeller bağımsız | **Ardışık:** her model öncekinin hatasına odaklanır |
| Amaç | **Varyansı** azaltmak | **Bias'ı** azaltmak |
| Temel öğrenici | Güçlü (derin ağaç) | Zayıf (sığ ağaç, stump) |
| Birleştirme | Oylama / ortalama | Ağırlıklı toplam |
| Aşırı öğrenme | Dirençli | Duyarlı (tur sayısı sınırlanmalı) |
| Örnek | Random Forest | AdaBoost, GBM, XGBoost, LightGBM |

**Sezgi:** Bagging, birbirinden bağımsız hata yapan uzmanların ortalamasını
alır, hatalar birbirini götürür. Boosting, bir uzmanın yanlışını bir sonraki
uzmana düzelttirir, her adımda kalan hata küçülür.

---

## 2. Bagging (Bootstrap Aggregating)

1. Veriden **yerine koyarak** (with replacement) n boyutlu alt örneklemler çek
2. Her örneklem için bir model eğit
3. Sınıflandırmada çoğunluk oyu, regresyonda ortalama

Bootstrap örnekleminde her örneğin seçilme olasılığı `1 - (1-1/n)ⁿ ≈ 0.632`'dir.
Yani her ağaç verinin ~%63'ünü görür; kalan **%37 "out-of-bag" (OOB)** örnek,
o ağaç için bedava bir doğrulama seti oluşturur.

**OOB skoru:** ayrı bir doğrulama seti ayırmadan genelleme hatası tahmini.
scikit-learn'de `oob_score=True`. Küçük veride değerlidir ama çapraz
doğrulamanın yerini tam olarak tutmaz (OOB, k-kat CV'den biraz kötümser olma
eğilimindedir).

---

## 3. Random Forest

Bagging + **özellik alt örneklemesi**.

Ders notundaki iki örnekleme:
- **Satır örneklemesi (row sampling):** bootstrap
- **Özellik örneklemesi (feature sampling):** her **bölmede** rastgele bir
  özellik alt kümesi denenir

İkincisi kritiktir. Sadece bagging yapılsaydı, çok güçlü bir özellik tüm
ağaçların köküne yerleşir ve ağaçlar birbirine benzerdi, ortalama almanın
faydası azalırdı. Özellik örneklemesi ağaçları **birbirinden bağımsızlaştırır.**

| Parametre | Tipik | Not |
|---|---|---|
| `n_estimators` | 100-500 | Artırmak zarar vermez, sadece yavaşlatır |
| `max_features` | `"sqrt"` (sınıflandırma), 1/3 (regresyon) | Çeşitliliğin ana düğmesi |
| `max_depth` | `None` | RF'de derin ağaç sorun değil |
| `min_samples_leaf` | 1-5 | Gürültülü veride artırın |

**Neden aşırı öğrenmiyor?** Ağaç sayısını artırmak varyansı azaltır, bias'ı
değiştirmez. Daha fazla ağaç eklemek test hatasını **kötüleştirmez:** bir
platoya oturur. (Boosting'de durum tam tersidir.)

**Artı:** kutudan çıktığı gibi güçlü, az ayar gerektirir, paralelleşir, OOB
skoru bedava, aykırı değerlere dayanıklı.
**Eksi:** tek ağacın yorumlanabilirliğini kaybeder, bellek tüketir, regresyonda
ekstrapolasyon yapamaz.

**Özellik önemleri uyarısı:** varsayılan `feature_importances_` (saflaşma
tabanlı) yüksek kardinaliteli özellikleri kayırır. `permutation_importance`
kullanın.

---

## 4. AdaBoost

Zayıf öğrenicileri (genelde **stump** = derinliği 1 olan ağaç) ardışık eğitir;
her turda **yanlış sınıflanan örneklerin ağırlığını artırır.**

### Adımlar

1. Tüm örneklere eşit ağırlık: wᵢ = 1/n
2. Zayıf öğrenici eğit, hata oranını hesapla: ε = (yanlışların ağırlık toplamı)
3. Öğrenicinin ağırlığı: **α = ½·ln((1-ε)/ε)**
4. Örnek ağırlıklarını güncelle: doğru → `w·e⁻ᵅ`, yanlış → `w·e⁺ᵅ`
5. Normalize et, 2. adıma dön
6. Nihai karar: `sign(Σ αₘ·hₘ(x))`

```python
from gozetimli.boosting import (adaboost_agirlik_katsayisi,
                                adaboost_agirliklari_guncelle,
                                adaboost_bin_araliklari, adaboost_nihai_skor)

adaboost_agirlik_katsayisi(1/7)          # 0.8959, ders notu örneği
```

### α'nın davranışı

| ε | α | Anlamı |
|---|---|---|
| 0.5 | **0** | Yazı tura kadar iyi: hiç söz hakkı yok |
| < 0.5 | > 0 | Hata küçüldükçe söz hakkı hızla büyür |
| > 0.5 | < 0 | Rastgeleden kötü; tahmin ters çevrilip kullanılır |
| 0 | +∞ | Kırpılır (sayısal taşma) |

### Ders notundaki "bin" mantığı

Normalize ağırlıklardan kümülatif aralıklar kurulur; sonraki turda örnekler
0-1 arası rastgele sayı çekilerek seçilir. Ağırlığı büyük örnek geniş aralık
kaplar, dolayısıyla birden çok kez seçilir.

Bu, ağırlıklı yeniden örnekleme (weighted resampling) uygulamasıdır. Alternatif
uygulama, ağırlıkları doğrudan öğreniciye geçirmektir (`sample_weight`);
scikit-learn bunu yapar. İkisi de aynı matematiği farklı şekilde gerçekler.

Ders notundaki 7 satırlık örneğin **tüm ara değerleri** test dosyasında
doğrulanmıştır: α₁ = 0.895, ham ağırlıklar 0.058/0.354, normalize 0.083/0.504,
α₂ = 0.552, F(x) = 1.447.

**Zayıflığı:** üstel kayıp kullandığı için gürültülü etiketlere ve aykırı
değerlere **çok duyarlıdır:** yanlış etiketli bir örneğin ağırlığı üstel
büyür ve model ona takılır. Bu yüzden pratikte gradient boosting tercih edilir.

---

## 5. Gradient Boosting

AdaBoost örnek ağırlıklarını değiştirirken, gradient boosting doğrudan
**artıkları (residual)** hedefler.

### Regresyon

1. `F₀ = ortalama(y)`
2. Artıkları hesapla: `rᵢ = yᵢ - F(xᵢ)`
3. Artıkları tahmin eden bir ağaç kur: `h(x)`
4. Güncelle: `F ← F + η·h(x)`
5. 2'ye dön

**Öğrenme oranı η** neden gerekli? Ders notunun örneği bunu iyi gösteriyor:
η = 1 ile model tek adımda 70 → 50/90'a sıçrar, η = 0.1 ile 68/72'ye. Küçük
adımlar, her ağacın yalnızca bir miktar düzeltme yapmasını sağlar; bu da aşırı
öğrenmeyi geciktirir ve daha iyi genelleme verir.

**η ile ağaç sayısı ters orantılıdır:** η'yı yarıya indirirseniz yaklaşık iki
kat ağaç gerekir. Tipik: η = 0.05-0.1, `n_estimators` = 100-1000, erken
durdurma ile.

### Sınıflandırma

Regresyondan farkı: tahminler **log-odds** uzayında tutulur.

1. `F₀ = log(p̄ / (1-p̄))`
2. `p = σ(F)` (sigmoid)
3. Artık = `y - p` (log-loss'un negatif gradyanı)
4. Artıkları tahmin eden ağaç, `F ← F + η·h(x)`

**Neden log-odds?** İki sebep: (1) toplama log-odds uzayında yapılır, bu
[0,1] sınırından taşmayı imkânsız kılar; (2) log-loss'un gradyanı bu uzayda
sade olur (`p - y`).

```python
from gozetimli.boosting import gb_baslangic_degeri, sigmoid
gb_baslangic_degeri([0, 0, 1, 1, 1], gorev="siniflandirma")   # log(3/2) = 0.405
```

**Önemli özellik:** F₀ log-odds seçildiğinde `Σ(y - p) = 0` olur. Bu bir
tesadüf değil, log-loss'un birinci derece optimallik koşuludur, ve XGBoost'ta
kök similarity'sinin neden sıfır olduğunu açıklar.

---

## 6. XGBoost

Gradient boosting'in düzenlileştirilmiş ve optimize edilmiş hali. Ders
notundaki farklar listesi doğru; en önemlisi **ikinci türev (Hessian)
kullanımıdır.**

### Gradyan ve Hessian

Log-loss için:
```
g = p - y          (1. türev: hatanın yönü ve büyüklüğü)
h = p(1-p)         (2. türev: hatanın değişim hızı)
```

Güncelleme `-g/(h+λ)` biçimindedir. Newton yöntemine benzer: eğrilik bilgisi
adım boyunu ayarlar. Model emin olduğu yerlerde (p → 0 veya 1) h küçülür,
adım kısalır.

### Similarity ve Gain

```
Similarity S = (Σg)² / (Σh + λ)
Gain = S_sol + S_sağ - S_kök   (- γ)
```

`λ` (L2 düzenlileştirme) paydayı büyütür → skorları ve yaprak değerlerini
bastırır. `λ = 0` klasik gradient boosting'e döner.

`γ` bir bölmenin yapılabilmesi için gereken **en az kazançtır:** ön budama
mekanizması.

```python
from gozetimli.boosting import xgb_benzerlik, xgb_kazanc, xgb_yaprak_degeri, xgb_cover

S_sol = xgb_benzerlik(artiklar=artiklar_sol, olasiliklar=p0, lam=1)
kazanc = xgb_kazanc(S_sol, S_sag, S_kok, gama=0.1)
V = xgb_yaprak_degeri(artiklar_sol, p0, lam=1)      # Σg / (Σh + λ)
```

**Similarity ile yaprak değeri arasındaki fark:** similarity'nin payında kare
vardır (bölmenin kalitesini ölçer, işaret önemsiz), yaprak değerinde yoktur
(düzeltmenin yönünü verir, işaret önemli).

**Cover** = Σ p(1-p) = hessian toplamı. Bir düğümün "bilgi ağırlığı".
`min_child_weight` bunun alt sınırıdır: cover küçükse ya çok az örnek vardır
ya da model zaten emindir, iki durumda da bölmeye devam etmek gürültü
öğrenmektir.

> **Ders notundaki hata.** Sağ düğüm similarity'si 0.702 verilmiş; notun kendi
> ara değerleriyle bile 1.144²/1.816 = 0.7207 çıkıyor (yuvarlanmamış: 0.7191).
> Buna bağlı olarak Gain de 1.511 değil 1.5292. Sol düğüm ve yaprak değerleri
> doğru; hata bölme kararını değiştirmiyor. Ayrıca S_kök = 0 sonucu "kök
> düğümde tek grup var" diye gerekçelendirilmiş, doğru sebep artıkların
> toplamının sıfır olmasıdır. [KAYNAK-NOTLARI.md](../KAYNAK-NOTLARI.md)

> **Formül farkı.** Ders notu `Gain = S_sol + S_sağ - S_kök` veriyor. XGBoost
> makalesindeki (Chen & Guestrin 2016, Denklem 7) tam biçim `½·[...] - γ`'dır.
> ½ ortak çarpan olduğu için **bölme sıralaması değişmez**, ama gerçek
> kütüphanenin bastığı gain ile karşılaştırırsanız sayılar 2 kat farklı çıkar.
> `xgb_kazanc(yarim_faktor=True)` tam biçimi verir.

### Ana parametreler

| Parametre | Rolü |
|---|---|
| `learning_rate` (η) | Adım boyu. 0.01-0.3 |
| `n_estimators` | Tur sayısı: erken durdurma ile birlikte |
| `max_depth` | 3-8. Boosting'de derin ağaç gerekmez |
| `reg_lambda` (λ) | L2: yaprak değerlerini bastırır |
| `reg_alpha` (α) | L1: yaprakları sıfırlar |
| `gamma` (γ) | En az bölme kazancı |
| `min_child_weight` | En az cover |
| `subsample` | Satır örneklemesi (0.8 tipik) |
| `colsample_bytree` | Özellik örneklemesi |

---

## 7. LightGBM

Microsoft'un uygulaması. Üç yapısal farkı var:

### Leaf-wise vs level-wise büyüme

| | XGBoost (varsayılan) | LightGBM |
|---|---|---|
| Büyüme | **Level-wise:** her seviye dengeli | **Leaf-wise:** en çok kazanç veren yaprak |
| Ağaç şekli | Simetrik | Asimetrik, derin dallar |
| Aynı yaprak sayısında hata | Daha yüksek | **Daha düşük** |
| Aşırı öğrenme riski | Daha düşük | **Daha yüksek** |

Leaf-wise, aynı sayıda yaprakla daha çok kayıp düşürür ama küçük veride
derin ve dar dallar üretip ezberleyebilir. `num_leaves` ve `min_data_in_leaf`
ile sınırlanmalıdır. Kural: `num_leaves < 2^max_depth`.

### Histogram/binning

Sürekli değişkenler önce **bin**'lere ayrılır (varsayılan 255). Tüm olası
eşikler yerine bin sınırları denenir. Sonuç: bellek ve hız kazancı, ihmal
edilebilir doğruluk kaybı. (XGBoost'ta bu `tree_method="hist"` ile açılır ve
modern sürümlerde varsayılandır, yani bu fark eskisi kadar belirgin değil.)

### Kategorik destek

LightGBM kategorik özellikleri **doğal olarak** işler: `CreditScore ∈ {Düşük,
Orta}` gibi alt küme bölmeleri yapabilir. XGBoost'ta kategoriyi sayıya
çevirmek (LabelEncoder) yapay bir sıralama uydurur ve anlamlı bölmeleri
zorlaştırır.

**Uyarı:** yüksek kardinaliteli kategorilerde LightGBM'in kategorik desteği
aşırı öğrenmeye açıktır; `cat_smooth` ve `min_data_per_group` ayarlanmalıdır.

---

## 8. Hangi topluluğu seçmeli?

```
Hızlı ve güvenilir bir baz çizgisi mi istiyorsun?
└── Random Forest, az ayar, iyi sonuç, aşırı öğrenmeye dirençli

Tablosal veride en yüksek doğruluk mu?
├── Veri < ~100k satır  → XGBoost
└── Veri > ~100k satır veya çok kategorik özellik → LightGBM

Çok sayıda kategorik özellik ve az ayar zamanı?
└── CatBoost (ders notlarında yok, ama bu alanda güçlü)

Yorumlanabilirlik zorunlu mu?
└── Tek karar ağacı veya lojistik regresyon, topluluk bunu veremez
```

**Gerçekçi bir not:** Kaggle'da ve pratikte tablosal veride gradient boosting
aileleri hâlâ derin öğrenmeyi geçer. Ama iyi ayarlanmış bir Random Forest ile
iyi ayarlanmış bir XGBoost arasındaki fark genellikle birkaç puandır:
veri kalitesi ve özellik mühendisliği, model seçiminden daha çok fark yaratır.

---

## 9. Boosting'de erken durdurma

Random Forest'ta ağaç eklemek zarar vermez; **boosting'de verir.** Belirli bir
tur sayısından sonra model gürültü öğrenmeye başlar.

```python
# Doğrulama setiyle erken durdurma (ör. xgboost)
model.fit(X_egitim, y_egitim,
          eval_set=[(X_dogrulama, y_dogrulama)],
          early_stopping_rounds=50)
```

Erken durdurma için kullanılan set, nihai skorun hesaplandığı test seti
**olmamalıdır:** yoksa sızıntı olur.
Bkz. [veri-sizintisi.md](../04-degerlendirme/veri-sizintisi.md).

---

## İlgili

- [Karar ağaçları](karar-agaci.md)
- [Çapraz doğrulama](../04-degerlendirme/capraz-dogrulama.md)
- [Kaynak notları ve tespit edilen hatalar](../KAYNAK-NOTLARI.md)
