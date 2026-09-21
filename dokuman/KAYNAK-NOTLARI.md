# Kaynak notları: ders materyalinde tespit edilen tutarsızlıklar

Bu deponun içeriği, **Atıl Samancıoğlu**'nun makine öğrenmesi kursundaki
gözetimli öğrenme bölümü ders notları (5-19 numaralı PDF'ler) çalışılarak
yazıldı. Ders notlarındaki her sayısal el hesabı kodla yeniden üretildi.

**Çoğu tuttu.** Entropi, Gini, bilgi kazancı, KNN mesafeleri, Naive Bayes
olasılıkları, AdaBoost ağırlık zinciri, gradient boosting artıkları, XGBoost
yaprak değerleri ve XGBoost regresyon hesaplarının tamamı doğrulandı.

**Beşi tutmadı ya da eksikti.** Aşağıda hepsi, doğru değerleriyle ve doğrulayan
testin adıyla listelenmiştir. Amaç kusur bulmak değil, bu depoyu kullanan
birinin yanlış bir sayıyı doğru sanmasını engellemek.

Tüm bulgular şu komutla yeniden üretilebilir:

```bash
pytest testler/test_ders_ornekleri.py -v -m ders
```

---

## 1. Karar ağacı: kök varyans yanlış hesaplanmış

**Kaynak:** `13-Decision_Tree_Algorithms.pdf` § 15.3
**Notta:** kök varyans = 124.83
**Doğrusu:** **113.2653**

Burs verisi: GPA `[2.5, 2.7, 3.0, 3.2, 3.5, 3.7, 4.0]`,
hedef `[10, 12, 18, 22, 30, 35, 40]` (bin TL).

```
ortalama          = 167/7 = 23.857
kareler toplamı   = 792.857
792.857 / 7       = 113.2653   ← doğru (ddof=0, ağaçların kullandığı)
792.857 / 6       = 132.1429   (ddof=1, örneklem varyansı)
```

124.83 hiçbir bölmeyle elde edilemiyor. Notun **alt küme** varyansları
(11.55 ve 44.19) doğrudur; yalnızca kök yanlıştır.

**Doğrulayan test:** `TestKararAgaciRegresyon::test_burs_kok_varyansi_notta_hatali`

---

## 2. Karar ağacı: en iyi eşik yanlış seçilmiş

**Kaynak:** `13-Decision_Tree_Algorithms.pdf` § 15.5, Tablo 3
**Notta:** GPA < 3.1 en yüksek varyans azaltımını verir (94.64)
**Doğrusu:** en iyi eşik **3.35**'tir

Tüm eşikler için varyans azaltımı (kök varyans 113.2653 ile):

| Eşik | Sol/Sağ | Ağırlıklı varyans | Azaltım |
|---|---|---|---|
| 2.60 | 1 / 6 | 81.2619 | 32.0034 |
| 2.85 | 2 / 5 | 47.1429 | 66.1224 |
| 3.10 | 3 / 4 | 30.2024 | 83.0629 |
| **3.35** | **4 / 3** | **20.1429** | **93.1224** ← en iyi |
| 3.60 | 5 / 2 | 38.8143 | 74.4510 |
| 3.85 | 6 / 1 | 69.8333 | 43.4320 |

Notun 94.64 rakamı hatalı kök varyanstan türemiş: `124.83 - 30.19 = 94.64`.
Ağırlıklı varyans 30.19 doğrudur.

Kök varyans tüm eşiklere sabit bir kaydırma uygular, **sıralamayı değiştirmez**:
yani 3.1'in seçilmesi ayrı bir hatadır. Notun Tablo 3'ündeki diğer özet sayılar
(3.35 → 80.3 gibi) de yeniden üretilemiyor.

**Doğrulayan test:** `TestKararAgaciRegresyon::test_burs_tum_esikler`

---

## 3. XGBoost: sağ düğüm similarity'si yanlış

**Kaynak:** `18-XGBoost_Algorithm.pdf` § 5 (Adım 3)
**Notta:** S_sağ = 0.702, Gain = 1.511
**Doğrusu:** S_sağ = **0.7191**, Gain = **1.5292**

Notun **kendi yuvarlanmış ara değerleriyle** hesaplandığında bile sonuç farklı:

```
ΣR = 1.144,  Σp(1-p) = 0.816,  λ = 1
1.144² / (0.816 + 1) = 1.30874 / 1.816 = 0.7207
```

Yuvarlanmamış değerlerle 0.7191. 0.702 ile arada 0.017 fark var; bu bir
aritmetik hatasıdır (muhtemelen 0.7206 → 0.702 basamak kayması).

Sol düğüm (0.809) ve her iki yaprak değeri (-0.709, +0.631) **doğrudur.**
Hata bölme kararını değiştirmiyor, ama ara değeri kontrol etmeye çalışan biri
tutturamaz.

**Doğrulayan test:** `TestXGBoostSiniflandirma::test_xgb_sag_benzerlik_notta_hatali`

---

## 4. XGBoost: S_kök = 0 sonucunun gerekçesi yanlış

**Kaynak:** `18-XGBoost_Algorithm.pdf` § 5
**Notta:** "S_root = 0 (çünkü kök düğümde tek bir grup var)"
**Doğrusu:** sonuç doğru, gerekçe yanlış

Tek grup olması similarity'yi sıfırlamaz, formül `(Σg)²/(Σh+λ)`, tek gruptaki
gradyan toplamı sıfırdan farklı olabilir.

Doğru sebep: **artıkların toplamı tam olarak sıfırdır.**

```
2·(-0.7143) + 5·(0.2857) = -1.4286 + 1.4286 = 0
```

Ve bu bir tesadüf değil: F₀ log-odds olarak seçildiğinde `Σ(y - p) = 0` olması,
F₀'ın log-loss'u minimize etmesinin **birinci derece koşuludur.** Yani her
gradient boosting turunun başında kök similarity'si sıfırdan başlar, bu,
anlaşılmaya değer bir özelliktir.

**Doğrulayan testler:**
`TestXGBoostSiniflandirma::test_kok_benzerlik_sifirdir_ama_gerekce_farkli`,
`TestGradientBoostingBaslangic::test_log_odds_baslangici_artiklari_sifirlar`

---

## 5. XGBoost: Gain formülü eksik (hata değil, sadeleştirme)

**Kaynak:** `18-XGBoost_Algorithm.pdf` § 5

Notta: `Gain = S_sol + S_sağ - S_kök`

XGBoost makalesindeki (Chen & Guestrin 2016, *XGBoost: A Scalable Tree Boosting
System*, Denklem 7) tam biçim:

```
Gain = ½ · [ G_sol²/(H_sol+λ) + G_sağ²/(H_sağ+λ) - (G_sol+G_sağ)²/(H+λ) ] - γ
```

½ ve γ notta yok. ½ ortak çarpan olduğu için **bölme sıralaması değişmez**:
öğretici amaçla makul bir sadeleştirme. Ama gerçek `xgboost` kütüphanesinin
bastığı gain değeriyle karşılaştırırsanız sayılar **2 kat** farklı çıkar ve
γ'sız formülde ön budama davranışı görünmez.

Bu depoda ikisi de mevcut:

```python
xgb_kazanc(s_sol, s_sag, s_kok)                      # ders notu biçimi
xgb_kazanc(s_sol, s_sag, s_kok, yarim_faktor=True)   # makale biçimi
xgb_kazanc(s_sol, s_sag, s_kok, gama=0.1)            # ön budama ile
```

**Doğrulayan test:** `TestXGBoostSiniflandirma::test_yarim_faktor_siralamayi_degistirmez`

---

## 6. AdaBoost: tablo tutarsızlığı

**Kaynak:** `16-Adaboost_Algorithm.pdf` Tablo 6

İkinci stump tablosunda ID 7'nin GPA'si `>3.0` yazılmış, oysa Tablo 1'de aynı
ID `<3.0`. ID 4'ün mülakat skoru da Tablo 1'de "Düşük", Tablo 6'da "Normal".

Tablo 6, ağırlıklı yeniden örnekleme sonrası çekilen satırları gösterdiği için
ID'lerin **tekrar etmesi** beklenir ve doğaldır. Fakat aynı ID'nin özellik
değerinin değişmesi bir dizgi hatasıdır. Bu depo Tablo 1'i esas alır.

Notun asıl hesapları (ε = 0.249, α₂ = 0.552) tutarlıdır ve doğrulanmıştır.

---

## 7. Regresyon metrikleri: kaynak eksik

Elimizdeki klasörde **1-4 numaralı PDF'ler yok** (dosyalar 5'ten başlıyor).
Mevcut notlarda MAE, RMSE, R², MAPE gibi regresyon metrikleri **tanımlanmıyor**;
MSE yalnızca doğrusal/Ridge/Lasso maliyet fonksiyonunun içinde, varyans ise
ağaç bölme ölçütü olarak geçiyor.

İçerik akışına bakılırsa eksik PDF'ler muhtemelen giriş, basit doğrusal
regresyon, gradient descent ve regresyon metriklerini kapsıyor.

`gozetimli/metrikler/regresyon.py` içindeki tanımlar standart istatistik/ML
literatüründen alınmış ve `testler/test_sklearn_uyumu.py` içinde scikit-learn'e
karşı doğrulanmıştır, ders notundan türetilmemiştir. Bu ayrım modülün
docstring'inde de yazılıdır.

---

## Doğrulanan hesaplar (tutanlar)

Aşağıdakilerin tamamı ders notundaki değerlerle **uyuştu**:

| Konu | Notta | Hesaplanan |
|---|---|---|
| Kök entropi (4 Onay / 3 Red) | 0.985 | 0.9852 |
| Kök Gini | 0.4898 | 0.4898 |
| Bilgi kazancı (Mülakat=Yüksek) | 0.521 | 0.5216 |
| Bilgi kazancı (GPA>3.0) | 0.021 | 0.0202 |
| Alt küme varyansları (burs) | 11.55 / 44.19 | 11.5556 / 44.1875 |
| Ağırlıklı varyans (3.1) | 30.19 | 30.2024 |
| KNN Öklid mesafeleri (6 örnek) | 2.24 … 5.66 | tamamı uyuyor |
| Naive Bayes prior / likelihood | 3/5, 1.0, 1/3 | tamamı uyuyor |
| Gaussian NB parametreleri | μ=27.5, σ=2.5 | uyuyor |
| Gaussian yoğunluk (yaş=27) | 0.156 | 0.1564 |
| AdaBoost α₁ | 0.895 | 0.8959 |
| AdaBoost ham ağırlıklar | 0.058 / 0.354 | 0.0584 / 0.3496 |
| AdaBoost normalize | 0.083 / 0.504 | 0.0834 / 0.4996 |
| AdaBoost α₂ | 0.552 | 0.5520 |
| AdaBoost F(x) | 1.447 | 1.4470 |
| GB F₀ (maaş) | 70 | 70.0 |
| GB sigmoid(±0.05) | 0.487 / 0.512 | 0.4875 / 0.5125 |
| XGBoost F₀ = ln(5/2) | 0.916 | 0.9163 |
| XGBoost p₀ | 0.714 | 0.7143 |
| XGBoost S_sol | 0.809 | 0.8101 |
| XGBoost yaprak değerleri | -0.709 / +0.631 | -0.7089 / +0.6292 |
| XGBoost F₁, p₁ | 0.845 / 0.699 | 0.8454 / 0.6996 |
| XGBoost regresyon Gain | 21 000 000 | 21 000 000 |
| XGBoost regresyon yapraklar | -1500 / +2000 | -1500 / +2000 |

---

## Telif notu

Ders notlarının kendisi (PDF'ler, şekiller, metin) bu depoya **dahil
edilmemiştir** ve eklenmemelidir; telif hakları kendi sahibine aittir.

Depodaki her satır metin ve kod özgün olarak yazılmıştır. Ders notlarındaki
öğretici mini veri setleri (7 satırlık kredi onay tablosu gibi) bir formülün
elle takip edilebilmesi için kaynak gösterilerek yeniden üretilmiştir; bunlar
telif korumasına konu olmayacak kadar küçük olgusal tablolardır ve
`src/gozetimli/veri/ders_verileri.py` içinde her biri kaynağıyla birlikte
işaretlenmiştir.
