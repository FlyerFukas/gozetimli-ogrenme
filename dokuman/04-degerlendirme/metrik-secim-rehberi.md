# Metrik seçim rehberi — hangi kıstas, ne zaman?

Bu dosya deponun çekirdeği. Bir modelin "iyi" olup olmadığı sorusunun tek bir
cevabı yoktur; cevap, **hangi hatanın size neye mal olduğuna** bağlıdır. Metrik
seçimi teknik değil, iş kararıdır.

> Ders notunun (9-Logistic_Regression_Performance_Metrics) kapanış cümlesi bunu
> doğru söylüyor: *"Hangi metriğin kullanılacağı, probleme ve risklere göre
> değişir."* Bu dosya o cümleyi karar tablosuna çevirir.

---

## 1. Önce şu üç soruyu yanıtla

**Soru 1 — Hedef sayısal mı, kategorik mi?**
Sayısal → [regresyon metrikleri](regresyon-metrikleri.md).
Kategorik → devam et.

**Soru 2 — İki hata türünden hangisi daha pahalı?**

| | Model "pozitif" dedi | Model "negatif" dedi |
|---|---|---|
| **Gerçekte pozitif** | TP — istediğimiz | **FN** — kaçırılan vaka |
| **Gerçekte negatif** | **FP** — yanlış alarm | TN — istediğimiz |

- FN pahalıysa → **recall**'u yükselt (kanser taraması: hastayı sağlıklı ilan etme)
- FP pahalıysa → **precision**'ı yükselt (spam filtresi: gerçek postayı çöpe atma)
- İkisi de benzer → **F1** ya da **MCC**

**Soru 3 — Sınıflar dengeli mi?**
%40–60 civarı → accuracy kullanılabilir.
Daha dengesiz → accuracy yanıltır, aşağıdaki tabloya bak.

---

## 2. Sınıflandırma karar tablosu

| Durum | Kullan | Kullanma | Neden |
|---|---|---|---|
| Sınıflar dengeli, hatalar eşit maliyetli | `dogruluk` | — | En sezgisel |
| Dengesiz (%10'dan az azınlık) | `f1_skoru`, `matthews_korelasyonu`, `ortalama_kesinlik` | `dogruluk` | Çoğunluğa "hayır" diyen model %90 accuracy alır |
| FN pahalı (teşhis, arıza, dolandırıcılık) | `f_beta_skoru(beta=2)`, `duyarlilik` | `kesinlik` tek başına | Kaçırılan vaka geri gelmez |
| FP pahalı (spam, otomatik blok, uyarı yorgunluğu) | `f_beta_skoru(beta=0.5)`, `kesinlik` | `duyarlilik` tek başına | Yanlış alarm güveni yıpratır |
| Eşik henüz seçilmedi, sıralama kalitesi ölçülüyor | `roc_auc` | Sert tahmin metrikleri | Tüm eşikleri birden değerlendirir |
| Çok dengesiz + sıralama kalitesi | `ortalama_kesinlik` (PR-AUC) | `roc_auc` | ROC dengesizlikte iyimserdir |
| Olasılık tahmini kullanılacak (fiyatlama, risk skoru) | `log_kaybi` | `dogruluk` | Kalibrasyonu ölçer |
| Çok sınıflı, tüm sınıflar eşit önemli | `ortalama="makro"` | `ortalama="agirlikli"` | Azınlık sınıfı gizlenmesin |
| Çok sınıflı, sınıf büyüklüğü önemli | `ortalama="agirlikli"` | — | Gerçek dağılımı yansıtır |
| Karşılaştırmalı rapor, tek sayı isteniyor | `matthews_korelasyonu` | `dogruluk` | Dört hücreyi de kullanır |

### ROC-AUC mı, PR-AUC mı?

Bu ayrım pratikte en çok karıştırılan yerdir:

- **ROC-AUC** sınıf dağılımından etkilenmez. Bu kararlılık bir avantaj gibi
  durur ama çok dengesiz veride **tuzaktır**: negatif sınıf devasa olduğu için
  FPR paydası çok büyüktür, yüzlerce yanlış alarm FPR'yi neredeyse hiç
  oynatmaz. Model gözle görülür şekilde kötüyken AUC 0.95 çıkabilir.
- **PR-AUC (`ortalama_kesinlik`)** taban çizgisi pozitif sınıf oranına eşittir.
  %1 pozitifli veride rastgele model 0.01 alır — 0.5 değil. Bu yüzden
  iyileşmeyi görmek çok daha kolaydır.

Kural: **pozitif sınıf %10'un altındaysa PR-AUC'yi birincil metrik yap**,
ROC-AUC'yi ikincil olarak raporla.

```python
from gozetimli.metrikler.siniflandirma import roc_auc, ortalama_kesinlik
# %5 pozitifli veride rastgele skor:
#   roc_auc          ≈ 0.50   (beklendiği gibi)
#   ortalama_kesinlik ≈ 0.05  (taban çizgisi = pozitif oranı)
```

### F-beta'da beta nasıl seçilir?

`beta`, **recall'un precision'a göre kaç kat önemli olduğudur.**

| beta | Anlamı | Tipik alan |
|---|---|---|
| 0.5 | Precision 2 kat önemli | Spam, otomatik içerik kaldırma |
| 1.0 | Eşit (F1) | Varsayılan, genel amaçlı |
| 2.0 | Recall 2 kat önemli | Hastalık taraması, güvenlik açığı tespiti |

Somut yol: bir FN'nin maliyetini bir FP'nin maliyetine böl, karekökünü al.
FN 25 kat pahalıysa beta = 5.

---

## 3. Regresyon karar tablosu

| Durum | Kullan | Neden |
|---|---|---|
| Genel amaçlı, raporlanacak | `rmse` | Hedefle aynı birimde |
| Aykırı değer var, dayanıklılık isteniyor | `mae`, `medyan_mutlak_hata` | Kareye almaz |
| Büyük hatalar orantısız zararlı | `mse` / `rmse` | Kare cezası |
| Hedef geniş ölçek aralığında (10 ile 10 000 arası) | `msle` / `rmsle` | Oransal hata cezalandırır |
| İş birimine yüzde olarak anlatılacak | `mape` (dikkatli), `smape` | Yorumlaması kolay |
| Modelin açıklama gücü soruluyor | `r2` | Ortalamaya göre kıyas |
| Farklı özellik sayılı modeller karşılaştırılacak | `duzeltilmis_r2` | Özellik eklemeyi cezalandırır |
| En kötü durum garantisi gerekiyor | `maksimum_hata` | Tek en kötü örnek |

### MAE mi MSE mi? Bu aslında bir merkezi eğilim seçimidir

Sabit bir sayı tahmin etmek zorunda olsaydınız:
- **MSE**'yi minimize eden sayı **ortalamadır**
- **MAE**'yi minimize eden sayı **medyandır**

Bu yüzden MSE ile eğitilen model aykırı değerlere doğru çekilir, MAE ile
eğitilen model çekilmez. Karar ağacı regresörünün yaprakta ortalama döndürmesi
tesadüf değil: varyans azaltımı (= MSE) ölçütü kullanıyor.

### MAPE'nin üç tuzağı

1. **y = 0'da tanımsız.** Talep tahmininde sıfır satışlı günler varsa çöker.
2. **Asimetrik.** Düşük tahminin cezası en fazla %100'dür (tahmin 0 olsa bile),
   yüksek tahminin cezası sınırsızdır. Model sistematik düşük tahmin etmeye
   itilir.
3. **Küçük gerçek değerlerde patlar.** y = 1 iken ŷ = 2 vermek %100 hatadır,
   oysa mutlak hata yalnızca 1 birimdir.

Sıfıra yakın değerler varsa `smape` (sınırlı, [0, 2]) veya doğrudan `mae` kullan.

---

## 4. Tek metrik yetmez — minimum rapor seti

Tek sayı her zaman bir şey gizler. Gerçek bir değerlendirme raporu şunları
içermeli:

**Sınıflandırma:**
1. Karmaşıklık matrisi (ham sayılar — yüzde değil)
2. Sınıf başına precision / recall / F1 / destek
3. Eşikten bağımsız bir metrik (ROC-AUC veya PR-AUC)
4. Çapraz doğrulama katlarının **ortalaması ve standart sapması**
5. Eğitim skoru (aşırı öğrenme farkını görmek için)

```python
from gozetimli.metrikler.siniflandirma import siniflandirma_raporu
print(siniflandirma_raporu(y_test, tahminler, metin=True))
```

**Regresyon:**
1. RMSE ve MAE birlikte (aradaki fark aykırı değer sinyalidir)
2. R² (tercihen düzeltilmiş)
3. Maksimum hata
4. Kalıntı grafiği — metrik değil ama en bilgilendirici tek görsel

```python
from gozetimli.metrikler.regresyon import regresyon_raporu
print(regresyon_raporu(y_test, tahminler, ozellik_sayisi=X.shape[1]))
```

**RMSE ile MAE arasındaki fark ne söyler?**
RMSE ≈ MAE ise hatalar birbirine yakın büyüklüktedir. RMSE, MAE'nin iki
katından fazlaysa birkaç büyük hata ortalamayı taşıyordur — o örneklere bakın.

---

## 5. Eşik 0.5 bir varsayımdır, sonuç değil

Lojistik regresyonun sigmoid çıktısını 0.5'te kesmek yalnızca iki koşul
sağlandığında doğrudur: sınıflar dengelidir **ve** FP ile FN aynı maliyettedir.
İkisi de çoğu gerçek problemde yanlıştır.

```python
from gozetimli.metrikler.siniflandirma import esik_tara

# F1'i maksimize eden eşik
esik, skor, tum_esikler, tum_skorlar = esik_tara(y_dogrulama, olasiliklar, metrik="f1")

# "Recall en az %90 olsun, o kısıt altında precision'ı maksimize et"
esik, kesinlik_degeri, _, _ = esik_tara(
    y_dogrulama, olasiliklar, metrik="kesinlik", en_az_duyarlilik=0.90)
```

**Kritik kural:** eşik **doğrulama** setinde seçilir, test setinde değil. Test
setinde seçilen eşik, test skorunu iyimser yönde bozar — bu da bir sızıntı
biçimidir.

---

## 6. Metriği seçtikten sonra: baz çizgisi kur

Bir skorun iyi olup olmadığı, neyle karşılaştırıldığına bağlıdır. Her zaman en
az bir aptal modelle kıyaslayın:

| Problem | Baz çizgisi | Beklenen skor |
|---|---|---|
| Sınıflandırma | Her zaman çoğunluk sınıfını söyle | accuracy = çoğunluk oranı |
| Sınıflandırma (sıralama) | Rastgele skor | ROC-AUC = 0.5, PR-AUC = pozitif oranı |
| Regresyon | Her zaman ortalamayı söyle | R² = 0 |
| Zaman serisi | "Yarın bugünkü gibi olacak" (naive) | Aşılması şaşırtıcı derecede zordur |

Modeliniz baz çizgisini anlamlı şekilde geçmiyorsa, metrik seçmeden önce
problem tanımına ya da veriye dönün.

---

## 7. Sık yapılan altı hata

1. **Dengesiz veride accuracy raporlamak.** %98 accuracy, %2 pozitifli veride
   hiçbir şey öğrenmemiş bir modeli tarif ediyor olabilir.
2. **Tek bir train/test bölmesine güvenmek.** Küçük veride bölme şansı birkaç
   puan oynatır. En az 5 katlı CV yapın, standart sapmayı raporlayın.
3. **Makro F1'i makro P ve makro R'den hesaplamak.** Makro F1, sınıf başına
   F1'lerin ortalamasıdır; bu iki yol farklı sayılar verir.
4. **Test setini birden çok kez kullanmak.** Her bakış, test setini biraz daha
   doğrulama setine dönüştürür. Test seti bir kez, en sonda.
5. **Eşiği test setinde seçmek.** Bkz. bölüm 5.
6. **Hiperparametre seçimiyle aynı CV'nin skorunu raporlamak.** İç içe (nested)
   CV gerekir; bkz. [capraz-dogrulama.md](capraz-dogrulama.md) bölüm 6.

---

## İlgili belgeler

- [Sınıflandırma metrikleri](siniflandirma-metrikleri.md) — tanımlar ve formüller
- [Regresyon metrikleri](regresyon-metrikleri.md)
- [Çapraz doğrulama](capraz-dogrulama.md) — bölme stratejileri
- [Veri sızıntısı](veri-sizintisi.md) — skorları sahte yapan hatalar
