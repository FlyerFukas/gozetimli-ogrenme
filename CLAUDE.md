# Claude Code için repo rehberi

Bu depo bir **referans kütüphanesi**dir: makine öğrenmesi projelerinde metrik,
çapraz doğrulama ve model değerlendirme işlerini yaparken buradaki fonksiyonlar
ve belgeler kullanılır.

---

## Bu depo bir projede nasıl kullanılır?

Başka bir projede referans olarak kullanmanın iki yolu var:

```bash
# 1) Doğrudan kurulum (önerilen)
pip install "gozetimli-ogrenme @ git+https://github.com/FlyerFukas/gozetimli-ogrenme.git"

# 2) Yerel geliştirme kopyası
pip install -e C:/Users/furka/Music/gozetimli-ogrenme
```

Sonra:

```python
from gozetimli.metrikler.siniflandirma import siniflandirma_raporu, esik_tara
from gozetimli.dogrulama.bolme import TabakaliKKat, GrupKKat, ZamanSerisiBolme
from gozetimli.dogrulama.capraz import capraz_dogrula, ic_ice_capraz_dogrula
```

---

## Bir değerlendirme kodu yazarken izlenecek sıra

**1. Bölücüyü doğru seç.** Bu, en sık yapılan hatanın kaynağı:

```
Veri zamana bağlı mı?         → ZamanSerisiBolme  (gerekiyorsa bosluk=h-1)
Satırlar gruplanıyor mu?      → GrupKKat          (gruplar= zorunlu)
Sınıflandırma mı?             → TabakaliKKat
Diğer                         → KKat
```

Tekrar eden bir varlık (hasta, mağaza, kullanıcı, belge) varsa `GrupKKat`
kullanılmadığında skorlar 20–45 puan şişer. Ölçülmüş örnek:
[`dokuman/04-degerlendirme/veri-sizintisi.md`](dokuman/04-degerlendirme/veri-sizintisi.md)

**2. Ön işlemenin tamamını Pipeline'a koy.** Ölçekleme, doldurma, kodlama,
özellik seçimi — hepsi CV döngüsünün içinde olmalı. Döngü dışında yapılan
özellik seçimi, sinyalsiz veride bile %73 doğruluk üretir.

**3. Metriği probleme göre seç**, varsayılana göre değil:
[`dokuman/04-degerlendirme/metrik-secim-rehberi.md`](dokuman/04-degerlendirme/metrik-secim-rehberi.md)
karar tablolarını içerir. Kısaca:
- Dengesiz veri → `f1_skoru`, `matthews_korelasyonu`, `ortalama_kesinlik`
- FN pahalı → `f_beta_skoru(beta=2)`
- FP pahalı → `f_beta_skoru(beta=0.5)`
- Olasılık kullanılacak → `log_kaybi`
- Regresyonda rapor → `rmse` + `mae` birlikte (fark aykırı değer sinyalidir)

**4. Ortalamanın yanında standart sapmayı da raporla.** `sonuc.ozet()` bunu
zaten verir. Tek bir ortalama sayı, modelin kararlı olup olmadığını gizler.

**5. Hiperparametre seçtiysen nested CV kullan** ya da ayrı test seti ayır.
Izgara aramasının en iyi CV skorunu "modelin performansı" diye raporlamak
iyimserdir.

**6. Karar eşiğini doğrulama setinde seç.** `esik_tara` bunu yapar. 0.5
varsayılanı yalnızca sınıflar dengeliyse ve hata maliyetleri eşitse doğrudur.

---

## Kod yazma kuralları

**Dil.** Kod, yorumlar, docstring'ler, hata mesajları, test adları — hepsi
Türkçe. Değişken ve fonksiyon adları Türkçe (`kesinlik`, `bolucu`,
`kat_skorlari`). İngilizce terimin kendisi gerektiğinde parantez içinde
verilir: `duyarlilik` (recall). Bu tutarlılığı bozmayın.

**Bağımlılık.** Kütüphane kodu yalnızca **NumPy** kullanır. scikit-learn,
pandas, matplotlib sadece `testler/` ve `ornekler/` içinde kullanılabilir.
`src/gozetimli/` altına yeni bir zorunlu bağımlılık eklemeyin.

**Hata mesajları.** Sorunu ve çözümü birlikte söyler:

```python
raise ValueError(
    f"kat_sayisi ({self.katlar}) örnek sayısından ({n}) büyük olamaz."
)
```

Sadece "invalid input" yazmayın; hangi değerin ne olduğunu ve ne olması
gerektiğini yazın.

**Docstring.** Her genel fonksiyonun docstring'i formülü, aralığı ve **ne zaman
kullanılmayacağını** içerir. Metrik dokümantasyonunda "bu metriği ne zaman
kullanmamalı" bilgisi, tanımdan daha değerlidir.

---

## Test kuralları

Yeni bir fonksiyon eklerken en az şu üç testi yazın:

1. **Elle hesaplanmış bir değer** — docstring'de hesabın kendisi yazılı olmalı
2. **Kenar durum** — boş, tek sınıf, sıfıra bölme, uzunluk uyuşmazlığı
3. **Özellik (property)** — simetri, sınır, monotonluk gibi değişmezler

scikit-learn'de karşılığı olan her metrik için
[`testler/test_sklearn_uyumu.py`](testler/test_sklearn_uyumu.py) içine bir
karşılaştırma ekleyin. Bu dosya kütüphanenin sayısal doğruluğunun tek
garantisidir.

```bash
pytest -q                          # hepsi (562 test, ~6 sn)
pytest -m ders -v                  # ders el hesapları
pytest -m sklearn -q               # sklearn uyumu
```

**Bir test tolerans gevşetilerek geçiriliyorsa durun.** Önce beklenen değerin
doğru olup olmadığını kontrol edin. Bu depodaki beş kaynak hatası tam olarak bu
şekilde bulundu.

---

## Kaynak materyal ve telif — dikkat

Bu deponun içeriği Atıl Samancıoğlu'nun ML kursundaki ders notları çalışılarak
yazıldı. **Ders notu PDF'leri bu depoya asla eklenmez** — telifli üçüncü taraf
materyali ve depo public.

`.gitignore` içinde `*.pdf` ve `ders-notlari/` bilinçli olarak engellidir. Bir
PDF'i depoya eklemek için `git add -f` gerekir; bunu yapmayın.

Ders notlarındaki sayısal hatalar
[`dokuman/KAYNAK-NOTLARI.md`](dokuman/KAYNAK-NOTLARI.md) içinde belgelendi. Bir
hesap ders notuyla uyuşmuyorsa önce kendi hesabınızı doğrulayın; kaynak yanlış
olabilir ve bu beş kez gerçekleşti.

---

## Depo haritası

```
src/gozetimli/
├── ortak.py                     Girdi doğrulama, güvenli bölme
├── saflik.py                    Entropi, Gini, bilgi kazancı, varyans azaltımı
├── boosting.py                  Sigmoid/log-odds, AdaBoost, XGBoost formülleri
├── metrikler/
│   ├── siniflandirma.py         18 fonksiyon — matris, P/R/F, MCC, ROC, PR, eşik
│   └── regresyon.py             13 fonksiyon — MSE/MAE/R²/MAPE/MSLE ailesi
├── dogrulama/
│   ├── bolme.py                 7 bölme stratejisi + egitim_test_bol
│   └── capraz.py                CV çalıştırıcı, OOF, nested CV, öğrenme eğrisi
└── veri/ders_verileri.py        Ders notlarındaki 10 mini veri seti

testler/                         562 test, dört doğrulama katmanı
dokuman/                         Türkçe referans belgeleri
ornekler/                        Çalıştırılabilir uçtan uca örnekler
```

---

## Git

Bu depo **FlyerFukas** hesabında. Depo-yerel kimlik ayarı zorunlu:

```bash
git config user.name "FlyerFukas"
git config user.email "250136320+FlyerFukas@users.noreply.github.com"
```

`furkanakduman3452@gmail.com` ile commit atmayın — GitHub hesabı üniversite
adresine bağlı ve Vercel Hobby planı yabancı commit yazarlı dağıtımı
`Deployment Blocked` ile reddediyor.

**Not:** hesap eskiden `mrFurkan33333` adındaydı, sonra `FlyerFukas` oldu.
Kullanıcı ID'si (250136320) değişmedi, bu yüzden eski noreply adresiyle atılmış
commit'ler de hesaba bağlı kalır. Yeni commit'lerde yukarıdaki adresi kullanın.
