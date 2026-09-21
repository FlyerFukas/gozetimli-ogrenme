# gozetimli-ogrenme

**Gözetimli öğrenme için Türkçe referans: metrikler, çapraz doğrulama ve ağaç/boosting ölçütleri — sıfırdan yazılmış, scikit-learn'e karşı doğrulanmış, 562 testle korunan.**

Bu depo bir "ders notu özeti" değil. Üç şey yapar:

1. **Değerlendirme araçlarını çalıştırılabilir kod olarak verir** — 18 sınıflandırma
   metriği, 13 regresyon metriği, 7 çapraz doğrulama stratejisi, ağaç bölme ve
   boosting hesapları. Hepsi saf NumPy, hepsi Türkçe adlandırılmış.
2. **Hangi metriğin ne zaman kullanılacağını karar tablolarıyla anlatır** —
   teknik tanım değil, iş kararı olarak.
3. **Kaynak materyaldeki hesap hatalarını bulur ve belgeler** — her el hesabı
   kodla yeniden üretildi; tutmayan beş tanesi doğru değerleriyle yazıldı.

```bash
pytest -q
# 562 passed in 5.93s
```

---

## Kurulum

```bash
git clone https://github.com/FlyerFukas/gozetimli-ogrenme.git
cd gozetimli-ogrenme
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[test]"
pytest -q
```

Kütüphanenin kendisi yalnızca **NumPy** ister. scikit-learn sadece testler ve
örnekler için gerekli.

---

## Beş dakikada

```python
from gozetimli.metrikler.siniflandirma import siniflandirma_raporu, esik_tara
from gozetimli.dogrulama.bolme import TabakaliKKat
from gozetimli.dogrulama.capraz import capraz_dogrula
from gozetimli.metrikler.siniflandirma import f1_skoru, roc_auc

# Kat kat çapraz doğrulama — aşırı öğrenme farkıyla birlikte
sonuc = capraz_dogrula(model, X, y,
                       bolucu=TabakaliKKat(5, karistir=True, tohum=42),
                       metrikler={"f1": f1_skoru, "auc": roc_auc},
                       olasilik_metrikleri=("auc",),
                       egitim_skoru=True)
print(sonuc.ozet())

# Sınıf başına rapor
print(siniflandirma_raporu(y_test, tahminler, metin=True))

# Karar eşiğini veriye göre seç — 0.5 kutsal bir sayı değil
esik, skor, _, _ = esik_tara(y_dogrulama, olasiliklar,
                             metrik="kesinlik", en_az_duyarlilik=0.90)
```

---

## İçerik

### Kod — [`src/gozetimli/`](src/gozetimli/)

| Modül | İçerik |
|---|---|
| [`metrikler/siniflandirma.py`](src/gozetimli/metrikler/siniflandirma.py) | Karmaşıklık matrisi, precision/recall/F-beta, MCC, kappa, log-loss, ROC & PR eğrileri, eşik tarama |
| [`metrikler/regresyon.py`](src/gozetimli/metrikler/regresyon.py) | MSE, RMSE, MAE, MAPE, sMAPE, MSLE, R², düzeltilmiş R², açıklanan varyans |
| [`dogrulama/bolme.py`](src/gozetimli/dogrulama/bolme.py) | K-Fold, Stratified, LOOCV, Leave-P-Out, Time Series, GrupKKat, TekrarliKKat |
| [`dogrulama/capraz.py`](src/gozetimli/dogrulama/capraz.py) | Çapraz doğrulama çalıştırıcısı, OOF tahmin, **iç içe (nested) CV**, öğrenme eğrisi |
| [`saflik.py`](src/gozetimli/saflik.py) | Entropi, Gini, bilgi kazancı, kazanç oranı, varyans azaltımı, en iyi bölme |
| [`boosting.py`](src/gozetimli/boosting.py) | Sigmoid/log-odds, AdaBoost ağırlıkları, XGBoost similarity/gain/cover |
| [`veri/ders_verileri.py`](src/gozetimli/veri/ders_verileri.py) | Ders notlarındaki 10 öğretici mini veri seti |

### Belgeler — [`dokuman/`](dokuman/)

**Değerlendirme (deponun çekirdeği):**
- **[Metrik seçim rehberi](dokuman/04-degerlendirme/metrik-secim-rehberi.md)** — hangi kıstas, ne zaman? Karar tabloları.
- [Sınıflandırma metrikleri](dokuman/04-degerlendirme/siniflandirma-metrikleri.md)
- [Regresyon metrikleri](dokuman/04-degerlendirme/regresyon-metrikleri.md)
- [Çapraz doğrulama](dokuman/04-degerlendirme/capraz-dogrulama.md) — 7 strateji + nested CV
- **[Veri sızıntısı](dokuman/04-degerlendirme/veri-sizintisi.md)** — ölçülmüş örneklerle

**Algoritmalar:**
- [Polinom regresyon ve düzenlileştirme](dokuman/01-regresyon/polinom-ve-duzenlilestirme.md)
- [Lojistik regresyon](dokuman/02-siniflandirma/lojistik-regresyon.md)
- [SVM, KNN, Naive Bayes](dokuman/02-siniflandirma/svm-knn-naive-bayes.md)
- [Karar ağaçları](dokuman/03-agac-tabanli/karar-agaci.md)
- [Topluluk öğrenme ve boosting](dokuman/03-agac-tabanli/topluluk-ve-boosting.md)

**[Kaynak notları](dokuman/KAYNAK-NOTLARI.md)** — ders materyalinde tespit edilen hatalar.

---

## Testler — sayıların gerçekten doğru olduğunun kanıtı

562 test, dört ayrı doğrulama katmanı:

| Katman | Dosya | Ne yapar |
|---|---|---|
| **Ders el hesapları** | [`test_ders_ornekleri.py`](testler/test_ders_ornekleri.py) | Ders notlarındaki her ara değeri yeniden üretir (51 test) |
| **scikit-learn uyumu** | [`test_sklearn_uyumu.py`](testler/test_sklearn_uyumu.py) | 40 rastgele senaryoda her metriği sklearn'e karşı karşılaştırır (255 test) |
| **Birim testler** | `test_*_metrikleri.py`, `test_bolme.py`, `test_saflik.py`, `test_boosting.py` | Elle hesaplanmış değerler, kenar durumlar, hata mesajları |
| **Sızıntı gösterimleri** | [`test_capraz.py`](testler/test_capraz.py) | Yanlış CV kurulumunun sahte başarısını **ölçer** |

```bash
pytest -q                          # hepsi
pytest -m ders -v                  # sadece ders el hesapları
pytest -m sklearn -q               # sadece sklearn uyumu
pytest testler/test_capraz.py -v   # sızıntı gösterimleri
```

### Sızıntı gösterimleri ne gösteriyor?

Testlerin ölçtüğü gerçek sayılar:

| Senaryo | Yanlış kurulum | Doğru kurulum | Sahte kazanç |
|---|---|---|---|
| Özellik seçimi tüm veriden (sinyalsiz veri) | 0.7342 | 0.5341 | **+20 puan** |
| Satır bazlı CV (20 denek, 10'ar ölçüm) | 1.0000 | 0.5400 | **+46 puan** |
| Zaman serisinde K-Fold (R²) | -0.04 | -22.45 | gerçeği gizliyor |

İlk satır özellikle çarpıcı: **tamamen rastgele etiketli veride %73 doğruluk.**

---

## Ders materyalinde bulunan hatalar

Kaynak notlardaki tüm el hesapları doğrulandı. Beşi tutmadı:

| # | Konu | Notta | Doğrusu |
|---|---|---|---|
| 1 | Karar ağacı kök varyansı | 124.83 | **113.2653** |
| 2 | En iyi bölme eşiği | GPA < 3.1 | **GPA < 3.35** |
| 3 | XGBoost sağ similarity | 0.702 | **0.7191** |
| 4 | XGBoost S_kök = 0 gerekçesi | "tek grup var" | **Σ artık = 0** |
| 5 | XGBoost Gain formülü | ½ ve γ eksik | makale biçimi de sunuldu |

Ayrıca 1–4 numaralı PDF'ler klasörde yoktu; regresyon metrikleri o yüzden
standart literatürden alındı ve ayrıca işaretlendi.

Her bulgu bir testle korunuyor:
[KAYNAK-NOTLARI.md](dokuman/KAYNAK-NOTLARI.md) · `pytest -m ders -v`

---

## Tasarım kararları

**Neden Türkçe?** Bu depo Türkçe çalışan projelerde referans olarak
kullanılıyor. `duyarlilik(y, t)` ile `recall_score(y, t)` aynı şeyi yapar ama
biri okunurken çeviri yapılmıyor. Her fonksiyonun docstring'inde İngilizce
karşılığı da yazılı.

**Neden sıfırdan yazıldı?** scikit-learn zaten var ve daha hızlı. Buradaki
implementasyonların amacı onu değiştirmek değil, **içini görünür kılmak**:
precision'ın hangi payda ile hesaplandığı, makro ortalamanın nasıl alındığı,
tabakalı bölmenin oranları nasıl koruduğu kodda okunabilir. Sayısal
doğrulukları scikit-learn'e karşı sürekli test ediliyor.

**Neden bölücüler sklearn uyumlu?** `split()` ve `get_n_splits()` takma adları
sayesinde bu bölücüler doğrudan `cross_val_score(..., cv=TabakaliKKat(5))`
şeklinde kullanılabilir.

---

## Lisans

**PolyForm Noncommercial 1.0.0 + ayrı ticari lisans.**

Kişisel kullanım, öğrenme, akademik araştırma, eğitim kurumları, kamu ve hayır
kurumları için **ücretsiz**. Bir işletme için ya da ticari bir amaçla kullanım
**ayrı, ücretli lisans gerektirir**.

- [LICENSE](LICENSE) — bağlayıcı metin
- [COMMERCIAL.md](COMMERCIAL.md) — ticari lisans koşulları ve iletişim

---

## Kaynak ve atıf

Bu deponun içeriği **Atıl Samancıoğlu**'nun makine öğrenmesi kursundaki
gözetimli öğrenme ders notları çalışılarak yazılmıştır. Kavramsal çerçeve ve
öğretici örnekler oradan gelir.

Ders notlarının kendisi (PDF'ler, şekiller, metin) **bu depoda yer almaz** ve
eklenmemelidir — telif hakları kendi sahibine aittir. Depodaki her satır metin
ve kod özgün olarak yazılmıştır.
