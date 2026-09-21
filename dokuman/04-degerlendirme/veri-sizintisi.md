# Veri sızıntısı: skorları sahte yapan hatalar

Veri sızıntısı (data leakage), modelin **eğitim sırasında görmemesi gereken
bilgiye** erişmesidir. Sonucu her zaman aynıdır: çapraz doğrulama skoru harika
çıkar, model üretimde çuvallar.

Bu dosyadaki üç örneğin tamamı bu depoda **ölçülür**; sayılar
[`testler/test_capraz.py`](../../testler/test_capraz.py) içindeki
`TestVeriSizintisi` sınıfından gelir ve her test koşusunda yeniden üretilir.

---

## 1. Özellik seçimi sızıntısı: en sinsisi

**Kurulum:** 200 örnek, 2000 tamamen rastgele özellik, tamamen rastgele etiket.
Veride **hiçbir gerçek sinyal yok.** Dürüst skor %50 olmalı.

| Yaklaşım | CV doğruluğu |
|---|---|
| **YANLIŞ:** en iyi 10 özelliği tüm veriden seç, sonra CV yap | **0.7342** |
| **DOĞRU:** seçimi Pipeline'a koy, her katta yeniden seç | 0.5341 |

**20 puanlık sahte başarı.** Hiçbir sinyalin olmadığı veriden.

```python
# YANLIŞ
secici = SelectKBest(f_classif, k=10).fit(X, y)     # y'nin TAMAMINI gördü
X_secili = secici.transform(X)
capraz_dogrula(model, X_secili, y, bolucu=TabakaliKKat(5), ...)

# DOĞRU
boru = Pipeline([("sec", SelectKBest(f_classif, k=10)), ("model", model)])
capraz_dogrula(boru, X, y, bolucu=TabakaliKKat(5), ...)
```

**Neden oluyor?** Seçim aşaması test katlarının etiketlerini görür. 2000
özellik arasından, şans eseri test katına uyan 10 tanesi bulunur. O özellikler
her katta da "uyar" çünkü seçilirken zaten oraya bakılmıştır.

Bu hata özellikle yüksek boyutlu verilerde (genomik, metin, sensör) yaygındır
ve yayımlanmış makalelerde bile görülür. Özellik sayısı örnek sayısından
büyükse bu riski ciddiye alın.

---

## 2. Grup sızıntısı: en sık yapılanı

**Kurulum:** 20 denek, her birinden 10 neredeyse aynı ölçüm (toplam 200 satır).
Denek kimliği hedefi belirliyor. Gerçek soru: **görülmemiş bir denekte** model
ne yapar?

| Bölme | CV doğruluğu |
|---|---|
| **YANLIŞ:** satır bazlı K-Fold | **1.0000** |
| **DOĞRU:** GrupKKat (denek bazlı) | 0.5400 |

**46 puanlık sahte başarı.** Satır bazlı bölmede model mükemmel görünüyor,
çünkü her deneği eğitimde zaten görmüş; yaptığı şey genelleme değil, ezber.

```python
capraz_dogrula(model, X, y, bolucu=GrupKKat(5), gruplar=denek_kimlikleri, ...)
```

**Nerede karşınıza çıkar:**

| Alan | Grup |
|---|---|
| Sağlık | Hasta kimliği (aynı hastanın çoklu ziyareti) |
| Perakende | Mağaza / ürün kimliği |
| Web analitiği | Kullanıcı / oturum kimliği |
| NLP | Kaynak belge (aynı belgeden çıkarılan cümleler) |
| Görüntü | Aynı sahnenin farklı kareleri, aynı hastadan çoklu kesit |
| Üretim | Aynı makine / parti numarası |

**Kural:** veride tekrar eden bir varlık varsa ve model o varlığın kimliğini
dolaylı olarak öğrenebiliyorsa, grup bazlı bölme zorunludur.

---

## 3. Zaman sızıntısı: en kolay fark edileni, yine de yapılanı

**Kurulum:** 200 adımlık bir seri. 100. adımda **rejim değişiyor**: ilişki
tersine dönüyor. Gerçek bir model bunu önceden bilemez.

| Bölme | CV R² |
|---|---|
| **YANLIŞ:** karıştırılmış K-Fold | **-0.04** |
| **DOĞRU:** ZamanSerisiBolme | -22.45 |

K-Fold, doğrusal modelin rejim değişimini tamamen kaçırdığını **gizliyor.**
Dürüst değerlendirme, modelin ortalamayı söylemekten çok daha kötü olduğunu
söylüyor, ki gerçek budur.

```python
from gozetimli.dogrulama.bolme import ZamanSerisiBolme
ZamanSerisiBolme(5)                    # genişleyen pencere
ZamanSerisiBolme(5, bosluk=6)          # 7 adım sonrasını tahmin ediyorsan
```

Zaman serisinde karıştırmak, **geleceği görüp geçmişi tahmin etmektir.**

---

## 4. Ölçekleme ve doldurma sızıntısı: küçük ama gerçek

`StandardScaler`, `SimpleImputer`, `PCA`, hedef kodlama (target encoding):
hepsi **veriden öğrenir**. Tüm veriye uygulanırsa test katının ortalaması,
varyansı veya hedef dağılımı eğitime karışır.

Etkisi özellik seçimi kadar dramatik değildir (genellikle birkaç ondalık),
ama parametreler test verisine bakarak hesaplanmış olur, üretimde
tekrarlanamayan bir avantaj.

```python
# YANLIŞ
X_olcekli = StandardScaler().fit_transform(X)

# DOĞRU
Pipeline([("olcek", StandardScaler()), ("model", model)])
```

**Hedef kodlama (target encoding) özellikle tehlikelidir:** kategori
ortalamalarını hedeften hesaplar. Döngü dışında yapılırsa doğrudan etiket
sızdırır; kat içinde bile out-of-fold kodlama gerekir.

---

## 5. Kaynağında sızıntı: model kurulmadan önce

Yukarıdakiler yöntem hatalarıdır. Bir de **veri toplama** kaynaklı olanlar var
ve bunlar Pipeline ile çözülmez:

| Sızıntı | Örnek |
|---|---|
| **Geleceğe ait sütun** | "Toplam sipariş tutarı" ile iptal tahmini: tutar iptalden sonra güncelleniyor |
| **Hedefin vekili (proxy)** | Hastalık tahmininde "reçete edilen ilaç" sütunu |
| **Sonradan güncellenen alan** | "Son durum", "kapanış tarihi" gibi olay sonrası dolan sütunlar |
| **Tekrarlanan satırlar** | Aynı kaydın kopyası hem eğitimde hem testte |
| **Önceden filtrelenmiş veri** | Yalnızca onaylanan krediler üzerinde temerrüt modeli (survivorship bias) |

**Teşhis:** modeliniz beklenenden çok iyi çalışıyorsa, bu iyi haber değildir.
Şu iki soruyu sorun:

1. **"Bu sütunun değeri, tahmin anında elimde olur muydu?"**
2. **"Bu sütun, tahmin etmeye çalıştığım şeyin bir sonucu mu?"**

İkisinden birine "hayır/evet" ise sütun çıkmalıdır.

---

## 6. Kontrol listesi

Bir CV kurulumunu teslim etmeden önce:

- [ ] Ölçekleme, doldurma, kodlama, özellik seçimi: hepsi Pipeline içinde mi?
- [ ] Veride tekrar eden bir varlık (kişi/mağaza/belge) var mı? Varsa `GrupKKat`.
- [ ] Veri zamana bağlı mı? Öyleyse `ZamanSerisiBolme`, gerekiyorsa `bosluk`.
- [ ] Sınıflandırmada `TabakaliKKat` kullanıldı mı?
- [ ] Hiperparametre seçildiyse nested CV yapıldı mı, yoksa ayrı bir test seti mi var?
- [ ] Karar eşiği doğrulama setinde mi seçildi?
- [ ] Her sütun için "tahmin anında elimde olur muydu?" sorusu soruldu mu?
- [ ] Tekrarlanan satırlar temizlendi mi?
- [ ] Test seti gerçekten bir kez mi kullanıldı?

---

## 7. Son uyarı: şüphe uyandıran iyi sonuç

Deneyimli bir bakış açısıyla:

> Bir sınıflandırma modeli ilk denemede %99 doğruluk veriyorsa, en olası üç
> açıklama şudur: (1) veri sızıntısı, (2) hedefin bir vekili özelliklere
> karışmış, (3) problem gerçekten kolay. İlk ikisi, üçüncüsünden çok daha
> yaygındır.

Sürpriz derecede iyi bir skor, kutlanacak değil **araştırılacak** bir bulgudur.
