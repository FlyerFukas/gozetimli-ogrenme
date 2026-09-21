# Karar ağaçları

Kaynak ders notu: `13-Decision_Tree_Algorithms.pdf`
Kod: [`gozetimli/saflik.py`](../../src/gozetimli/saflik.py) ·
Test: [`testler/test_saflik.py`](../../testler/test_saflik.py)

> ⚠️ Bu ders notunda **iki hesap hatası** tespit edildi (kök varyans ve en iyi
> eşik seçimi). Ayrıntı ve doğrulama: [KAYNAK-NOTLARI.md](../KAYNAK-NOTLARI.md).

---

## 1. Temel fikir

Karar ağacı, veriyi ardışık if-else sorularıyla böler. Her bölme, alt kümeleri
**daha homojen** yapmayı amaçlar. Yorumlanabilirliği en yüksek modellerden
biridir: bir tahminin gerekçesi, kökten yaprağa giden yoldur.

### ID3 vs CART

| | ID3 / C4.5 | CART |
|---|---|---|
| Dallanma | Çok yollu (her kategori bir dal) | **Yalnızca ikili** |
| Ölçüt | Entropi / kazanç oranı | Gini (varsayılan) |
| Regresyon | Hayır | **Evet** |
| scikit-learn |: | CART kullanır |

scikit-learn yalnızca CART uygular, yani her düğüm iki dala ayrılır.
Kategorik bir değişken için `outlook ∈ {sunny, rainy}` gibi ikili bölmeler
üretir.

---

## 2. Saflık ölçütleri

### Entropi

```
H(S) = -Σ pᵢ · log₂(pᵢ)
```

Bilgi teorisinden gelir: "bu kümedeki belirsizlik kaç bit?" Tek sınıflı küme
0, iki sınıfın yarı yarıya olduğu küme 1 bit. Aralık: [0, log₂(k)].

### Gini saflıksızlığı

```
G(S) = 1 - Σ pᵢ²
```

"Rastgele seçilen bir örneğin yanlış sınıflandırılma olasılığı." Aralık:
[0, 1 - 1/k]. İki sınıfta en fazla 0.5.

```python
from gozetimli.saflik import entropi, gini
entropi([0, 1, 1, 0, 1, 1, 0])   # 0.9852
gini([0, 1, 1, 0, 1, 1, 0])      # 0.4898
```

### Hangisi?

Ders notunun değerlendirmesi doğru: *"Gini hesaplaması daha hızlıdır ve genelde
benzer sonuçlar verir."* Pratikte aynı ağacı üretirler. Gini logaritma
içermediği için varsayılan olmuştur.

Üçüncü bir ölçüt daha var: **sınıflandırma hatası** (1 - max pᵢ). Ağaç
*büyütmek* için kötüdür (çoğu bölmede türevi sıfırdır, iyileşmeyi göremez) ama
*budama* aşamasında kullanılır çünkü doğrudan hatayı temsil eder.

---

## 3. Bilgi kazancı

```
Gain = Saflık(kök) - Σ (|Sᵢ|/|S|) · Saflık(Sᵢ)
```

Ağırlıklandırma şart: 1 örnekli saf yaprak ile 100 örnekli saf yaprak aynı
değerde sayılamaz.

**Ders notu örneği** (7 örnek: 4 Onay, 3 Red):

| Bölme | Kazanç |
|---|---|
| Mülakat = Yüksek | **0.521** ← seçilir |
| GPA > 3.0 | 0.021 |

```python
from gozetimli.saflik import bilgi_kazanci, en_iyi_bolunme
bilgi_kazanci(y, maske=(mulakat == "Yüksek"))
en_iyi_bolunme(X, y, olcut="entropi", ozellik_adlari=["gpa", "mulakat"])
```

**Kazanç asla negatif olamaz:** saflık ölçütleri içbükey olduğu için. Bu, test
dosyasında rastgele verilerle doğrulanır.

### Kazanç oranı (C4.5): ham kazancın tuzağı

Ham bilgi kazancı, **çok değerli kategorik özellikleri kayırır.** "Müşteri
kimliği" gibi her satırda farklı olan bir sütun, her yaprağı tek örnekli yapar,
entropiyi sıfırlar ve en yüksek kazancı alır, ama hiçbir şey öğretmez.

```
SplitInfo = -Σ (|Sᵢ|/|S|) · log₂(|Sᵢ|/|S|)
GainRatio = Gain / SplitInfo
```

Bölme bilgisine bölmek bu kayırmayı giderir. Test dosyasındaki
`test_cok_dalli_bolmeyi_cezalandirir` bunu ölçerek gösterir.

---

## 4. Regresyon ağaçları: varyans azaltımı

Saflık ölçütleri sürekli hedefte anlamsızdır. Yerine:

```
Varyans azaltımı = Var(kök) - Σ (|Sᵢ|/|S|) · Var(Sᵢ)
```

Bu, **ağırlıklı MSE azalışıyla aynı bölmeyi seçer** (test dosyasında
doğrulanır). Yaprak tahmini, o yapraktaki hedeflerin **ortalamasıdır:** çünkü
ortalama, MSE'yi minimize eden sabittir.

**Önemli:** ders notundaki hesaplar gibi bu implementasyon da **ddof=0**
(popülasyon varyansı) kullanır. pandas'ın `.var()` varsayılanı ddof=1'dir ve
küçük yapraklarda belirgin farklı sayı verir.

### Sürekli değişkende eşik arama

Ders notundaki GPA örneği yöntemi doğru anlatıyor: değerler sıralanır, ardışık
farklı değerlerin **orta noktaları** aday eşik olur.

```python
from gozetimli.saflik import esik_adaylari, varyans_azaltimi
esik_adaylari([2.5, 2.7, 3.0, 3.2, 3.5, 3.7, 4.0])
# [2.6, 2.85, 3.1, 3.35, 3.6, 3.85]
```

Orta nokta kullanmak, eşiğin iki gözlem arasına düşmesini garanti eder.

> **Ders notundaki hata.** Not, GPA<3.1 eşiğinin en yüksek varyans azaltımını
> (94.64) verdiğini söylüyor. Yeniden hesaplandığında en iyi eşik **3.35**'tir
> (azaltım 93.12); 3.1 ise 83.06 verir. Hatanın kaynağı kök varyansın 124.83
> olarak verilmesi (doğrusu 113.27). Ayrıntı ve doğrulama testi:
> [KAYNAK-NOTLARI.md](../KAYNAK-NOTLARI.md).

---

## 5. Aşırı öğrenme ve budama

Budanmamış bir ağaç, her yaprakta tek örnek kalana kadar büyür: eğitim
doğruluğu %100, test doğruluğu düşük. Bu, karar ağacının **yapısal**
zayıflığıdır, bu yüzden tek başına nadiren kullanılır.

### Ön budama (pre-pruning): büyümeyi sınırla

| Parametre | Etkisi |
|---|---|
| `max_depth` | En etkili tek düğme. 3-10 tipik |
| `min_samples_split` | Bölmek için gereken en az örnek |
| `min_samples_leaf` | Yaprakta kalması gereken en az örnek |
| `max_features` | Her bölmede denenecek özellik sayısı |
| `min_impurity_decrease` | Bölmenin sağlaması gereken en az kazanç |

### Son budama (post-pruning): büyüt, sonra kes

scikit-learn'de `ccp_alpha` (cost-complexity pruning). Ağaç tam büyütülür,
sonra "karmaşıklık başına getiri"si düşük dallar kesilir. Genellikle ön
budamadan daha iyi sonuç verir çünkü bir bölme tek başına zayıf olup altındaki
bölmeleri mümkün kılıyor olabilir.

```python
from sklearn.tree import DecisionTreeClassifier
from gozetimli.dogrulama.capraz import capraz_dogrula
from gozetimli.dogrulama.bolme import TabakaliKKat
from gozetimli.metrikler.siniflandirma import dogruluk

for derinlik in (1, 3, 5, 10, None):
    ozet = capraz_dogrula(DecisionTreeClassifier(max_depth=derinlik, random_state=0),
                          X, y, bolucu=TabakaliKKat(5),
                          metrikler={"d": dogruluk}, egitim_skoru=True).ozet()
    print(derinlik, ozet["d"]["ortalama"], ozet["d"]["asiri_ogrenme_farki"])
```

`asiri_ogrenme_farki` (eğitim - test) doğrudan budama ihtiyacını gösterir.

---

## 6. Artı ve eksileri

**Artı:**
- En yorumlanabilir modellerden biri; bir tahminin gerekçesi bir yoldur
- Ölçekleme gerektirmez (sıralamaya bakar, mesafeye değil)
- Sayısal ve kategorik özellikleri birlikte işler
- Doğrusal olmayan ilişkileri ve etkileşimleri doğal olarak yakalar
- Aykırı değerlere dayanıklı (bölme noktası değişmez)

**Eksi:**
- **Kararsız:** verinin küçük bir değişimi tamamen farklı bir ağaç üretebilir.
  Bu, topluluk yöntemlerinin varlık sebebidir.
- Tek başına aşırı öğrenmeye çok yatkın
- Eksenlere paralel bölmeler yapar; çapraz bir karar sınırını merdiven basamağı
  gibi yaklaşık olarak temsil eder
- Regresyonda **ekstrapolasyon yapamaz:** eğitim aralığı dışında sabit tahmin
  verir. Zaman serisi trendinde ölümcül bir sınırlamadır.
- Özellik önemleri yanlıdır: yüksek kardinaliteli (çok farklı değerli)
  özellikler şişirilmiş önem alır. Permütasyon önemini tercih edin.

---

## Sonraki adım

Karar ağacının kararsızlığı bir kusur gibi görünür ama topluluk yöntemleri için
**avantaja** dönüşür: çeşitlilik, ortalamanın işe yaramasını sağlar.

→ [Topluluk öğrenme ve boosting](topluluk-ve-boosting.md)
