# Örnekler

Üçü de bağımsız çalışır ve konsola yazar; dosya üretmez.

```bash
pip install -e ".[ornek]"
python ornekler/01-siniflandirma-boru-hatti.py
python ornekler/02-ders-hesaplari.py
python ornekler/03-sizinti-karsilastirmasi.py
```

| Betik | Ne gösteriyor |
|---|---|
| **01-siniflandirma-boru-hatti.py** | Dengesiz veride uçtan uca doğru kurulum: üçlü ayrım, Pipeline, tabakalı CV, doğrulama setinde eşik seçimi, test setinin tek kullanımı |
| **02-ders-hesaplari.py** | Ders notlarındaki her el hesabını yeniden üretir; tutmayanları işaretler |
| **03-sizinti-karsilastirmasi.py** | Üç sızıntı türünün skorlara etkisini ölçer (yanlış vs doğru kurulum) |

---

## 01 — ne öğretiyor?

Eşik seçiminin gerçek etkisi bu betikte görünür. Aynı model, aynı veri:

| Eşik | F1 |
|---|---|
| 0.50 (varsayılan) | 0.5467 |
| 0.7688 (F1'i maksimize eden) | 0.6400 |

Ayrıca bir iş kısıtının nasıl uygulandığını gösterir: "duyarlılık en az %90
olsun, o kısıt altında kesinliği maksimize et."

## 02 — ne buluyor?

Ders notlarındaki 25+ el hesabının çoğu tutuyor; ikisi tutmuyor ve betik
bunları açıkça işaretliyor:

- Kök varyans: notta 124.83, hesap 113.2653
- En iyi eşik: notta GPA<3.1, hesap GPA<3.35
- XGBoost sağ similarity: notta 0.702, hesap 0.7191

Ayrıntı: [dokuman/KAYNAK-NOTLARI.md](../dokuman/KAYNAK-NOTLARI.md)

## 03 — ne ölçüyor?

| Senaryo | Yanlış | Doğru | Sahte kazanç |
|---|---|---|---|
| Özellik seçimi tüm veriden | 0.7342 | 0.5341 | +0.2001 |
| Satır bazlı CV (gruplu veri) | 1.0000 | 0.5400 | +0.4600 |
| Zaman serisinde K-Fold (R²) | -0.04 | -22.45 | +22.41 |

İlk satırda etiketler **tamamen rastgeledir** — yani %73 doğruluk tümüyle
sızıntıdan geliyor.
