"""Veri sızıntısının skorlara etkisini ölçer.

Üç senaryonun her birinde aynı veri iki kez değerlendirilir: bir kez yanlış
kurulumla, bir kez doğru kurulumla. Aradaki fark, sızıntının ürettiği sahte
başarıdır.

Çalıştır:  python ornekler/03-sizinti-karsilastirmasi.py
Gereksinim: pip install -e ".[ornek]"
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from gozetimli.dogrulama.bolme import (GrupKKat, KKat, TabakaliKKat,
                                       ZamanSerisiBolme)
from gozetimli.dogrulama.capraz import capraz_dogrula
from gozetimli.metrikler.regresyon import r2
from gozetimli.metrikler.siniflandirma import dogruluk


def baslik(metin: str) -> None:
    print(f"\n{'=' * 72}\n{metin}\n{'=' * 72}")


def ortalama_skor(model, X, y, bolucu, gruplar=None, metrik=dogruluk) -> float:
    sonuc = capraz_dogrula(model, X, y, bolucu=bolucu, gruplar=gruplar,
                           metrikler={"m": metrik})
    return float(np.mean(sonuc.kat_skorlari["m"]))


def karsilastir(yanlis: float, dogru: float, birim: str = "") -> None:
    print(f"\n  {'YANLIŞ kurulum':<28} {yanlis:>10.4f}{birim}")
    print(f"  {'DOĞRU kurulum':<28} {dogru:>10.4f}{birim}")
    print(f"  {'SAHTE KAZANÇ':<28} {yanlis - dogru:>+10.4f}{birim}")


def ozellik_secimi_sizintisi() -> None:
    baslik("1. ÖZELLİK SEÇİMİ SIZINTISI")
    print("Kurulum: 200 örnek, 2000 rastgele özellik, RASTGELE etiket.")
    print("Veride hiçbir gerçek sinyal yok — dürüst skor 0.50 olmalı.")

    uretec = np.random.default_rng(42)
    X = uretec.normal(size=(200, 2000))
    y = uretec.integers(0, 2, size=200)

    # YANLIŞ: seçim tüm veriden, y'nin tamamı görülüyor
    secici = SelectKBest(f_classif, k=10).fit(X, y)
    yanlis = ortalama_skor(LogisticRegression(max_iter=1000),
                           secici.transform(X), y, TabakaliKKat(5))

    # DOĞRU: seçim Pipeline içinde, her katta yeniden
    boru = Pipeline([("sec", SelectKBest(f_classif, k=10)),
                     ("model", LogisticRegression(max_iter=1000))])
    dogru = ortalama_skor(boru, X, y, TabakaliKKat(5))

    karsilastir(yanlis, dogru)
    print("\n  Seçim aşaması test katlarının etiketlerini gördü. 2000 özellik")
    print("  arasından şans eseri uyan 10 tanesi bulundu; o özellikler her")
    print("  katta da 'uyuyor' çünkü seçilirken zaten oraya bakılmıştı.")


def grup_sizintisi() -> None:
    baslik("2. GRUP SIZINTISI")
    print("Kurulum: 20 denek, her birinden 10 neredeyse aynı ölçüm (200 satır).")
    print("Gerçek soru: GÖRÜLMEMİŞ bir denekte model ne yapar?")

    uretec = np.random.default_rng(3)
    denek_sayisi, olcum = 20, 10
    gruplar = np.repeat(np.arange(denek_sayisi), olcum)
    y = np.repeat(uretec.integers(0, 2, denek_sayisi), olcum)
    imza = uretec.normal(size=(denek_sayisi, 3))
    X = (np.repeat(imza, olcum, axis=0)
         + uretec.normal(0, 0.01, size=(denek_sayisi * olcum, 3)))

    model = KNeighborsClassifier(1)
    yanlis = ortalama_skor(model, X, y, KKat(5, karistir=True, tohum=0))
    dogru = ortalama_skor(model, X, y, GrupKKat(5), gruplar=gruplar)

    karsilastir(yanlis, dogru)
    print("\n  Satır bazlı bölmede model her deneği eğitimde zaten görmüş.")
    print("  Yaptığı şey genelleme değil, ezber.")


def zaman_sizintisi() -> None:
    baslik("3. ZAMAN SIZINTISI")
    print("Kurulum: 200 adımlık seri, 100. adımda REJİM DEĞİŞİYOR.")
    print("Gerçek bir model bunu önceden bilemez.")

    n = 200
    zaman = np.arange(n, dtype=float)
    uretec = np.random.default_rng(11)
    y = np.where(zaman < 100, 0.5 * zaman, 50 - 0.5 * (zaman - 100))
    y = y + uretec.normal(0, 1, n)
    X = zaman.reshape(-1, 1)

    yanlis = ortalama_skor(LinearRegression(), X, y,
                           KKat(5, karistir=True, tohum=0), metrik=r2)
    dogru = ortalama_skor(LinearRegression(), X, y,
                          ZamanSerisiBolme(5), metrik=r2)

    karsilastir(yanlis, dogru, birim=" R²")
    print("\n  K-Fold, modelin rejim değişimini tamamen kaçırdığını GİZLİYOR.")
    print("  Dürüst değerlendirme, modelin ortalamayı söylemekten bile kötü")
    print("  olduğunu gösteriyor — ki gerçek budur.")


def main() -> None:
    ozellik_secimi_sizintisi()
    grup_sizintisi()
    zaman_sizintisi()

    baslik("KONTROL LİSTESİ")
    for madde in [
        "Ölçekleme, doldurma, kodlama, özellik seçimi — hepsi Pipeline içinde mi?",
        "Tekrar eden bir varlık (kişi/mağaza/belge) var mı? -> GrupKKat",
        "Veri zamana bağlı mı? -> ZamanSerisiBolme, gerekiyorsa bosluk=",
        "Sınıflandırmada TabakaliKKat kullanıldı mı?",
        "Hiperparametre seçildiyse nested CV veya ayrı test seti var mı?",
        "Karar eşiği doğrulama setinde mi seçildi?",
        "Her sütun için 'tahmin anında elimde olur muydu?' soruldu mu?",
        "Test seti gerçekten bir kez mi kullanıldı?",
    ]:
        print(f"  [ ] {madde}")
    print("\nAyrıntı: dokuman/04-degerlendirme/veri-sizintisi.md")


if __name__ == "__main__":
    main()
