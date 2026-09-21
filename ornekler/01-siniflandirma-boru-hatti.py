"""Uçtan uca sınıflandırma değerlendirmesi, doğru kurulmuş bir boru hattı.

Gösterilen adımlar:
    1. Üçlü ayrım (eğitim / doğrulama / test)
    2. Ön işlemenin Pipeline içine konması
    3. Tabakalı çapraz doğrulama, kat kat skorlar ve aşırı öğrenme farkı
    4. Karar eşiğinin DOĞRULAMA setinde seçilmesi
    5. Test setinin yalnızca en sonda, bir kez kullanılması

Çalıştır:  python ornekler/01-siniflandirma-boru-hatti.py
Gereksinim: pip install -e ".[ornek]"
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gozetimli.dogrulama.bolme import TabakaliKKat, egitim_test_bol
from gozetimli.dogrulama.capraz import capraz_dogrula
from gozetimli.metrikler.siniflandirma import (esik_tara, f1_skoru,
                                               karmasiklik_matrisi, kesinlik,
                                               ortalama_kesinlik, roc_auc,
                                               siniflandirma_raporu,
                                               duyarlilik)

TOHUM = 42


def baslik(metin: str) -> None:
    print(f"\n{'=' * 68}\n{metin}\n{'=' * 68}")


def main() -> None:
    # ---------------------------------------------------------------- veri
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=6, n_redundant=4,
        weights=[0.88, 0.12],          # dengesiz: %12 pozitif
        flip_y=0.03, random_state=TOHUM)

    baslik("1. VERİ")
    print(f"Örnek sayısı : {len(y)}")
    print(f"Özellik      : {X.shape[1]}")
    print(f"Pozitif oranı: {y.mean():.3f}  -> dengesiz, accuracy yanıltır")

    # ------------------------------------------------- üçlü ayrım (60/20/20)
    X_gecici, X_test, y_gecici, y_test = egitim_test_bol(
        X, y, test_orani=0.2, tohum=TOHUM, tabaka=y)
    X_egitim, X_dog, y_egitim, y_dog = egitim_test_bol(
        X_gecici, y_gecici, test_orani=0.25, tohum=TOHUM, tabaka=y_gecici)

    baslik("2. ÜÇLÜ AYRIM")
    for ad, hedef in (("Eğitim", y_egitim), ("Doğrulama", y_dog), ("Test", y_test)):
        print(f"{ad:<10} {len(hedef):>5} örnek, pozitif oranı {hedef.mean():.3f}")
    print("\nTest seti buradan sonra 5. adıma kadar HİÇ kullanılmayacak.")

    # ------------------------------------------------------------ boru hattı
    # Ölçekleyici Pipeline'ın İÇİNDE: her katta yalnızca o katın eğitim
    # verisinden öğrenilir, test katının ortalaması sızmaz.
    boru = Pipeline([
        ("olcek", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    baslik("3. ÇAPRAZ DOĞRULAMA (eğitim seti üzerinde)")
    sonuc = capraz_dogrula(
        boru, X_egitim, y_egitim,
        bolucu=TabakaliKKat(5, karistir=True, tohum=TOHUM),
        metrikler={"f1": f1_skoru, "auc": roc_auc, "ap": ortalama_kesinlik},
        olasilik_metrikleri=("auc", "ap"),
        egitim_skoru=True)

    ozet = sonuc.ozet()
    print(f"{'metrik':<8} {'ortalama':>9} {'std':>8} {'min':>8} {'max':>8} "
          f"{'aşırı öğrenme':>15}")
    for ad, d in ozet.items():
        print(f"{ad:<8} {d['ortalama']:>9.4f} {d['std']:>8.4f} "
              f"{d['min']:>8.4f} {d['max']:>8.4f} "
              f"{d.get('asiri_ogrenme_farki', 0):>15.4f}")
    print(f"\nKat kat F1: {ozet['f1']['katlar']}")
    print("std küçükse model bölmeye karşı kararlı demektir.")

    # --------------------------------------------- eşik seçimi (doğrulama!)
    boru.fit(X_egitim, y_egitim)
    olasilik_dog = boru.predict_proba(X_dog)[:, 1]

    baslik("4. EŞİK SEÇİMİ (doğrulama seti üzerinde)")
    varsayilan = (olasilik_dog >= 0.5).astype(int)
    print(f"Varsayılan eşik 0.50 -> F1 = {f1_skoru(y_dog, varsayilan):.4f}, "
          f"kesinlik = {kesinlik(y_dog, varsayilan):.4f}, "
          f"duyarlilik = {duyarlilik(y_dog, varsayilan):.4f}")

    esik_f1, en_iyi_f1, _, _ = esik_tara(y_dog, olasilik_dog, metrik="f1")
    print(f"F1'i maksimize eden eşik: {esik_f1:.4f} -> F1 = {en_iyi_f1:.4f}")

    # İş kısıtı: "pozitiflerin en az %90'ını yakala, o kısıt altında
    # yanlış alarmı en aza indir"
    esik_kisit, kesinlik_kisit, _, _ = esik_tara(
        y_dog, olasilik_dog, metrik="kesinlik", en_az_duyarlilik=0.90)
    tahmin_kisit = (olasilik_dog >= esik_kisit).astype(int)
    print(f"\nKısıt: duyarlilik >= 0.90")
    print(f"Seçilen eşik: {esik_kisit:.4f} -> "
          f"kesinlik = {kesinlik_kisit:.4f}, "
          f"duyarlilik = {duyarlilik(y_dog, tahmin_kisit):.4f}")

    # ------------------------------------------------ nihai rapor (test seti)
    baslik("5. NİHAİ RAPOR (test seti: ilk ve tek kullanım)")
    boru.fit(X_gecici, y_gecici)        # eğitim + doğrulama birlikte
    olasilik_test = boru.predict_proba(X_test)[:, 1]
    tahmin_test = (olasilik_test >= esik_f1).astype(int)

    print(f"Kullanılan eşik: {esik_f1:.4f} (doğrulama setinde seçildi)\n")
    print(siniflandirma_raporu(y_test, tahmin_test, metin=True))

    matris = karmasiklik_matrisi(y_test, tahmin_test)
    print(f"\nKarmaşıklık matrisi (satır=gerçek, sütun=tahmin):")
    print(f"        tahmin 0  tahmin 1")
    print(f"gerçek 0 {matris[0, 0]:>8} {matris[0, 1]:>9}")
    print(f"gerçek 1 {matris[1, 0]:>8} {matris[1, 1]:>9}")

    print(f"\nEşikten bağımsız:")
    print(f"  ROC-AUC          : {roc_auc(y_test, olasilik_test):.4f}")
    print(f"  Ortalama kesinlik: {ortalama_kesinlik(y_test, olasilik_test):.4f} "
          f"(taban çizgisi {y_test.mean():.4f})")

    baslik("ÖZET")
    print("Bu kurulumda sızıntı yok çünkü:")
    print("  - Ölçekleyici Pipeline içinde, her katta yeniden öğreniliyor")
    print("  - Eşik doğrulama setinde seçildi, test setinde değil")
    print("  - Test seti yalnızca en sonda, bir kez kullanıldı")
    print("  - Dengesiz veri için tabakalı bölme ve F1/AP kullanıldı")


if __name__ == "__main__":
    main()
