"""Ders notlarındaki el hesaplarını adım adım yeniden üretir.

Her satırda notun verdiği değer ile hesaplananı yan yana gösterir ve
uyuşup uyuşmadığını işaretler. Uyuşmayanlar `KAYNAK-NOTLARI.md` içinde
belgelenmiştir.

Çalıştır:  python ornekler/02-ders-hesaplari.py
"""

from __future__ import annotations

import numpy as np

from gozetimli.boosting import (adaboost_agirlik_katsayisi,
                                adaboost_agirliklari_guncelle,
                                adaboost_bin_araliklari, adaboost_nihai_skor,
                                gb_baslangic_degeri, sigmoid, xgb_benzerlik,
                                xgb_kazanc, xgb_yaprak_degeri)
from gozetimli.saflik import (bilgi_kazanci, entropi, esik_adaylari, gini,
                              varyans, varyans_azaltimi)
from gozetimli.veri.ders_verileri import (KAYNAK_NOTLARI, burs_regresyon,
                                          kredi_onay, xgb_regresyon,
                                          xgb_siniflandirma)


def baslik(metin: str) -> None:
    print(f"\n{'=' * 72}\n{metin}\n{'=' * 72}")


def satir(ad: str, hesap: float, notta: float, tolerans: float = 0.01) -> bool:
    uyuyor = abs(hesap - notta) <= tolerans
    isaret = "OK " if uyuyor else "FARKLI"
    print(f"  {ad:<42} hesap={hesap:>13.4f}  not={notta:>12.4f}  {isaret}")
    return uyuyor


def main() -> None:
    farklar = []

    # ------------------------------------------------- 13: karar ağacı
    baslik("13 — KARAR AĞACI (kredi onay verisi: 4 Onay, 3 Red)")
    veri = kredi_onay()
    print(veri.tablo())
    print()
    satir("kök entropi", entropi(veri.y), 0.985)
    satir("kök Gini", gini(veri.y), 0.4898)

    mulakat_yuksek = veri.X[:, 1] == 2
    gpa_ustu = veri.X[:, 0] == 1
    satir("bilgi kazancı (Mülakat=Yüksek)",
          bilgi_kazanci(veri.y, maske=mulakat_yuksek), 0.521)
    satir("bilgi kazancı (GPA>3.0)",
          bilgi_kazanci(veri.y, maske=gpa_ustu), 0.021)
    print("  -> Mülakat bölmesi seçilir (daha yüksek kazanç)")

    # --------------------------------------- 13: regresyon ağacı (HATA VAR)
    baslik("13 — KARAR AĞACI REGRESÖRÜ (burs verisi)  [NOTTA HATA]")
    burs = burs_regresyon()
    gpa, hedef = burs.X.ravel(), burs.y
    maske = gpa <= 3.1

    satir("sol varyans (GPA<=3.1)", varyans(hedef[maske]), 11.55)
    satir("sağ varyans (GPA>3.1)", varyans(hedef[~maske]), 44.19)
    agirlikli = (maske.sum() / hedef.size * varyans(hedef[maske])
                 + (~maske).sum() / hedef.size * varyans(hedef[~maske]))
    satir("ağırlıklı varyans (3.1)", agirlikli, 30.19, tolerans=0.02)

    if not satir("kök varyans", varyans(hedef), 124.83):
        farklar.append("karar_agaci_kok_varyans")
    print(f"     ddof=0 -> {varyans(hedef):.4f}   "
          f"ddof=1 -> {float(np.var(hedef, ddof=1)):.4f}   "
          "124.83 hiçbiriyle elde edilemiyor")

    print("\n  Tüm eşikler için varyans azaltımı:")
    azaltimlar = {round(float(e), 2): varyans_azaltimi(hedef, maske=gpa <= e)
                  for e in esik_adaylari(gpa)}
    for esik, azaltim in azaltimlar.items():
        en_iyi = " <- EN YÜKSEK" if azaltim == max(azaltimlar.values()) else ""
        print(f"     eşik {esik:<5} azaltım {azaltim:>8.4f}{en_iyi}")
    if max(azaltimlar, key=azaltimlar.get) != 3.1:
        farklar.append("karar_agaci_en_iyi_esik")
        print("  -> Not 3.1'i en iyi sayıyor; hesap 3.35'i gösteriyor.")

    # ----------------------------------------------------- 16: AdaBoost
    baslik("16 — ADABOOST (7 örnek, 1 yanlış)")
    satir("alfa_1 = ½ln((1-ε)/ε), ε=1/7", adaboost_agirlik_katsayisi(1 / 7), 0.895)

    w = np.full(7, 1 / 7)
    dogru = np.array([True] * 5 + [False] + [True])
    ham = w * np.exp(np.where(dogru, -1.0, 1.0) * 0.895)
    satir("ham ağırlık (doğru)", float(ham[0]), 0.058)
    satir("ham ağırlık (yanlış)", float(ham[5]), 0.354)
    satir("ham toplam", float(ham.sum()), 0.702)

    yeni = adaboost_agirliklari_guncelle(w, dogru, 0.895)
    satir("normalize (doğru)", float(yeni[0]), 0.083)
    satir("normalize (yanlış)", float(yeni[5]), 0.504)

    araliklar = adaboost_bin_araliklari(yeni)
    print(f"\n  Bin aralıkları (yanlış örneğin genişliği = seçilme olasılığı):")
    for i, (alt, ust) in enumerate(araliklar, start=1):
        vurgu = "  <- yanlış sınıflanan" if i == 6 else ""
        print(f"     ID {i}: [{alt:.3f}, {ust:.3f})  genişlik {ust - alt:.3f}{vurgu}")

    satir("alfa_2 (ε=0.249)", adaboost_agirlik_katsayisi(0.249), 0.552)
    satir("F(x) = α₁(+1) + α₂(+1)",
          adaboost_nihai_skor([0.895, 0.552], [1, 1]), 1.447)

    # -------------------------------------------- 18: XGBoost (HATA VAR)
    baslik("18 — XGBOOST SINIFLANDIRMA (5 pozitif / 2 negatif, λ=1)  [NOTTA HATA]")
    xgb = xgb_siniflandirma()
    f0 = gb_baslangic_degeri(xgb.y, gorev="siniflandirma")
    p0 = float(sigmoid(f0))
    satir("F0 = ln(5/2)", f0, 0.916)
    satir("p0 = sigmoid(F0)", p0, 0.714)

    artiklar = xgb.y - p0
    satir("artık (y=0)", float(artiklar[0]), -0.714)
    satir("artık (y=1)", float(artiklar[2]), 0.286)

    s_sol = xgb_benzerlik(artiklar=artiklar[:3], olasiliklar=p0, lam=1)
    s_sag = xgb_benzerlik(artiklar=artiklar[3:], olasiliklar=p0, lam=1)
    s_kok = xgb_benzerlik(artiklar=artiklar, olasiliklar=p0, lam=1)

    satir("S_sol (ID 1,2,3)", s_sol, 0.809)
    if not satir("S_sağ (ID 4-7)", s_sag, 0.702):
        farklar.append("xgboost_sag_benzerlik")
        print(f"     notun kendi ara değerleriyle: 1.144² / 1.816 = "
              f"{1.144**2 / 1.816:.4f}")
    satir("S_kök", s_kok, 0.0, tolerans=1e-9)
    print(f"     S_kök = 0 çünkü Σ artık = {float(artiklar.sum()):.2e}, "
          "'tek grup var' olduğu için değil")

    if not satir("Gain", xgb_kazanc(s_sol, s_sag, s_kok), 1.511):
        print("     (S_sağ hatasından türüyor)")
    satir("yaprak sol", xgb_yaprak_degeri(artiklar[:3], p0, lam=1), -0.709)
    satir("yaprak sağ", xgb_yaprak_degeri(artiklar[3:], p0, lam=1), 0.631,
          tolerans=0.002)

    f1 = f0 + 0.1 * xgb_yaprak_degeri(artiklar[:3], p0, lam=1)
    satir("F1 (ID1), η=0.1", f1, 0.845)
    satir("p1 (ID1)", float(sigmoid(f1)), 0.699)

    # ----------------------------------------------- 18: XGBoost regresyon
    baslik("18 — XGBOOST REGRESYON (λ=0)")
    reg = xgb_regresyon()
    satir("F0 (ortalama)", gb_baslangic_degeri(reg.y), 5000)
    r = reg.y - 5000.0
    satir("Gain (maaş<=60)",
          xgb_kazanc(xgb_benzerlik(artiklar=r[:4], lam=0),
                     xgb_benzerlik(artiklar=r[4:], lam=0),
                     xgb_benzerlik(artiklar=r, lam=0)), 21_000_000, tolerans=1)
    satir("yaprak sol", xgb_yaprak_degeri(r[:4], lam=0), -1500, tolerans=1)
    satir("yaprak sağ", xgb_yaprak_degeri(r[4:], lam=0), 2000, tolerans=1)

    # ------------------------------------------------------------- özet
    baslik("ÖZET")
    if farklar:
        print(f"{len(set(farklar))} tutarsızlık tespit edildi:\n")
        for anahtar in dict.fromkeys(farklar):
            print(f"  * {anahtar}")
            print(f"    {KAYNAK_NOTLARI[anahtar][:150]}...\n")
        print("Tam açıklama: dokuman/KAYNAK-NOTLARI.md")
        print("Doğrulayan testler: pytest -m ders -v")
    else:
        print("Tüm hesaplar uyuştu.")


if __name__ == "__main__":
    main()
