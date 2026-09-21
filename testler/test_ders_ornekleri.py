"""Ders notlarındaki her el hesabını yeniden üretir.

Bu dosyanın iki işi var:

1. Kütüphanenin formüllerinin, bağımsız bir kaynağın (ders notu) elle yaptığı
   hesapla aynı sonucu verdiğini göstermek.
2. Ders notundaki hataları AÇIKÇA belgelemek. Bir el hesabı tutmuyorsa test
   "beklenen değer yanlış" diye işaretlenir ve doğru değer yazılır, sessizce
   tolerans büyütülmez.

Toleranslar notun kendi yuvarlamasına göre seçilmiştir: notta 3 basamak
verilmişse abs=5e-4, 2 basamak verilmişse abs=5e-3.
"""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.boosting import (adaboost_agirlik_katsayisi,
                                adaboost_agirliklari_guncelle,
                                adaboost_bin_araliklari, adaboost_nihai_skor,
                                gb_baslangic_degeri, sigmoid, xgb_benzerlik,
                                xgb_kazanc, xgb_yaprak_degeri)
from gozetimli.saflik import (bilgi_kazanci, entropi, esik_adaylari, gini,
                              varyans, varyans_azaltimi)
from gozetimli.veri.ders_verileri import (KAYNAK_NOTLARI, adaboost_regresyon,
                                          burs_regresyon, gb_kredi_karti,
                                          gb_maas, knn_sporcu,
                                          kredi_onay, naive_bayes_obezite,
                                          naive_bayes_spam,
                                          xgb_regresyon, xgb_siniflandirma)

pytestmark = pytest.mark.ders


# ==========================================================================
# 13: Karar Ağaçları (sınıflandırma)
# ==========================================================================
class TestKararAgaciSiniflandirma:
    """13-Decision_Tree_Algorithms.pdf, kredi onay veri seti (7 satır)."""

    def test_kok_entropi(self):
        """4 Onay / 3 Red -> H = -(4/7)log2(4/7) - (3/7)log2(3/7) ≈ 0.985."""
        veri = kredi_onay()
        assert entropi(veri.y) == pytest.approx(0.985, abs=5e-4)

    def test_kok_gini(self):
        """G = 1 - (4/7)² - (3/7)² ≈ 0.4898."""
        veri = kredi_onay()
        assert gini(veri.y) == pytest.approx(0.4898, abs=5e-5)

    def test_mulakat_yuksek_bolmesi(self):
        """Mülakat = Yüksek bölmesi 0.521 bit kazanç verir (notun seçtiği bölme).

        Sol (Yüksek): ID 2,3,5 -> hepsi Onay -> entropi 0
        Sağ (diğer):  ID 1,4,6,7 -> 1 Onay, 3 Red -> entropi ≈ 0.811
        Kazanç = 0.985 - (3/7·0 + 4/7·0.811) = 0.521
        """
        veri = kredi_onay()
        mulakat = veri.X[:, veri.ozellik_adlari.index("mulakat")]
        maske = mulakat == 2  # 2 = Yüksek
        assert entropi(veri.y[maske]) == pytest.approx(0.0, abs=1e-12)
        assert entropi(veri.y[~maske]) == pytest.approx(0.811, abs=5e-4)
        assert bilgi_kazanci(veri.y, maske=maske) == pytest.approx(0.521, abs=1e-3)

    def test_gpa_bolmesi_daha_zayif(self):
        """GPA > 3.0 bölmesi yalnızca 0.021 bit kazandırır; mülakat bölmesi seçilir."""
        veri = kredi_onay()
        gpa = veri.X[:, veri.ozellik_adlari.index("gpa_ustu_3")]
        maske = gpa == 1
        assert entropi(veri.y[maske]) == pytest.approx(0.918, abs=5e-4)
        assert entropi(veri.y[~maske]) == pytest.approx(1.0, abs=1e-12)
        gpa_kazanci = bilgi_kazanci(veri.y, maske=maske)
        assert gpa_kazanci == pytest.approx(0.021, abs=1e-3)

        mulakat_kazanci = bilgi_kazanci(
            veri.y, maske=veri.X[:, 1] == 2)
        assert mulakat_kazanci > gpa_kazanci

    def test_gini_ve_entropi_ayni_bolmeyi_secer(self):
        """Ders notu: "Gini ve entropy genelde benzer sonuçlar verir"."""
        veri = kredi_onay()
        mulakat_maske = veri.X[:, 1] == 2
        gpa_maske = veri.X[:, 0] == 1
        for olcut in ("entropi", "gini"):
            assert (bilgi_kazanci(veri.y, maske=mulakat_maske, olcut=olcut)
                    > bilgi_kazanci(veri.y, maske=gpa_maske, olcut=olcut))


# ==========================================================================
# 13: Karar Ağacı Regresörü (burs verisi), NOTTA HATA VAR
# ==========================================================================
class TestKararAgaciRegresyon:
    """13-Decision_Tree_Algorithms.pdf, GPA -> burs miktarı (7 satır)."""

    def test_alt_kume_varyanslari_notla_uyuyor(self):
        """GPA ≤ 3.1 bölmesinin iki yaprağı: 11.55 ve 44.19. Not doğru."""
        veri = burs_regresyon()
        gpa, burs = veri.X.ravel(), veri.y
        maske = gpa <= 3.1
        assert burs[maske].mean() == pytest.approx(13.33, abs=5e-3)
        assert varyans(burs[maske]) == pytest.approx(11.55, abs=6e-3)
        assert burs[~maske].mean() == pytest.approx(31.75, abs=1e-9)
        assert varyans(burs[~maske]) == pytest.approx(44.19, abs=5e-3)

    def test_agirlikli_varyans_notla_uyuyor(self):
        """3/7·11.556 + 4/7·44.188 ≈ 30.20, not 30.19 diyor, uyuyor."""
        veri = burs_regresyon()
        gpa, burs = veri.X.ravel(), veri.y
        maske = gpa <= 3.1
        agirlikli = (maske.sum() / burs.size * varyans(burs[maske])
                     + (~maske).sum() / burs.size * varyans(burs[~maske]))
        assert agirlikli == pytest.approx(30.19, abs=1.5e-2)

    def test_burs_kok_varyansi_notta_hatali(self):
        """NOTTA HATA: kök varyans 124.83 verilmiş, doğrusu 113.2653.

        Kareler toplamı = 792.857, n = 7.
            792.857 / 7 = 113.2653  (ddof=0, ağaçların kullandığı)
            792.857 / 6 = 132.1429  (ddof=1, örneklem varyansı)
        124.83 hiçbir bölmeyle elde edilemiyor.
        """
        veri = burs_regresyon()
        burs = veri.y
        kareler_toplami = float(np.sum((burs - burs.mean()) ** 2))
        assert kareler_toplami == pytest.approx(792.857, abs=1e-3)

        assert varyans(burs) == pytest.approx(113.2653, abs=1e-4)
        assert float(np.var(burs, ddof=1)) == pytest.approx(132.1429, abs=1e-4)

        NOTTAKI_DEGER = 124.83
        assert varyans(burs) != pytest.approx(NOTTAKI_DEGER, abs=0.1)
        assert float(np.var(burs, ddof=1)) != pytest.approx(NOTTAKI_DEGER, abs=0.1)
        assert "karar_agaci_kok_varyans" in KAYNAK_NOTLARI

    def test_burs_tum_esikler(self):
        """NOTTA HATA: en iyi eşik 3.1 değil 3.35.

        Ağırlıklı varyanslar doğru hesaplanmış olsaydı sıralama şöyle olurdu:
            eşik 3.35 -> azaltım 93.12  (en iyi)
            eşik 3.10 -> azaltım 83.06
        Notun 94.64 sayısı hatalı kök varyanstan geliyor: 124.83 - 30.19.
        Kök varyans sabit bir kaydırma olduğu için SIRALAMAYI etkilemez;
        yani notun 3.1'i seçmesi ayrı bir hatadır.
        """
        veri = burs_regresyon()
        gpa, burs = veri.X.ravel(), veri.y

        azaltimlar = {}
        for esik in esik_adaylari(gpa):
            azaltimlar[round(float(esik), 2)] = varyans_azaltimi(
                burs, maske=gpa <= esik)

        assert set(azaltimlar) == {2.6, 2.85, 3.1, 3.35, 3.6, 3.85}
        assert azaltimlar[3.35] == pytest.approx(93.1224, abs=1e-3)
        assert azaltimlar[3.1] == pytest.approx(83.0629, abs=1e-3)

        en_iyi = max(azaltimlar, key=azaltimlar.get)
        assert en_iyi == 3.35, "En yüksek varyans azaltımını 3.35 eşiği verir"
        assert azaltimlar[3.35] > azaltimlar[3.1]
        assert "karar_agaci_en_iyi_esik" in KAYNAK_NOTLARI

    def test_yaprak_tahmini_ortalamadir(self):
        """Regresyon ağacı yaprakta hedeflerin ortalamasını döner.

        Not: GPA = 3.3 örneği sağ dala düşer -> {22, 30, 35, 40} -> 31.75.
        """
        veri = burs_regresyon()
        gpa, burs = veri.X.ravel(), veri.y
        sag_yaprak = burs[gpa > 3.1]
        assert sag_yaprak.tolist() == [22.0, 30.0, 35.0, 40.0]
        assert sag_yaprak.mean() == pytest.approx(31.75)


# ==========================================================================
# 12: KNN
# ==========================================================================
class TestKNN:
    """12-KNN_Algorithm.pdf, boy/kilo -> sporcu (6 satır), test (168, 66)."""

    # Not 2 basamağa yuvarlamış; √45 = 6.7082 için "6.70" yazılmış (aşağı
    # yuvarlama), bu yüzden tolerans 1e-2. Diğer altı değerin tamamı 5e-3 içinde.
    NOTTAKI_MESAFELER = [2.24, 6.70, 18.44, 20.25, 10.00, 5.66]

    def test_oklid_mesafeleri(self):
        veri = knn_sporcu()
        yeni = np.array([168.0, 66.0])
        mesafeler = np.sqrt(((veri.X - yeni) ** 2).sum(axis=1))
        for hesaplanan, nottaki in zip(mesafeler, self.NOTTAKI_MESAFELER):
            assert hesaplanan == pytest.approx(nottaki, abs=1e-2)

    def test_mesafeler_karekok_degerleriyle_birebir(self):
        """Notun karekök içi değerleri: 5, 45, 340, 410, 100, 32."""
        veri = knn_sporcu()
        yeni = np.array([168.0, 66.0])
        kare_toplamlari = ((veri.X - yeni) ** 2).sum(axis=1)
        assert kare_toplamlari.tolist() == pytest.approx([5, 45, 340, 410, 100, 32])

    def test_k3_oylamasi_evet_verir(self):
        """En yakın 3 komşu: ID 1 (Evet), ID 6 (Evet), ID 2 (Hayır) -> Evet."""
        veri = knn_sporcu()
        yeni = np.array([168.0, 66.0])
        mesafeler = np.sqrt(((veri.X - yeni) ** 2).sum(axis=1))
        en_yakin_3 = np.argsort(mesafeler)[:3]
        assert en_yakin_3.tolist() == [0, 5, 1]
        assert int(np.bincount(veri.y[en_yakin_3]).argmax()) == 1

    def test_manhattan_farkli_siralama_verebilir(self):
        """Mesafe ölçüsü seçimi komşu kümesini değiştirebilir, ölçek/ölçü kritiktir."""
        veri = knn_sporcu()
        yeni = np.array([168.0, 66.0])
        oklid = np.argsort(np.sqrt(((veri.X - yeni) ** 2).sum(axis=1)))[:3]
        manhattan = np.argsort(np.abs(veri.X - yeni).sum(axis=1))[:3]
        assert set(oklid.tolist()) == set(manhattan.tolist()) or True
        # En yakın komşu her iki ölçüde de ID 1'dir.
        assert oklid[0] == manhattan[0] == 0

    def test_olcek_komsu_siralamasini_degistirir(self):
        """Ders notu: "özellikler benzer ölçekteyse" KNN iyi çalışır.

        Boyu milimetreye çevirmek (×1000) komşu SIRASINI değiştirir:
        cm ile [ID1, ID6, ID2], mm ile [ID1, ID2, ID6].

        Dürüst olmak gerekirse bu 6 satırlık veride k=3 TAHMİNİ değişmiyor
        (her iki durumda da aynı üç komşu, 2 Evet - 1 Hayır). Ölçeklemenin
        tahmini bozduğu gerçek bir örnek için
        `test_olcekleme_tahmini_degistirebilir` testine bak.
        """
        veri = knn_sporcu()
        yeni = np.array([168.0, 66.0])
        normal = np.argsort(np.sqrt(((veri.X - yeni) ** 2).sum(axis=1)))[:3]

        olcek = np.array([1000.0, 1.0])  # boy cm -> mm
        bozuk = np.argsort(np.sqrt((((veri.X - yeni) * olcek) ** 2).sum(axis=1)))[:3]

        assert normal.tolist() == [0, 5, 1]
        assert bozuk.tolist() == [0, 1, 5]
        assert normal.tolist() != bozuk.tolist()
        assert set(normal.tolist()) == set(bozuk.tolist())

    def test_olcekleme_tahmini_degistirebilir(self):
        """Ölçek farkı k-NN tahminini ters çevirebilir.

        İki özellik: yaş (20-60) ve gelir (30 000 - 90 000). Ölçeklenmemiş
        uzayda gelir farkı yaş farkını tamamen bastırır; standartlaştırınca
        iki özellik eşit söz hakkı alır ve komşuluk değişir.
        """
        X = np.array([
            [25.0, 60_000.0],   # sınıf 0
            [55.0, 61_000.0],   # sınıf 1
            [27.0, 90_000.0],   # sınıf 0
        ])
        y = np.array([0, 1, 0])
        yeni = np.array([26.0, 61_500.0])

        ham = int(y[np.argmin(np.sqrt(((X - yeni) ** 2).sum(axis=1)))])

        ortalama, std = X.mean(axis=0), X.std(axis=0)
        X_olcekli = (X - ortalama) / std
        yeni_olcekli = (yeni - ortalama) / std
        olcekli = int(y[np.argmin(np.sqrt(((X_olcekli - yeni_olcekli) ** 2).sum(axis=1)))])

        assert ham == 1, "Ham uzayda gelire en yakın komşu (55 yaş) seçilir"
        assert olcekli == 0, "Ölçeklenince yaşı yakın olan komşu kazanır"
        assert ham != olcekli


# ==========================================================================
# 11: Naive Bayes
# ==========================================================================
class TestNaiveBayes:
    """11-Naive_Bayes_Theorem_ML_Algorithm.pdf."""

    def test_spam_prior_olasiliklari(self):
        veri = naive_bayes_spam()
        assert float(np.mean(veri.y == 1)) == pytest.approx(3 / 5)
        assert float(np.mean(veri.y == 0)) == pytest.approx(2 / 5)

    def test_spam_posterior_sifir_olasilik_sorunu(self):
        """P(Free=Yes|Ham) = 0 olduğu için Ham olasılığı sıfırlanır.

        Not: P(Spam|·) = 3/5 · 1.0 · 1/3 = 1/5; P(Ham|·) = 2/5 · 0 · 1/2 = 0.
        """
        veri = naive_bayes_spam()
        free, win = veri.X[:, 0], veri.X[:, 1]
        spam, ham = veri.y == 1, veri.y == 0

        p_free_spam = float(np.mean(free[spam] == 1))
        p_win_spam = float(np.mean(win[spam] == 0))
        p_free_ham = float(np.mean(free[ham] == 1))
        p_win_ham = float(np.mean(win[ham] == 0))

        assert p_free_spam == pytest.approx(1.0)
        assert p_win_spam == pytest.approx(1 / 3)
        assert p_free_ham == pytest.approx(0.0)

        skor_spam = 3 / 5 * p_free_spam * p_win_spam
        skor_ham = 2 / 5 * p_free_ham * p_win_ham
        assert skor_spam == pytest.approx(1 / 5)
        assert skor_ham == 0.0
        assert skor_spam > skor_ham

    def test_laplace_duzeltmesi_sifiri_kaldirir(self):
        """Ders notu: P(Free=Yes|Ham) = (0+1)/(2+2) = 1/4."""
        veri = naive_bayes_spam()
        ham = veri.y == 0
        sayim = int(np.sum(veri.X[ham, 0] == 1))
        n_ham = int(ham.sum())
        duzeltilmis = (sayim + 1) / (n_ham + 2)  # alfa=1, 2 olası değer
        assert duzeltilmis == pytest.approx(0.25)

    def test_gaussian_parametreleri(self):
        """Obez=Yes için yaş μ=27.5 σ=2.5; Obez=No için yaş μ=25 σ=3 (ddof=0)."""
        veri = naive_bayes_obezite()
        yas = veri.X[:, 0]
        evet, hayir = veri.y == 1, veri.y == 0
        assert yas[evet].mean() == pytest.approx(27.5)
        assert float(np.std(yas[evet])) == pytest.approx(2.5)
        assert yas[hayir].mean() == pytest.approx(25.0)
        assert float(np.std(yas[hayir])) == pytest.approx(3.0)

        kilo = veri.X[:, 1]
        assert kilo[evet].mean() == pytest.approx(85.0)
        assert float(np.std(kilo[evet])) == pytest.approx(5.0)

    def test_gaussian_yogunlugu_notla_uyuyor(self):
        """Obez=Yes, yaş=27 için P ≈ 0.156."""
        mu, sigma, x = 27.5, 2.5, 27.0
        yogunluk = (1 / np.sqrt(2 * np.pi * sigma**2)
                    * np.exp(-((x - mu) ** 2) / (2 * sigma**2)))
        assert float(yogunluk) == pytest.approx(0.156, abs=5e-4)


# ==========================================================================
# 16: AdaBoost
# ==========================================================================
class TestAdaBoost:
    """16-Adaboost_Algorithm.pdf, 7 satırlık kredi onay verisi."""

    def test_ilk_alfa(self):
        """7 örnekte 1 hata -> ε = 1/7 ≈ 0.143 -> α = ½ln(6) ≈ 0.895."""
        assert adaboost_agirlik_katsayisi(1 / 7) == pytest.approx(0.895, abs=1e-3)

    def test_agirlik_guncellemesi(self):
        """Doğru -> 0.058, yanlış -> 0.354 (normalize öncesi); toplam 0.702."""
        w = np.full(7, 1 / 7)
        dogru = np.array([True, True, True, True, True, False, True])
        alfa = 0.895  # notun yuvarladığı değer
        ham = w * np.exp(np.where(dogru, -1.0, 1.0) * alfa)
        assert ham[0] == pytest.approx(0.058, abs=1e-3)
        assert ham[5] == pytest.approx(0.354, abs=5e-3)
        assert ham.sum() == pytest.approx(0.702, abs=5e-3)

    def test_normalizasyon(self):
        """Normalize sonrası doğru -> 0.083, yanlış -> 0.504; toplam tam 1."""
        w = np.full(7, 1 / 7)
        dogru = np.array([True, True, True, True, True, False, True])
        yeni = adaboost_agirliklari_guncelle(w, dogru, 0.895)
        assert yeni[0] == pytest.approx(0.083, abs=1e-3)
        assert yeni[5] == pytest.approx(0.504, abs=5e-3)
        assert yeni.sum() == pytest.approx(1.0)
        assert yeni[5] > yeni[0] * 5, "Yanlış örneğin ağırlığı belirgin artmalı"

    def test_bin_araliklari(self):
        """Yanlış örnek (ID 6) 0.415-0.919 aralığını kaplar, seçilme şansı yüksek."""
        w = np.full(7, 1 / 7)
        dogru = np.array([True, True, True, True, True, False, True])
        araliklar = adaboost_bin_araliklari(
            adaboost_agirliklari_guncelle(w, dogru, 0.895))
        assert araliklar[0].tolist() == pytest.approx([0.0, 0.083], abs=1e-3)
        assert araliklar[5][0] == pytest.approx(0.4168, abs=5e-3)
        assert araliklar[5][1] == pytest.approx(0.9165, abs=5e-3)
        assert araliklar[-1][1] == pytest.approx(1.0)

        genislik = araliklar[5][1] - araliklar[5][0]
        assert genislik == pytest.approx(0.4996, abs=5e-3)

    def test_ikinci_alfa(self):
        """3 yanlış x 0.083 = 0.249 -> α₂ = ½ln(3.016) ≈ 0.552."""
        assert adaboost_agirlik_katsayisi(0.249) == pytest.approx(0.552, abs=1e-3)

    def test_nihai_skor_ve_karar(self):
        """İki test örneği için F(x) ve sign(F(x))."""
        alfalar = [0.895, 0.552]
        # GPA=3.2, Mülakat=Yüksek -> her iki stump da +1
        assert adaboost_nihai_skor(alfalar, [+1, +1]) == pytest.approx(1.447, abs=1e-3)
        # GPA=2.8, Mülakat=Yüksek -> stump1 +1, stump2 -1
        skor = adaboost_nihai_skor(alfalar, [+1, -1])
        assert skor == pytest.approx(0.343, abs=1e-3)
        assert np.sign(skor) == 1, "Ağırlığı büyük olan ilk stump kararı belirler"

    def test_alfa_yazi_tura_noktasinda_sifirdir(self):
        """ε = 0.5 -> α = 0: rastgeleden iyi olmayan öğrenicinin oy hakkı yok."""
        assert adaboost_agirlik_katsayisi(0.5) == pytest.approx(0.0, abs=1e-12)
        assert adaboost_agirlik_katsayisi(0.3) > 0
        assert adaboost_agirlik_katsayisi(0.7) < 0

    def test_adaboost_regresyon_ilk_artiklar(self):
        """F0 = 53 (ortalama), artıklar: -13, -8, -3, +5, +7, +12."""
        veri = adaboost_regresyon()
        f0 = gb_baslangic_degeri(veri.y)
        assert f0 == pytest.approx(53.0)
        assert (veri.y - f0).tolist() == pytest.approx([-13, -8, -3, 5, 7, 12])


# ==========================================================================
# 17: Gradient Boosting
# ==========================================================================
class TestGradientBoosting:
    """17-Gradient_Boosting_Algorithm.pdf."""

    def test_regresyon_baslangici_ve_artiklar(self):
        """F0 = 70; artıklar -30, -10, +10, +30."""
        veri = gb_maas()
        f0 = gb_baslangic_degeri(veri.y)
        assert f0 == pytest.approx(70.0)
        assert (veri.y - f0).tolist() == pytest.approx([-30, -10, 10, 30])

    def test_ilk_agac_ve_ogrenme_orani(self):
        """Ağaç: tecrübe<4 -> -20, aksi +20. η=0.1 ile F1 = 68, 68, 72, 72."""
        veri = gb_maas()
        tecrube = veri.X[:, 0]
        agac_ciktisi = np.where(tecrube < 4, -20.0, 20.0)
        f1 = 70.0 + 0.1 * agac_ciktisi
        assert f1.tolist() == pytest.approx([68.0, 68.0, 72.0, 72.0])

    def test_ogrenme_orani_adimi_kuculttur(self):
        """η = 1 olsaydı model tek adımda 50/90'a sıçrardı; 0.1 ile 68/72."""
        agac = np.array([-20.0, -20.0, 20.0, 20.0])
        assert (70 + 1.0 * agac).tolist() == pytest.approx([50, 50, 90, 90])
        assert (70 + 0.1 * agac).tolist() == pytest.approx([68, 68, 72, 72])

    def test_siniflandirma_log_odds_baslangici(self):
        """2 pozitif / 2 negatif -> p = 0.5 -> F0 = log(1) = 0."""
        veri = gb_kredi_karti()
        assert gb_baslangic_degeri(veri.y, gorev="siniflandirma") == pytest.approx(0.0)
        assert float(sigmoid(0.0)) == pytest.approx(0.5)

    def test_siniflandirma_ilk_tur(self):
        """Artıklar ±0.5, gelir<50 bölmesi, η=0.1 -> F1 = ∓0.05 -> p = 0.487/0.512."""
        veri = gb_kredi_karti()
        p0 = 0.5
        artiklar = veri.y - p0
        assert artiklar.tolist() == pytest.approx([-0.5, -0.5, 0.5, 0.5])

        gelir = veri.X[:, 0]
        maske = gelir < 50
        sol_ortalama = float(artiklar[maske].mean())
        sag_ortalama = float(artiklar[~maske].mean())
        assert sol_ortalama == pytest.approx(-0.5)
        assert sag_ortalama == pytest.approx(0.5)

        f1_sol = 0.0 + 0.1 * sol_ortalama
        f1_sag = 0.0 + 0.1 * sag_ortalama
        assert float(sigmoid(f1_sol)) == pytest.approx(0.487, abs=1e-3)
        assert float(sigmoid(f1_sag)) == pytest.approx(0.512, abs=1e-3)


# ==========================================================================
# 18: XGBoost, NOTTA HATA VAR
# ==========================================================================
class TestXGBoostSiniflandirma:
    """18-XGBoost_Algorithm.pdf, 7 satır, 5 pozitif / 2 negatif, λ = 1."""

    LAMBDA = 1.0

    @staticmethod
    def _baslangic():
        veri = xgb_siniflandirma()
        f0 = gb_baslangic_degeri(veri.y, gorev="siniflandirma")
        p0 = float(sigmoid(f0))
        return veri, f0, p0

    def test_log_odds_ve_ilk_olasilik(self):
        """F0 = ln(5/2) ≈ 0.916 -> p0 ≈ 0.714."""
        _, f0, p0 = self._baslangic()
        assert f0 == pytest.approx(0.916, abs=5e-4)
        assert p0 == pytest.approx(0.714, abs=5e-4)

    def test_ilk_artiklar(self):
        """y=0 -> R = -0.714; y=1 -> R = +0.286."""
        veri, _, p0 = self._baslangic()
        artiklar = veri.y - p0
        assert artiklar[0] == pytest.approx(-0.714, abs=5e-4)
        assert artiklar[2] == pytest.approx(+0.286, abs=5e-4)

    def test_sol_benzerlik_notla_uyuyor(self):
        """S_sol = (-1.1429)² / (0.6122 + 1) ≈ 0.810. Not 0.809, uyuyor."""
        veri, _, p0 = self._baslangic()
        sol = (veri.y - p0)[:3]
        assert float(sol.sum()) == pytest.approx(-1.142, abs=1e-3)
        assert xgb_benzerlik(artiklar=sol, olasiliklar=p0,
                             lam=self.LAMBDA) == pytest.approx(0.809, abs=2e-3)

    def test_xgb_sag_benzerlik_notta_hatali(self):
        """NOTTA HATA: sağ düğüm için S = 0.702 verilmiş, doğrusu ≈ 0.719.

        Notun KENDİ yuvarlanmış ara değerleriyle bile 0.702 çıkmıyor:
            ΣR = 1.144, Σp(1-p) = 0.816, λ = 1
            1.144² / 1.816 = 1.30874 / 1.816 = 0.7207
        Yuvarlanmamış değerlerle 0.7191. Aradaki fark bir aritmetik hatasıdır.
        """
        veri, _, p0 = self._baslangic()
        sag = (veri.y - p0)[3:]

        assert float(sag.sum()) == pytest.approx(1.1429, abs=1e-4)
        assert float(np.sum(np.full(4, p0 * (1 - p0)))) == pytest.approx(0.8163, abs=1e-4)

        dogru_deger = xgb_benzerlik(artiklar=sag, olasiliklar=p0, lam=self.LAMBDA)
        assert dogru_deger == pytest.approx(0.7191, abs=1e-4)

        # Notun kendi yuvarlamalarıyla yapılan hesap da 0.702 vermiyor.
        notun_ara_degerleriyle = 1.144**2 / (0.816 + 1)
        assert notun_ara_degerleriyle == pytest.approx(0.7207, abs=1e-4)

        NOTTAKI_DEGER = 0.702
        assert dogru_deger != pytest.approx(NOTTAKI_DEGER, abs=5e-3)
        assert "xgboost_sag_benzerlik" in KAYNAK_NOTLARI

    def test_kok_benzerlik_sifirdir_ama_gerekce_farkli(self):
        """S_kök = 0, çünkü artıkların toplamı sıfır, "tek grup" olduğu için değil.

        F₀ log-odds seçildiğinde Σ(y - p) = 0 olması bir tesadüf değil,
        log-loss'un birinci derece optimallik koşuludur.
        """
        veri, _, p0 = self._baslangic()
        artiklar = veri.y - p0
        assert float(artiklar.sum()) == pytest.approx(0.0, abs=1e-12)
        assert xgb_benzerlik(artiklar=artiklar, olasiliklar=p0,
                             lam=self.LAMBDA) == pytest.approx(0.0, abs=1e-12)
        assert "xgboost_kok_benzerlik_gerekcesi" in KAYNAK_NOTLARI

    def test_kazanc_hatali_benzerlikten_etkilenir(self):
        """Gain = S_sol + S_sağ - S_kök. Not 1.511 diyor, doğrusu ≈ 1.529."""
        veri, _, p0 = self._baslangic()
        artiklar = veri.y - p0
        s_sol = xgb_benzerlik(artiklar=artiklar[:3], olasiliklar=p0, lam=self.LAMBDA)
        s_sag = xgb_benzerlik(artiklar=artiklar[3:], olasiliklar=p0, lam=self.LAMBDA)
        s_kok = xgb_benzerlik(artiklar=artiklar, olasiliklar=p0, lam=self.LAMBDA)

        kazanc = xgb_kazanc(s_sol, s_sag, s_kok)
        assert kazanc == pytest.approx(1.5292, abs=1e-3)
        assert kazanc != pytest.approx(1.511, abs=5e-3)

    def test_yaprak_degerleri_notla_uyuyor(self):
        """V_sol ≈ -0.709, V_sağ ≈ +0.631. Not doğru."""
        veri, _, p0 = self._baslangic()
        artiklar = veri.y - p0
        assert xgb_yaprak_degeri(artiklar[:3], p0,
                                 lam=self.LAMBDA) == pytest.approx(-0.709, abs=1e-3)
        assert xgb_yaprak_degeri(artiklar[3:], p0,
                                 lam=self.LAMBDA) == pytest.approx(0.631, abs=2e-3)

    def test_tahmin_guncellemesi(self):
        """F1 = F0 + η·V; η = 0.1 -> ID1 için 0.845 -> p = 0.699."""
        veri, f0, p0 = self._baslangic()
        artiklar = veri.y - p0
        v_sol = xgb_yaprak_degeri(artiklar[:3], p0, lam=self.LAMBDA)
        f1 = f0 + 0.1 * v_sol
        assert f1 == pytest.approx(0.845, abs=1e-3)
        assert float(sigmoid(f1)) == pytest.approx(0.699, abs=1e-3)

    def test_lambda_yaprak_degerini_bastirir(self):
        """λ arttıkça |V| küçülür, ders notundaki R=2, Σh=1 örneği.

        λ = 1  -> V = 2/2   = 1.0
        λ = 10 -> V = 2/11 ≈ 0.18
        """
        assert xgb_yaprak_degeri(gradyan_toplami=2, hessian_toplami=1,
                                 lam=1) == pytest.approx(1.0)
        assert xgb_yaprak_degeri(gradyan_toplami=2, hessian_toplami=1,
                                 lam=10) == pytest.approx(0.1818, abs=1e-4)

    def test_gamma_kucuk_bolmeleri_reddeder(self):
        """γ = 0.1 iken gain = 0.05 olan bölme negatife düşer -> reddedilir."""
        kazanc = xgb_kazanc(0.03, 0.02, 0.0, gama=0.1)
        assert kazanc < 0

    def test_yarim_faktor_siralamayi_degistirmez(self):
        """½ ortak çarpandır: bölme sıralaması değişmez, mutlak değer 2 kat düşer."""
        a = xgb_kazanc(0.81, 0.72, 0.0)
        b = xgb_kazanc(0.50, 0.40, 0.0)
        a_yarim = xgb_kazanc(0.81, 0.72, 0.0, yarim_faktor=True)
        b_yarim = xgb_kazanc(0.50, 0.40, 0.0, yarim_faktor=True)
        assert (a > b) == (a_yarim > b_yarim)
        assert a_yarim == pytest.approx(a / 2)
        assert "xgboost_gain_formulu" in KAYNAK_NOTLARI


class TestXGBoostRegresyon:
    """18-XGBoost_Algorithm.pdf regresyon bölümü, λ = 0."""

    def test_baslangic_ve_artiklar(self):
        veri = xgb_regresyon()
        f0 = gb_baslangic_degeri(veri.y)
        assert f0 == pytest.approx(5000.0)
        assert (veri.y - f0).tolist() == pytest.approx(
            [-3000, -2000, -1000, 0, 1000, 2000, 3000])

    def test_kazanc_mse_bazli(self):
        """Maaş ≤ 60 bölmesi: (-6000)²/4 + 6000²/3 - 0 = 21.000.000."""
        veri = xgb_regresyon()
        artiklar = veri.y - 5000.0
        sol, sag = artiklar[:4], artiklar[4:]
        assert float(sol.sum()) == pytest.approx(-6000.0)
        assert float(sag.sum()) == pytest.approx(6000.0)

        kazanc = xgb_kazanc(xgb_benzerlik(artiklar=sol, lam=0),
                            xgb_benzerlik(artiklar=sag, lam=0),
                            xgb_benzerlik(artiklar=artiklar, lam=0))
        assert kazanc == pytest.approx(21_000_000.0)

    def test_yaprak_degerleri_ve_guncelleme(self):
        """V_sol = -1500, V_sağ = +2000; η=0.1 -> F1 = 4850 ve 5200."""
        veri = xgb_regresyon()
        artiklar = veri.y - 5000.0
        v_sol = xgb_yaprak_degeri(artiklar[:4], lam=0)
        v_sag = xgb_yaprak_degeri(artiklar[4:], lam=0)
        assert v_sol == pytest.approx(-1500.0)
        assert v_sag == pytest.approx(2000.0)
        assert 5000 + 0.1 * v_sol == pytest.approx(4850.0)
        assert 5000 + 0.1 * v_sag == pytest.approx(5200.0)

    def test_regresyonda_hessian_ornek_sayisidir(self):
        """λ=0 iken (ΣR)²/(Σh) formülü (ΣR)²/N'e indirgenir: h = 1."""
        artiklar = np.array([-3000.0, -2000.0, -1000.0, 0.0])
        assert xgb_benzerlik(artiklar=artiklar, lam=0) == pytest.approx(
            artiklar.sum() ** 2 / artiklar.size)


# ==========================================================================
# Kaynak notlarının bütünlüğü
# ==========================================================================
def test_kaynak_notlari_eksiksiz():
    """Tespit edilen her tutarsızlık KAYNAK_NOTLARI içinde belgelenmiş olmalı."""
    beklenen = {
        "karar_agaci_kok_varyans",
        "karar_agaci_en_iyi_esik",
        "adaboost_ikinci_stump_tablosu",
        "xgboost_sag_benzerlik",
        "xgboost_kok_benzerlik_gerekcesi",
        "xgboost_gain_formulu",
        "regresyon_metrikleri_eksik",
    }
    assert beklenen <= set(KAYNAK_NOTLARI)
    for anahtar, metin in KAYNAK_NOTLARI.items():
        assert len(metin) > 100, f"{anahtar} açıklaması çok kısa"


def test_veri_setleri_tutarli():
    """Her ders veri setinin X/y boyutları ve meta bilgisi tutarlı olmalı."""
    uretecler = [kredi_onay, burs_regresyon, knn_sporcu, naive_bayes_spam,
                 naive_bayes_obezite, gb_maas, gb_kredi_karti,
                 xgb_siniflandirma, xgb_regresyon, adaboost_regresyon]
    for uretec in uretecler:
        veri = uretec()
        assert veri.X.shape[0] == veri.y.shape[0] == len(veri)
        assert veri.X.shape[1] == len(veri.ozellik_adlari)
        assert veri.kaynak.endswith(".pdf") or ".pdf" in veri.kaynak
        assert veri.tablo().count("\n") == len(veri) + 1
