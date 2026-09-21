"""Boosting yardımcı fonksiyonlarının testleri (ders örnekleri dışındaki davranış)."""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.boosting import (adaboost_agirlik_katsayisi,
                                adaboost_agirliklari_guncelle,
                                adaboost_bin_araliklari, adaboost_hata_orani,
                                adaboost_nihai_skor, gb_baslangic_degeri,
                                log_loss_gradyan, log_loss_hessian, log_odds,
                                sigmoid, xgb_benzerlik, xgb_cover, xgb_kazanc,
                                xgb_yaprak_degeri)


class TestSigmoidVeLogOdds:
    def test_sifirda_yarim(self):
        assert float(sigmoid(0.0)) == pytest.approx(0.5)

    def test_simetri(self):
        """σ(-z) = 1 - σ(z)."""
        for z in (-3.0, -0.5, 0.0, 1.2, 4.0):
            assert float(sigmoid(-z)) == pytest.approx(1 - float(sigmoid(z)))

    def test_buyuk_degerlerde_tasma_yok(self):
        """Naif 1/(1+exp(-z)) formülü z=-1000'de taşar; bu implementasyon taşmaz."""
        with np.errstate(over="raise"):
            assert float(sigmoid(-1000.0)) == pytest.approx(0.0, abs=1e-300)
            assert float(sigmoid(1000.0)) == pytest.approx(1.0)
        assert np.all(np.isfinite(sigmoid(np.array([-800.0, 0.0, 800.0]))))

    def test_log_odds_sigmoidin_tersi(self):
        for p in (0.01, 0.25, 0.5, 0.73, 0.99):
            assert float(sigmoid(log_odds(p))) == pytest.approx(p)

    def test_log_odds_kirpma(self):
        """p = 0 veya 1'de logit ±sonsuz; epsilon ile sonlu tutulur."""
        assert np.isfinite(log_odds(0.0))
        assert np.isfinite(log_odds(1.0))
        assert log_odds(0.0) < -30
        assert log_odds(1.0) > 30

    def test_dizi_girdisi(self):
        p = sigmoid(np.array([-2.0, 0.0, 2.0]))
        assert p.shape == (3,)
        assert p[0] < p[1] < p[2]


class TestLogLossTurevleri:
    def test_gradyan_artigin_negatifi(self):
        y, p = np.array([1, 0]), np.array([0.7, 0.3])
        assert log_loss_gradyan(y, p).tolist() == pytest.approx([-0.3, 0.3])
        assert log_loss_gradyan(y, p).tolist() == pytest.approx((-(y - p)).tolist())

    def test_hessian_tepe_noktasi(self):
        """h = p(1-p), p=0.5'te maksimum (0.25), uçlarda sıfıra gider."""
        assert float(log_loss_hessian(0.5)) == pytest.approx(0.25)
        assert float(log_loss_hessian(0.01)) < 0.01
        assert float(log_loss_hessian(0.99)) < 0.01

    def test_ders_ornegi_kucuk_hata_kucuk_duzeltme(self):
        """Not: y=1, p=0.9 -> g = -0.1, h = 0.09 -> düzeltme küçük."""
        assert float(log_loss_gradyan(1, 0.9)) == pytest.approx(-0.1)
        assert float(log_loss_hessian(0.9)) == pytest.approx(0.09)


class TestAdaBoostAgirliklari:
    def test_hata_orani_agirlikli(self):
        agirliklar = np.array([0.5, 0.2, 0.2, 0.1])
        dogru = np.array([True, False, True, False])
        assert adaboost_hata_orani(agirliklar, dogru) == pytest.approx(0.3)

    def test_alfa_hata_azaldikca_artar(self):
        alfalar = [adaboost_agirlik_katsayisi(e) for e in (0.4, 0.3, 0.2, 0.1)]
        assert alfalar == sorted(alfalar)

    def test_alfa_sifir_hatada_sonsuza_gitmez(self):
        assert np.isfinite(adaboost_agirlik_katsayisi(0.0))
        assert adaboost_agirlik_katsayisi(0.0) > 10

    def test_guncelleme_yanlislari_agirlastirir(self):
        agirliklar = np.full(4, 0.25)
        dogru = np.array([True, True, True, False])
        yeni = adaboost_agirliklari_guncelle(agirliklar, dogru, 0.5)
        assert yeni[3] > yeni[0]
        assert yeni.sum() == pytest.approx(1.0)

    def test_normalize_kapatilabilir(self):
        agirliklar = np.full(4, 0.25)
        dogru = np.ones(4, dtype=bool)
        ham = adaboost_agirliklari_guncelle(agirliklar, dogru, 0.5, normalize=False)
        assert ham.sum() != pytest.approx(1.0)
        assert ham.tolist() == pytest.approx((0.25 * np.exp(-0.5)).repeat(4).tolist())

    def test_alfa_sifirken_agirliklar_degismez(self):
        """ε = 0.5 -> α = 0 -> e^0 = 1: hiçbir örneğin ağırlığı değişmez."""
        agirliklar = np.array([0.4, 0.3, 0.2, 0.1])
        dogru = np.array([True, False, True, False])
        yeni = adaboost_agirliklari_guncelle(agirliklar, dogru, 0.0)
        assert yeni.tolist() == pytest.approx(agirliklar.tolist())

    def test_bin_araliklari_kapsayici(self):
        agirliklar = np.array([0.1, 0.2, 0.3, 0.4])
        araliklar = adaboost_bin_araliklari(agirliklar)
        assert araliklar[0][0] == 0.0
        assert araliklar[-1][1] == pytest.approx(1.0)
        # Aralıklar birbirini takip etmeli, boşluk olmamalı.
        for once, sonra in zip(araliklar, araliklar[1:]):
            assert once[1] == pytest.approx(sonra[0])

    def test_bin_genisligi_agirlikla_orantili(self):
        agirliklar = np.array([0.1, 0.9])
        araliklar = adaboost_bin_araliklari(agirliklar)
        genislikler = araliklar[:, 1] - araliklar[:, 0]
        assert genislikler.tolist() == pytest.approx([0.1, 0.9])

    def test_normalize_edilmemis_agirlik_kabul_edilir(self):
        araliklar = adaboost_bin_araliklari(np.array([1.0, 3.0]))
        assert (araliklar[:, 1] - araliklar[:, 0]).tolist() == pytest.approx([0.25, 0.75])

    def test_nihai_skor_matris_bicimi(self):
        """3 model x 4 örnek -> örnek başına skor."""
        alfalar = np.array([0.9, 0.5, 0.3])
        tahminler = np.array([[1, 1, -1, -1],
                              [1, -1, 1, -1],
                              [1, 1, 1, -1]])
        skorlar = adaboost_nihai_skor(alfalar, tahminler)
        assert skorlar.shape == (4,)
        assert skorlar[0] == pytest.approx(1.7)
        assert skorlar[-1] == pytest.approx(-1.7)
        assert np.sign(skorlar).tolist() == [1.0, 1.0, -1.0, -1.0]

    def test_uzunluk_kontrolleri(self):
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            adaboost_hata_orani([0.5, 0.5], [True])
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            adaboost_agirliklari_guncelle([0.5, 0.5], [True], 0.5)


class TestGradientBoostingBaslangic:
    def test_regresyon_ortalama(self):
        assert gb_baslangic_degeri([10.0, 20.0, 30.0]) == pytest.approx(20.0)

    def test_siniflandirma_log_odds(self):
        y = np.array([1, 1, 1, 0])
        assert gb_baslangic_degeri(y, gorev="siniflandirma") == pytest.approx(
            np.log(3))
        assert float(sigmoid(gb_baslangic_degeri(y, gorev="siniflandirma"))) == \
            pytest.approx(0.75)

    def test_tek_sinifli_veri_hata(self):
        with pytest.raises(ValueError, match="Tek sınıflı"):
            gb_baslangic_degeri([1, 1, 1], gorev="siniflandirma")

    def test_gecersiz_gorev(self):
        with pytest.raises(ValueError, match="gorev"):
            gb_baslangic_degeri([1, 0], gorev="kumeleme")

    def test_log_odds_baslangici_artiklari_sifirlar(self):
        """F₀ = logit(p̄) seçimi, Σ(y - p) = 0 olmasını garanti eder.

        Bu, log-loss'un birinci derece optimallik koşuludur ve XGBoost'ta kök
        similarity'sinin neden sıfır olduğunu açıklar.
        """
        for y in ([0, 0, 1, 1, 1], [1, 0], [0] * 7 + [1] * 3):
            y = np.array(y)
            p0 = float(sigmoid(gb_baslangic_degeri(y, gorev="siniflandirma")))
            assert float(np.sum(y - p0)) == pytest.approx(0.0, abs=1e-10)


class TestXGBoostFormulleri:
    def test_benzerlik_isaretten_bagimsiz(self):
        """Pay karesi alındığı için artıkların işareti skoru değiştirmez."""
        assert xgb_benzerlik(artiklar=[1.0, 1.0], lam=1) == pytest.approx(
            xgb_benzerlik(artiklar=[-1.0, -1.0], lam=1))

    def test_ayni_yonlu_artiklar_daha_yuksek_skor(self):
        """Saflığın boosting karşılığı: artıklar aynı yöne bakarsa skor yüksek."""
        ayni_yon = xgb_benzerlik(artiklar=[0.5, 0.5, 0.5], lam=1)
        karisik = xgb_benzerlik(artiklar=[0.5, -0.5, 0.5], lam=1)
        assert ayni_yon > karisik

    def test_lambda_skoru_kucultur(self):
        degerler = [xgb_benzerlik(artiklar=[2.0, 2.0], lam=lam)
                    for lam in (0, 1, 5, 20)]
        assert degerler == sorted(degerler, reverse=True)

    def test_lambda_sifirda_klasik_gb(self):
        """λ = 0, regresyonda (ΣR)²/N demektir."""
        artiklar = np.array([-3.0, -1.0, 2.0, 2.0])
        assert xgb_benzerlik(artiklar=artiklar, lam=0) == pytest.approx(
            artiklar.sum() ** 2 / artiklar.size)

    def test_yaprak_degeri_isaret_korur(self):
        assert xgb_yaprak_degeri([-2.0, -2.0], lam=1) < 0
        assert xgb_yaprak_degeri([2.0, 2.0], lam=1) > 0

    def test_yaprak_degeri_lambda_ile_kuculur(self):
        buyuk = abs(xgb_yaprak_degeri([3.0, 3.0], lam=0))
        kucuk = abs(xgb_yaprak_degeri([3.0, 3.0], lam=50))
        assert buyuk > kucuk

    def test_dogrudan_toplamlarla_cagri(self):
        assert xgb_benzerlik(4.0, 3.0, lam=1) == pytest.approx(16 / 4)
        assert xgb_yaprak_degeri(gradyan_toplami=4.0, hessian_toplami=3.0,
                                 lam=1) == pytest.approx(1.0)

    def test_eksik_argumanlar_hata(self):
        with pytest.raises(ValueError, match="artiklar verilmelidir"):
            xgb_benzerlik()
        with pytest.raises(ValueError, match="artiklar verilmelidir"):
            xgb_yaprak_degeri()

    def test_sifir_payda_hata(self):
        with pytest.raises(ValueError, match="sıfır olamaz"):
            xgb_benzerlik(1.0, 0.0, lam=0)

    def test_gamma_esik_gorevi_gorur(self):
        ham_kazanc = xgb_kazanc(0.5, 0.4, 0.8)
        assert ham_kazanc == pytest.approx(0.1)
        assert xgb_kazanc(0.5, 0.4, 0.8, gama=0.05) == pytest.approx(0.05)
        assert xgb_kazanc(0.5, 0.4, 0.8, gama=0.2) < 0

    def test_cover_hessian_toplamidir(self):
        olasiliklar = np.array([0.5, 0.5, 0.9])
        assert xgb_cover(olasiliklar) == pytest.approx(0.25 + 0.25 + 0.09)

    def test_cover_emin_tahminlerde_kucuktur(self):
        """Model emin olduğu düğümde cover düşer -> bölmeye gerek yok sinyali."""
        kararsiz = xgb_cover(np.full(10, 0.5))
        emin = xgb_cover(np.full(10, 0.99))
        assert kararsiz == pytest.approx(2.5)
        assert emin < 0.15

    def test_olasilik_skaler_verilebilir(self):
        """Tüm örnekler aynı olasılıktaysa tek sayı yeterli."""
        artiklar = np.array([0.3, 0.3, 0.3])
        assert xgb_benzerlik(artiklar=artiklar, olasiliklar=0.7, lam=1) == \
            pytest.approx(xgb_benzerlik(artiklar=artiklar,
                                        olasiliklar=np.full(3, 0.7), lam=1))

    def test_uzunluk_uyusmazligi(self):
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            xgb_benzerlik(artiklar=[1.0, 2.0], olasiliklar=[0.5, 0.5, 0.5], lam=1)
