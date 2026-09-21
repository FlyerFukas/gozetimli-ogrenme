"""Regresyon metriklerinin birim testleri."""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.metrikler.regresyon import (aciklanan_varyans, duzeltilmis_r2,
                                           mae, maksimum_hata, mape,
                                           medyan_mutlak_hata, mse, msle, r2,
                                           regresyon_raporu, rmse, rmsle,
                                           smape)


@pytest.fixture
def basit():
    """Hatalar: +1, -1, +2, -2 -> MSE = 10/4 = 2.5, MAE = 6/4 = 1.5."""
    gercek = np.array([10.0, 20.0, 30.0, 40.0])
    tahmin = np.array([11.0, 19.0, 32.0, 38.0])
    return gercek, tahmin


class TestTemelHatalar:
    def test_mse(self, basit):
        gercek, tahmin = basit
        assert mse(gercek, tahmin) == pytest.approx(2.5)

    def test_rmse_mse_karekoku(self, basit):
        gercek, tahmin = basit
        assert rmse(gercek, tahmin) == pytest.approx(np.sqrt(2.5))

    def test_mae(self, basit):
        gercek, tahmin = basit
        assert mae(gercek, tahmin) == pytest.approx(1.5)

    def test_rmse_mae_den_buyuk_esit(self, basit):
        """Jensen eşitsizliği: RMSE >= MAE, eşitlik yalnızca tüm hatalar eşitken."""
        gercek, tahmin = basit
        assert rmse(gercek, tahmin) >= mae(gercek, tahmin)

        esit_hatalar_gercek = np.array([1.0, 2.0, 3.0])
        esit_hatalar_tahmin = np.array([2.0, 3.0, 4.0])
        assert rmse(esit_hatalar_gercek, esit_hatalar_tahmin) == pytest.approx(
            mae(esit_hatalar_gercek, esit_hatalar_tahmin))

    def test_mukemmel_tahmin_sifir(self, basit):
        gercek, _ = basit
        assert mse(gercek, gercek) == 0.0
        assert mae(gercek, gercek) == 0.0
        assert rmse(gercek, gercek) == 0.0

    def test_maksimum_hata(self, basit):
        gercek, tahmin = basit
        assert maksimum_hata(gercek, tahmin) == pytest.approx(2.0)

    def test_medyan_mutlak_hata(self, basit):
        """|hatalar| = [1,1,2,2] -> medyan 1.5."""
        gercek, tahmin = basit
        assert medyan_mutlak_hata(gercek, tahmin) == pytest.approx(1.5)

    def test_ornek_agirliklari(self):
        """Ağırlıklı MSE: ilk örneğe 3 kat ağırlık."""
        gercek = np.array([0.0, 0.0])
        tahmin = np.array([2.0, 0.0])      # hatalar: 4 ve 0
        assert mse(gercek, tahmin) == pytest.approx(2.0)
        assert mse(gercek, tahmin, ornek_agirliklari=[3, 1]) == pytest.approx(3.0)


class TestAykiriDegerDavranisi:
    """MSE ile MAE arasındaki seçim, aykırı değer politikasıdır."""

    def test_tek_aykiri_deger_mse_yi_patlatir(self):
        gercek = np.zeros(100)
        tahmin = np.zeros(100)
        tahmin[0] = 100.0                  # tek büyük hata

        assert mae(gercek, tahmin) == pytest.approx(1.0)     # 100/100
        assert mse(gercek, tahmin) == pytest.approx(100.0)   # 10000/100
        assert medyan_mutlak_hata(gercek, tahmin) == 0.0

    def test_mse_ortalamayi_mae_medyani_odullendirir(self):
        """Sabit tahmin arayışı: MSE ortalamayı, MAE medyanı seçer."""
        y = np.array([1.0, 2.0, 3.0, 100.0])
        ortalama, medyan = float(np.mean(y)), float(np.median(y))

        assert mse(y, np.full(4, ortalama)) < mse(y, np.full(4, medyan))
        assert mae(y, np.full(4, medyan)) < mae(y, np.full(4, ortalama))


class TestR2:
    def test_mukemmel_tahmin_bir(self, basit):
        gercek, _ = basit
        assert r2(gercek, gercek) == pytest.approx(1.0)

    def test_ortalama_tahmini_sifir(self, basit):
        """Her şeye ortalamayı söyleyen model R² = 0 alır."""
        gercek, _ = basit
        assert r2(gercek, np.full(4, gercek.mean())) == pytest.approx(0.0)

    def test_ortalamadan_kotu_model_negatif(self, basit):
        """R² negatif olabilir, test setinde bu gerçek bir uyarıdır."""
        gercek, _ = basit
        kotu = gercek[::-1]
        assert r2(gercek, kotu) < 0

    def test_r2_mse_ile_iliskisi(self, basit):
        """R² = 1 - MSE/Var(y)."""
        gercek, tahmin = basit
        beklenen = 1 - mse(gercek, tahmin) / float(np.var(gercek))
        assert r2(gercek, tahmin) == pytest.approx(beklenen)

    def test_sabit_hedef_kenar_durumu(self):
        sabit = np.full(5, 7.0)
        assert r2(sabit, sabit) == 1.0
        assert r2(sabit, np.full(5, 8.0)) == 0.0

    def test_duzeltilmis_r2_ozellik_sayisini_cezalandirir(self):
        gercek = np.arange(50, dtype=float)
        tahmin = gercek + np.linspace(-1, 1, 50)
        ham = r2(gercek, tahmin)
        az = duzeltilmis_r2(gercek, tahmin, ozellik_sayisi=2)
        cok = duzeltilmis_r2(gercek, tahmin, ozellik_sayisi=20)
        assert ham > az > cok

    def test_duzeltilmis_r2_yetersiz_ornek(self):
        with pytest.raises(ValueError, match="en az 2 fazla"):
            duzeltilmis_r2(np.arange(5.0), np.arange(5.0), ozellik_sayisi=4)

    def test_aciklanan_varyans_yanliligi_gormez(self):
        """Sistematik +5 sapma: R² düşer, açıklanan varyans 1.0 kalır."""
        gercek = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kaydirilmis = gercek + 5.0
        assert aciklanan_varyans(gercek, kaydirilmis) == pytest.approx(1.0)
        assert r2(gercek, kaydirilmis) < 0
        assert aciklanan_varyans(gercek, kaydirilmis) > r2(gercek, kaydirilmis)

    def test_yanlilik_yoksa_iki_metrik_esit(self, basit):
        gercek, tahmin = basit
        # Bu örnekte kalıntı ortalaması 0 -> iki metrik çakışır.
        assert float(np.mean(gercek - tahmin)) == pytest.approx(0.0)
        assert r2(gercek, tahmin) == pytest.approx(aciklanan_varyans(gercek, tahmin))


class TestYuzdeHatalar:
    def test_mape_oran_dondurur(self):
        """Hatalar %10 ve %10 -> MAPE = 0.10."""
        gercek = np.array([100.0, 200.0])
        tahmin = np.array([110.0, 180.0])
        assert mape(gercek, tahmin) == pytest.approx(0.10)

    def test_mape_asimetriktir(self):
        """Aynı mutlak hata, düşük tahminde daha az cezalandırılır."""
        gercek = np.array([100.0])
        assert mape(gercek, np.array([50.0])) == pytest.approx(0.5)    # düşük tahmin
        assert mape(gercek, np.array([150.0])) == pytest.approx(0.5)   # yüksek tahmin
        # Asimetri, tahmin 0'a veya sonsuza giderken ortaya çıkar:
        assert mape(gercek, np.array([0.0])) == pytest.approx(1.0)     # üst sınır 1
        assert mape(gercek, np.array([300.0])) == pytest.approx(2.0)   # sınırsız

    def test_mape_kucuk_gercek_degerde_patlar(self):
        gercek = np.array([1.0])
        assert mape(gercek, np.array([2.0])) == pytest.approx(1.0)   # %100
        assert mae(gercek, np.array([2.0])) == pytest.approx(1.0)    # mutlak 1 birim

    def test_mape_sifirda_epsilon_ile_korunur(self):
        gercek = np.array([0.0, 100.0])
        tahmin = np.array([1.0, 100.0])
        assert np.isfinite(mape(gercek, tahmin))

    def test_smape_sinirlidir(self):
        """sMAPE [0, 2] aralığındadır; MAPE sınırsızdır."""
        gercek = np.array([1.0])
        assert smape(gercek, np.array([1000.0])) <= 2.0
        assert mape(gercek, np.array([1000.0])) > 900

    def test_smape_sifir_sifir_katkisiz(self):
        gercek = np.array([0.0, 10.0])
        tahmin = np.array([0.0, 10.0])
        assert smape(gercek, tahmin) == pytest.approx(0.0)


class TestLogaritmikHatalar:
    def test_msle_oransal_hatayi_cezalandirir(self):
        """Aynı ORAN hatası (2 kat), farklı ölçekte aynı MSLE'ye yakınsar.

        Sınır değer log(2)² ≈ 0.4805'tir; y büyüdükçe MSLE buna yaklaşır.
        MSE ise aynı oran hatasında ölçeğin karesiyle büyür.
        """
        kucuk = msle(np.array([100.0]), np.array([200.0]))
        buyuk = msle(np.array([10_000.0]), np.array([20_000.0]))
        assert kucuk == pytest.approx(buyuk, rel=0.02)
        assert buyuk == pytest.approx(np.log(2) ** 2, rel=0.01)

        # MSE ise ölçekle birlikte patlar: 100^2 kat.
        assert mse(np.array([10_000.0]), np.array([20_000.0])) == pytest.approx(
            mse(np.array([100.0]), np.array([200.0])) * 100**2)

    def test_msle_kucuk_degerlerde_log1p_sapmasi(self):
        """log1p'teki +1, küçük değerlerde oransal denkliği bozar, bilinçli bir ödün.

        y = 10 -> 20 için MSLE 0.418; y = 10 000 -> 20 000 için 0.480.
        +1 terimi olmasaydı sıfır hedefte logaritma tanımsız olurdu.
        """
        assert msle(np.array([10.0]), np.array([20.0])) == pytest.approx(
            0.4181, abs=1e-4)
        assert msle(np.array([10_000.0]), np.array([20_000.0])) == pytest.approx(
            0.4803, abs=1e-4)
        assert msle(np.array([0.0]), np.array([0.0])) == 0.0

    def test_rmsle_karekok(self):
        gercek, tahmin = np.array([10.0, 20.0]), np.array([12.0, 18.0])
        assert rmsle(gercek, tahmin) == pytest.approx(np.sqrt(msle(gercek, tahmin)))

    def test_negatif_deger_hata_verir(self):
        with pytest.raises(ValueError, match="negatif"):
            msle(np.array([-1.0]), np.array([1.0]))


class TestRapor:
    def test_rapor_anahtarlari(self, basit):
        gercek, tahmin = basit
        rapor = regresyon_raporu(gercek, tahmin, ozellik_sayisi=1)
        assert {"mse", "rmse", "mae", "r2", "mape", "duzeltilmis_r2"} <= set(rapor)
        assert rapor["mse"] == pytest.approx(2.5)
        assert rapor["mae"] == pytest.approx(1.5)

    def test_sifir_iceren_hedefte_mape_atlanir(self):
        gercek = np.array([0.0, 10.0, 20.0])
        tahmin = np.array([1.0, 11.0, 19.0])
        rapor = regresyon_raporu(gercek, tahmin)
        assert "mape" not in rapor
        assert "rmse" in rapor


class TestGirdiDogrulama:
    def test_uzunluk_uyusmazligi(self):
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            mse([1, 2, 3], [1, 2])

    def test_bos_dizi(self):
        with pytest.raises(ValueError, match="boş olamaz"):
            mse([], [])

    def test_iki_boyutlu_dizi_reddedilir(self):
        with pytest.raises(ValueError, match="1 boyutlu"):
            mse(np.zeros((3, 2)), np.zeros((3, 2)))

    def test_tek_sutunlu_matris_kabul_edilir(self):
        """(n, 1) şekli yaygın bir kazadır; sessizce düzleştirilir."""
        assert mse(np.array([[1.0], [2.0]]), np.array([1.0, 2.0])) == 0.0
