"""Karar ağacı bölme ölçütlerinin testleri."""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.saflik import (agirlikli_saflik, bilgi_kazanci, en_iyi_bolunme,
                              entropi, esik_adaylari, gini, kazanc_orani,
                              siniflandirma_hatasi, varyans,
                              varyans_azaltimi)


class TestSaflikOlcutleri:
    def test_saf_kume_sifir(self):
        saf = np.array([1, 1, 1, 1])
        assert entropi(saf) == 0.0
        assert gini(saf) == 0.0
        assert siniflandirma_hatasi(saf) == 0.0

    def test_yari_yariya_maksimum(self):
        """İki sınıf yarı yarıya: entropi 1 bit, Gini 0.5, hata 0.5."""
        karisik = np.array([0, 0, 1, 1])
        assert entropi(karisik) == pytest.approx(1.0)
        assert gini(karisik) == pytest.approx(0.5)
        assert siniflandirma_hatasi(karisik) == pytest.approx(0.5)

    def test_entropi_ust_siniri_log2_k(self):
        """k sınıf eşit dağılımda entropi = log2(k)."""
        for k in (2, 4, 8):
            y = np.arange(k * 10) % k
            assert entropi(y) == pytest.approx(np.log2(k))

    def test_gini_ust_siniri(self):
        """k sınıf eşit dağılımda Gini = 1 - 1/k."""
        for k in (2, 3, 5):
            y = np.arange(k * 10) % k
            assert gini(y) == pytest.approx(1 - 1 / k)

    def test_etiket_isimleri_onemsiz(self):
        """Saflık ölçütleri etiketin kendisine değil dağılımına bakar."""
        assert entropi([0, 0, 1]) == pytest.approx(entropi(["a", "a", "b"]))
        assert gini([0, 0, 1]) == pytest.approx(gini(["kedi", "kedi", "kopek"]))

    def test_siralama_onemsiz(self):
        assert entropi([0, 1, 0, 1]) == pytest.approx(entropi([0, 0, 1, 1]))

    def test_bos_kume_sifir(self):
        assert entropi([]) == 0.0
        assert gini([]) == 0.0
        assert varyans([]) == 0.0

    def test_entropi_gini_den_buyuk_kalir(self):
        """İki sınıflı durumda H(p) >= Gini(p); ikisi de p=0.5'te tepe yapar."""
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            y = np.array([1] * int(p * 100) + [0] * int((1 - p) * 100))
            assert entropi(y) >= gini(y) - 1e-12

    def test_taban_degistirme(self):
        """Doğal logaritma tabanı: H_e = H_2 · ln(2)."""
        y = np.array([0, 0, 1, 1])
        assert entropi(y, taban=np.e) == pytest.approx(np.log(2))


class TestVaryans:
    def test_ddof_sifir_kullanilir(self):
        """Ağaçlar popülasyon varyansı kullanır; pandas varsayılanı (ddof=1) farklıdır."""
        y = np.array([10.0, 20.0, 30.0])
        assert varyans(y) == pytest.approx(float(np.var(y)))
        assert varyans(y) != pytest.approx(float(np.var(y, ddof=1)))
        assert varyans(y) == pytest.approx(200 / 3)

    def test_sabit_deger_sifir(self):
        assert varyans(np.full(5, 7.0)) == 0.0


class TestAgirlikliSaflik:
    def test_buyuk_alt_kume_agir_basar(self):
        """1 örnekli saf yaprak ile 100 örnekli saf yaprak aynı sayılmaz."""
        kucuk_saf = np.array([1])
        buyuk_karisik = np.array([0] * 50 + [1] * 50)
        sonuc = agirlikli_saflik([kucuk_saf, buyuk_karisik], olcut="entropi")
        assert sonuc == pytest.approx(100 / 101 * 1.0, abs=1e-3)

    def test_bilinmeyen_olcut_hata(self):
        with pytest.raises(ValueError, match="olcut"):
            agirlikli_saflik([[0, 1]], olcut="bilinmeyen")

    def test_cagrilabilir_olcut(self):
        assert agirlikli_saflik([[0, 1]], olcut=lambda y: 0.42) == pytest.approx(0.42)


class TestBilgiKazanci:
    def test_mukemmel_bolme_kok_entropisi_kadar_kazandirir(self):
        y = np.array([0, 0, 1, 1])
        maske = np.array([True, True, False, False])
        assert bilgi_kazanci(y, maske=maske) == pytest.approx(entropi(y))
        assert bilgi_kazanci(y, maske=maske) == pytest.approx(1.0)

    def test_ise_yaramaz_bolme_sifir_kazandirir(self):
        """Her iki tarafa da aynı oranı bırakan bölme bilgi katmaz."""
        y = np.array([0, 1, 0, 1])
        maske = np.array([True, True, False, False])
        assert bilgi_kazanci(y, maske=maske) == pytest.approx(0.0)

    def test_kazanc_asla_negatif_degil(self):
        """Saflık ölçütleri içbükey olduğu için kazanç >= 0 olmak zorundadır."""
        uretec = np.random.default_rng(0)
        for _ in range(50):
            y = uretec.integers(0, 3, size=20)
            maske = uretec.random(20) > 0.5
            if maske.all() or not maske.any():
                continue
            for olcut in ("entropi", "gini"):
                assert bilgi_kazanci(y, maske=maske, olcut=olcut) >= -1e-12

    def test_alt_kume_listesiyle_de_calisir(self):
        y = np.array([0, 0, 1, 1])
        maske = np.array([True, True, False, False])
        assert bilgi_kazanci(y, [y[maske], y[~maske]]) == pytest.approx(
            bilgi_kazanci(y, maske=maske))

    def test_uc_yollu_bolme(self):
        """ID3 ikiden fazla dala izin verir; ağırlıklı saflık aynı şekilde çalışır."""
        y = np.array([0, 0, 1, 1, 2, 2])
        alt_kumeler = [y[:2], y[2:4], y[4:]]
        assert bilgi_kazanci(y, alt_kumeler) == pytest.approx(entropi(y))

    def test_maske_uzunlugu_kontrolu(self):
        with pytest.raises(ValueError, match="maske uzunluğu"):
            bilgi_kazanci([0, 1, 1], maske=[True, False])

    def test_maske_veya_altkume_zorunlu(self):
        with pytest.raises(ValueError, match="alt_kumeler veya maske"):
            bilgi_kazanci([0, 1])


class TestKazancOrani:
    def test_cok_dalli_bolmeyi_cezalandirir(self):
        """C4.5'in varlık sebebi: "kimlik" sütunu ham kazançta kazanır, oranda kaybeder.

        6 örnek, her biri farklı kimlik -> her yaprak tek örnekli, entropi 0,
        ham kazanç maksimum. Ama SplitInfo = log2(6) = 2.585 olduğu için
        kazanç oranı belirgin düşer.
        """
        y = np.array([0, 0, 0, 1, 1, 1])
        kimlik_bolmesi = [np.array([deger]) for deger in y]      # 6 yaprak
        anlamli_bolme = [y[:3], y[3:]]                            # 2 yaprak

        ham_kimlik = bilgi_kazanci(y, kimlik_bolmesi)
        ham_anlamli = bilgi_kazanci(y, anlamli_bolme)
        assert ham_kimlik == pytest.approx(ham_anlamli)  # ikisi de tam ayırıyor

        oran_kimlik = kazanc_orani(y, kimlik_bolmesi)
        oran_anlamli = kazanc_orani(y, anlamli_bolme)
        assert oran_anlamli > oran_kimlik
        assert oran_anlamli == pytest.approx(1.0)
        assert oran_kimlik == pytest.approx(1 / np.log2(6), abs=1e-6)


class TestVaryansAzaltimi:
    def test_mukemmel_bolme(self):
        y = np.array([10.0, 10.0, 20.0, 20.0])
        maske = np.array([True, True, False, False])
        assert varyans_azaltimi(y, maske=maske) == pytest.approx(varyans(y))
        assert varyans_azaltimi(y, maske=maske) == pytest.approx(25.0)

    def test_isE_yaramaz_bolme(self):
        y = np.array([10.0, 20.0, 10.0, 20.0])
        maske = np.array([True, True, False, False])
        assert varyans_azaltimi(y, maske=maske) == pytest.approx(0.0)

    def test_mse_azalisiyla_ayni_bolmeyi_secer(self):
        """Varyans azaltımı ile ağırlıklı MSE azalışı aynı sıralamayı verir."""
        uretec = np.random.default_rng(1)
        X = uretec.random(40)
        y = 3 * X + uretec.normal(0, 0.1, 40)

        azaltimlar, mse_dususleri = [], []
        for esik in esik_adaylari(X):
            maske = X <= esik
            if maske.sum() < 2 or (~maske).sum() < 2:
                continue
            azaltimlar.append(varyans_azaltimi(y, maske=maske))
            agirlikli_mse = (maske.sum() / y.size * np.mean((y[maske] - y[maske].mean())**2)
                             + (~maske).sum() / y.size
                             * np.mean((y[~maske] - y[~maske].mean())**2))
            mse_dususleri.append(float(np.mean((y - y.mean())**2) - agirlikli_mse))
        assert np.argmax(azaltimlar) == np.argmax(mse_dususleri)


class TestEsikAdaylari:
    def test_orta_noktalar(self):
        assert esik_adaylari([2.5, 2.7, 3.0]) == pytest.approx([2.6, 2.85])

    def test_tekrarlar_teklestirilir(self):
        assert esik_adaylari([1.0, 1.0, 2.0, 2.0]) == pytest.approx([1.5])

    def test_tek_deger_aday_yok(self):
        assert esik_adaylari([5.0, 5.0, 5.0]).size == 0

    def test_esikler_gozlemlerin_arasina_duser(self):
        degerler = np.array([1.0, 4.0, 9.0])
        for esik in esik_adaylari(degerler):
            assert degerler.min() < esik < degerler.max()
            assert esik not in degerler


class TestEnIyiBolunme:
    def test_tek_ozellikli_veri(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        sonuc = en_iyi_bolunme(X, y)
        assert sonuc["ozellik"] == 0
        assert sonuc["esik"] == pytest.approx(2.5)
        assert sonuc["kazanc"] == pytest.approx(1.0)
        assert sonuc["sol_sayi"] == sonuc["sag_sayi"] == 2

    def test_bilgilendirici_ozelligi_secer(self):
        """İkinci sütun hedefi belirliyor, ilki gürültü."""
        uretec = np.random.default_rng(0)
        gurultu = uretec.random(40)
        sinyal = np.arange(40, dtype=float)
        X = np.column_stack([gurultu, sinyal])
        y = (sinyal >= 20).astype(int)

        sonuc = en_iyi_bolunme(X, y, ozellik_adlari=["gurultu", "sinyal"])
        assert sonuc["ozellik_adi"] == "sinyal"
        assert sonuc["esik"] == pytest.approx(19.5)
        assert sonuc["kazanc"] == pytest.approx(1.0)

    def test_en_az_yaprak_kisiti(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.array([0] + [1] * 9)
        serbest = en_iyi_bolunme(X, y, en_az_yaprak=1)
        kisitli = en_iyi_bolunme(X, y, en_az_yaprak=3)
        assert serbest["sol_sayi"] == 1
        assert kisitli["sol_sayi"] >= 3
        assert kisitli["kazanc"] <= serbest["kazanc"]

    def test_bolunemeyen_veri(self):
        X = np.full((5, 1), 3.0)
        y = np.array([0, 1, 0, 1, 0])
        sonuc = en_iyi_bolunme(X, y)
        assert sonuc["esik"] is None
        assert sonuc["kazanc"] == 0.0

    def test_tum_adaylar_kazanca_gore_sirali(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.array([0] * 5 + [1] * 5)
        adaylar = en_iyi_bolunme(X, y)["tum_adaylar"]
        kazanclar = [aday["kazanc"] for aday in adaylar]
        assert kazanclar == sorted(kazanclar, reverse=True)

    def test_regresyon_olcutu(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([1.0] * 4 + [10.0] * 4)
        sonuc = en_iyi_bolunme(X, y, olcut="varyans")
        assert sonuc["esik"] == pytest.approx(3.5)

    def test_boyut_uyusmazligi(self):
        with pytest.raises(ValueError, match="satır sayısı"):
            en_iyi_bolunme(np.arange(6.0).reshape(-1, 1), np.array([0, 1]))
