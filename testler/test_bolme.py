"""Veri bölme stratejilerinin testleri.

Her bölücü için üç şey doğrulanır:
  1. Sözleşme : kaç bölme üretiyor, indeksler geçerli mi, eğitim/test ayrık mı.
  2. Vaat     : tabakalı gerçekten oranı koruyor mu, grup gerçekten sızdırmıyor mu.
  3. Tuzak    : yanlış bölücü seçildiğinde ne bozuluyor (sızıntı gösterimleri).
"""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.dogrulama.bolme import (BirDisarida, GrupKKat, KKat, PDisarida,
                                       TabakaliKKat, TekrarliKKat,
                                       ZamanSerisiBolme, egitim_test_bol)


def _sozlesmeyi_dogrula(bolucu, X, y=None, gruplar=None):
    """Her bölücünün sağlaması gereken ortak kurallar."""
    n = len(X)
    bolmeler = list(bolucu.bol(X, y, gruplar))
    assert len(bolmeler) == bolucu.kat_sayisi(X, y, gruplar)
    for egitim, test in bolmeler:
        assert len(np.intersect1d(egitim, test)) == 0, "eğitim ve test ayrık olmalı"
        assert egitim.size > 0 and test.size > 0
        assert np.all((egitim >= 0) & (egitim < n))
        assert np.all((test >= 0) & (test < n))
    return bolmeler


class TestKKat:
    def test_sozlesme(self):
        X = np.arange(20).reshape(-1, 1)
        _sozlesmeyi_dogrula(KKat(5), X)

    def test_her_ornek_tam_bir_kez_test_olur(self):
        X = np.arange(20).reshape(-1, 1)
        tum_test = np.concatenate([test for _, test in KKat(5).bol(X)])
        assert np.array_equal(np.sort(tum_test), np.arange(20))

    def test_kat_boyutlari_dengeli(self):
        """n katlara tam bölünmüyorsa fark en fazla 1 olmalı: 10/3 -> 4,3,3."""
        X = np.arange(10).reshape(-1, 1)
        boyutlar = [len(test) for _, test in KKat(3).bol(X)]
        assert boyutlar == [4, 3, 3]
        assert sum(boyutlar) == 10

    def test_karistirmadan_deterministik(self):
        X = np.arange(12).reshape(-1, 1)
        ilk = [t.tolist() for _, t in KKat(4).bol(X)]
        ikinci = [t.tolist() for _, t in KKat(4).bol(X)]
        assert ilk == ikinci
        assert ilk[0] == [0, 1, 2], "karıştırmadan sıralı bloklar"

    def test_tohum_tekrarlanabilirlik_saglar(self):
        X = np.arange(20).reshape(-1, 1)
        a = [t.tolist() for _, t in KKat(5, karistir=True, tohum=42).bol(X)]
        b = [t.tolist() for _, t in KKat(5, karistir=True, tohum=42).bol(X)]
        c = [t.tolist() for _, t in KKat(5, karistir=True, tohum=7).bol(X)]
        assert a == b
        assert a != c

    def test_tohum_karistirmadan_verilirse_hata(self):
        with pytest.raises(ValueError, match="karistir=True"):
            KKat(5, tohum=42)

    def test_kat_sayisi_sinirlari(self):
        with pytest.raises(ValueError, match="en az 2"):
            KKat(1)
        with pytest.raises(ValueError, match="büyük olamaz"):
            list(KKat(10).bol(np.arange(5).reshape(-1, 1)))

    def test_sirali_veride_karistirmamak_tehlikeli(self):
        """Veri sınıfa göre sıralıysa düz K-Fold tek sınıflı test katı üretir."""
        y = np.array([0] * 10 + [1] * 10)
        X = np.arange(20).reshape(-1, 1)
        kat_siniflari = [set(y[test].tolist()) for _, test in KKat(2).bol(X)]
        assert kat_siniflari == [{0}, {1}], "her kat tek sınıflı, felaket"


class TestTabakaliKKat:
    @pytest.fixture
    def dengesiz(self):
        """100 örnek, %20 pozitif."""
        y = np.array([1] * 20 + [0] * 80)
        X = np.arange(100).reshape(-1, 1)
        return X, y

    def test_sozlesme(self, dengesiz):
        X, y = dengesiz
        _sozlesmeyi_dogrula(TabakaliKKat(5), X, y)

    def test_sinif_orani_her_katta_korunur(self, dengesiz):
        X, y = dengesiz
        for _, test in TabakaliKKat(5).bol(X, y):
            assert float(np.mean(y[test])) == pytest.approx(0.20, abs=0.02)

    def test_her_katta_iki_sinif_da_var(self, dengesiz):
        X, y = dengesiz
        for _, test in TabakaliKKat(5).bol(X, y):
            assert set(y[test].tolist()) == {0, 1}

    def test_duz_kfold_azinlik_sinifini_kaybedebilir(self):
        """Ders notunun asıl mesajı: dengesiz veride tabakalama şart.

        %4 pozitifli 50 örnekte, sınıfa göre sıralı veri düz K-Fold'da
        hiç pozitif içermeyen test katları üretir; o katlarda recall
        tanımsızdır ve skor ortalaması anlamını yitirir.
        """
        y = np.array([1, 1] + [0] * 48)
        X = np.arange(50).reshape(-1, 1)

        duz_pozitif_sayilari = [int(y[test].sum()) for _, test in KKat(5).bol(X)]
        assert duz_pozitif_sayilari.count(0) >= 3, "çoğu kat pozitifsiz"

        tabakali_sayilar = [int(y[test].sum())
                            for _, test in TabakaliKKat(2).bol(X, y)]
        assert all(sayi >= 1 for sayi in tabakali_sayilar)

    def test_y_zorunlu(self, dengesiz):
        X, _ = dengesiz
        with pytest.raises(ValueError, match="y .*zorunludur"):
            list(TabakaliKKat(5).bol(X))

    def test_yetersiz_sinif_ornegi_aciklayici_hata(self):
        y = np.array([1, 0, 0, 0, 0, 0])
        X = np.arange(6).reshape(-1, 1)
        with pytest.raises(ValueError, match="yetersiz"):
            list(TabakaliKKat(3).bol(X, y))

    def test_cok_sinifli_tabakalama(self):
        y = np.array([0] * 30 + [1] * 20 + [2] * 10)
        X = np.arange(60).reshape(-1, 1)
        for _, test in TabakaliKKat(5).bol(X, y):
            sayimlar = np.bincount(y[test], minlength=3)
            assert sayimlar.tolist() == [6, 4, 2]


class TestBirDisarida:
    def test_n_tur_yapar(self):
        X = np.arange(7).reshape(-1, 1)
        bolmeler = _sozlesmeyi_dogrula(BirDisarida(), X)
        assert len(bolmeler) == 7

    def test_her_turda_tek_test_ornegi(self):
        X = np.arange(7).reshape(-1, 1)
        for egitim, test in BirDisarida().bol(X):
            assert test.size == 1
            assert egitim.size == 6

    def test_her_ornek_bir_kez_test_olur(self):
        X = np.arange(7).reshape(-1, 1)
        testler = np.concatenate([t for _, t in BirDisarida().bol(X)])
        assert np.array_equal(np.sort(testler), np.arange(7))

    def test_tek_ornekli_metrik_ya_sifir_ya_bir(self):
        """LOOCV'nin gerçek zayıflığı: kat başına skor ikili olur, varyans anlamsızdır."""
        from gozetimli.metrikler.siniflandirma import dogruluk
        y = np.array([0, 1, 1, 0, 1])
        X = np.arange(5).reshape(-1, 1)
        skorlar = []
        for _, test in BirDisarida().bol(X):
            # Her zaman 1 tahmin eden aptal model
            skorlar.append(dogruluk(y[test], np.ones(1, dtype=int)))
        assert set(skorlar) <= {0.0, 1.0}
        assert float(np.mean(skorlar)) == pytest.approx(0.6)


class TestPDisarida:
    def test_kombinasyon_sayisi(self):
        from math import comb
        X = np.arange(6).reshape(-1, 1)
        bolucu = PDisarida(2)
        assert bolucu.kat_sayisi(X) == comb(6, 2) == 15
        assert len(list(bolucu.bol(X))) == 15

    def test_test_boyutu_p(self):
        X = np.arange(6).reshape(-1, 1)
        for egitim, test in PDisarida(3).bol(X):
            assert test.size == 3
            assert egitim.size == 3

    def test_p_1_loocv_ile_ayni(self):
        X = np.arange(5).reshape(-1, 1)
        p1 = [(e.tolist(), t.tolist()) for e, t in PDisarida(1).bol(X)]
        loo = [(e.tolist(), t.tolist()) for e, t in BirDisarida().bol(X)]
        assert p1 == loo

    def test_guvenlik_siniri_patlamayi_engeller(self):
        """C(50, 5) = 2 118 760, sessizce çalışmaya başlamamalı."""
        X = np.arange(50).reshape(-1, 1)
        with pytest.raises(ValueError, match="güvenlik sınırı"):
            list(PDisarida(5).bol(X))

    def test_guvenlik_siniri_bilinerek_yukseltilebilir(self):
        X = np.arange(12).reshape(-1, 1)
        bolucu = PDisarida(3, guvenlik_siniri=1000)
        assert len(list(bolucu.bol(X))) == 220


class TestZamanSerisiBolme:
    def test_egitim_her_zaman_testten_once(self):
        X = np.arange(30).reshape(-1, 1)
        for egitim, test in ZamanSerisiBolme(5).bol(X):
            assert egitim.max() < test.min(), "gelecek eğitime sızmamalı"

    def test_genisleyen_pencere_buyur(self):
        X = np.arange(30).reshape(-1, 1)
        boyutlar = [len(e) for e, _ in ZamanSerisiBolme(5, pencere="genisleyen").bol(X)]
        assert boyutlar == sorted(boyutlar)
        assert boyutlar[0] < boyutlar[-1]

    def test_kayan_pencere_sabit_kalir(self):
        X = np.arange(30).reshape(-1, 1)
        boyutlar = [len(e) for e, _ in ZamanSerisiBolme(5, pencere="kayan").bol(X)]
        assert len(set(boyutlar)) == 1, "kayan pencerede eğitim uzunluğu sabit"

    def test_kayan_pencere_eski_veriyi_atar(self):
        X = np.arange(30).reshape(-1, 1)
        baslangiclar = [int(e.min()) for e, _ in
                        ZamanSerisiBolme(5, pencere="kayan").bol(X)]
        assert baslangiclar == sorted(baslangiclar)
        assert baslangiclar[0] < baslangiclar[-1]

    def test_bosluk_sizintiyi_engeller(self):
        """bosluk=3 -> eğitim sonu ile test başı arasında 3 örnek atlanır."""
        X = np.arange(40).reshape(-1, 1)
        for egitim, test in ZamanSerisiBolme(4, bosluk=3).bol(X):
            assert test.min() - egitim.max() > 3

    def test_test_bloklari_ardisik_ve_ayrik(self):
        X = np.arange(30).reshape(-1, 1)
        testler = [t for _, t in ZamanSerisiBolme(5).bol(X)]
        for test in testler:
            assert np.array_equal(test, np.arange(test.min(), test.max() + 1))
        for once, sonra in zip(testler, testler[1:]):
            assert once.max() < sonra.min()

    def test_ilk_ornekler_asla_test_edilmez(self):
        """Bu yüzden zaman serisinde out-of-fold tahmin tüm veriyi kapsamaz."""
        X = np.arange(30).reshape(-1, 1)
        testler = np.concatenate([t for _, t in ZamanSerisiBolme(5).bol(X)])
        assert 0 not in testler.tolist()
        assert len(testler) < 30

    def test_yetersiz_veri_aciklayici_hata(self):
        with pytest.raises(ValueError, match="yetersiz"):
            list(ZamanSerisiBolme(10).bol(np.arange(5).reshape(-1, 1)))

    def test_gecersiz_pencere_adi(self):
        with pytest.raises(ValueError, match="pencere"):
            ZamanSerisiBolme(3, pencere="dairesel")

    def test_kfold_zaman_serisinde_gelecegi_sizdirir(self):
        """Karşıt gösterim: K-Fold eğitim setine test sonrası tarihleri koyar."""
        X = np.arange(30).reshape(-1, 1)
        sizinti_sayisi = sum(
            1 for egitim, test in KKat(5).bol(X) if egitim.max() > test.min())
        assert sizinti_sayisi >= 4, "K-Fold katlarının çoğunda gelecek sızıyor"

        for egitim, test in ZamanSerisiBolme(5).bol(X):
            assert egitim.max() < test.min()


class TestGrupKKat:
    @pytest.fixture
    def gruplu(self):
        """6 hasta, her birinden 5 ölçüm = 30 satır."""
        gruplar = np.repeat(np.arange(6), 5)
        X = np.arange(30).reshape(-1, 1)
        return X, gruplar

    def test_sozlesme(self, gruplu):
        X, gruplar = gruplu
        _sozlesmeyi_dogrula(GrupKKat(3), X, gruplar=gruplar)

    def test_grup_egitim_ve_testte_birlikte_bulunmaz(self, gruplu):
        X, gruplar = gruplu
        for egitim, test in GrupKKat(3).bol(X, gruplar=gruplar):
            assert set(gruplar[egitim].tolist()).isdisjoint(gruplar[test].tolist())

    def test_her_grup_bir_kez_test_olur(self, gruplu):
        X, gruplar = gruplu
        test_gruplari = []
        for _, test in GrupKKat(3).bol(X, gruplar=gruplar):
            test_gruplari.extend(set(gruplar[test].tolist()))
        assert sorted(test_gruplari) == list(range(6))

    def test_kfold_grup_sizintisina_yol_acar(self):
        """Karşıt gösterim: düz K-Fold aynı hastayı iki tarafa birden koyar."""
        gruplar = np.repeat(np.arange(6), 5)
        X = np.arange(30).reshape(-1, 1)
        sizan = 0
        for egitim, test in KKat(3, karistir=True, tohum=0).bol(X):
            ortak = set(gruplar[egitim].tolist()) & set(gruplar[test].tolist())
            if ortak:
                sizan += 1
        assert sizan == 3, "her katta sızıntı var"

    def test_dengesiz_gruplar_dengelenir(self):
        """Grup boyutları çok farklıysa katlar yine de makul dengelenir."""
        gruplar = np.array([0] * 20 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5)
        X = np.arange(40).reshape(-1, 1)
        boyutlar = [len(t) for _, t in GrupKKat(3).bol(X, gruplar=gruplar)]
        assert sum(boyutlar) == 40
        assert max(boyutlar) <= 20

    def test_gruplar_zorunlu(self, gruplu):
        X, _ = gruplu
        with pytest.raises(ValueError, match="gruplar .*zorunludur"):
            list(GrupKKat(3).bol(X))

    def test_yetersiz_grup_sayisi(self):
        gruplar = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="Grup sayısı"):
            list(GrupKKat(3).bol(np.arange(4).reshape(-1, 1), gruplar=gruplar))


class TestTekrarliKKat:
    def test_toplam_bolme_sayisi(self):
        X = np.arange(20).reshape(-1, 1)
        bolucu = TekrarliKKat(5, tekrar=3, tohum=0)
        assert bolucu.kat_sayisi(X) == 15
        assert len(list(bolucu.bol(X))) == 15

    def test_tekrarlar_farkli_bolmeler_uretir(self):
        X = np.arange(20).reshape(-1, 1)
        bolmeler = [tuple(t.tolist()) for _, t in TekrarliKKat(5, tekrar=2,
                                                               tohum=0).bol(X)]
        assert len(set(bolmeler)) > 5, "iki tekrar aynı bölmeleri vermemeli"

    def test_tabakali_tekrar(self):
        y = np.array([1] * 20 + [0] * 80)
        X = np.arange(100).reshape(-1, 1)
        for _, test in TekrarliKKat(5, tekrar=2, tohum=1, tabakali=True).bol(X, y):
            assert float(np.mean(y[test])) == pytest.approx(0.20, abs=0.05)

    def test_tohum_tekrarlanabilir(self):
        X = np.arange(20).reshape(-1, 1)
        a = [t.tolist() for _, t in TekrarliKKat(5, tekrar=2, tohum=3).bol(X)]
        b = [t.tolist() for _, t in TekrarliKKat(5, tekrar=2, tohum=3).bol(X)]
        assert a == b


class TestEgitimTestBol:
    def test_oran(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        X_e, X_t, y_e, y_t = egitim_test_bol(X, y, test_orani=0.25, tohum=0)
        assert len(X_t) == 25 and len(X_e) == 75
        assert len(y_t) == 25 and len(y_e) == 75

    def test_x_y_hizasi_korunur(self):
        """Bölme sonrası X ve y satırları eşleşmeye devam etmeli."""
        X = np.arange(50).reshape(-1, 1)
        y = X.ravel() * 10
        X_e, X_t, y_e, y_t = egitim_test_bol(X, y, test_orani=0.2, tohum=1)
        assert np.array_equal(X_e.ravel() * 10, y_e)
        assert np.array_equal(X_t.ravel() * 10, y_t)

    def test_ayrik_ve_eksiksiz(self):
        X = np.arange(40).reshape(-1, 1)
        X_e, X_t = egitim_test_bol(X, test_orani=0.3, tohum=2)
        birlesim = np.sort(np.concatenate([X_e.ravel(), X_t.ravel()]))
        assert np.array_equal(birlesim, np.arange(40))

    def test_tabakalama_orani_korur(self):
        y = np.array([1] * 20 + [0] * 80)
        X = np.arange(100).reshape(-1, 1)
        _, _, y_e, y_t = egitim_test_bol(X, y, test_orani=0.2, tohum=0, tabaka=y)
        assert float(np.mean(y_t)) == pytest.approx(0.2, abs=0.01)
        assert float(np.mean(y_e)) == pytest.approx(0.2, abs=0.01)

    def test_tabakalamasiz_oran_kayabilir(self):
        """Küçük ve dengesiz veride tabakalamamak test setini bozabilir."""
        y = np.array([1] * 5 + [0] * 45)
        X = np.arange(50).reshape(-1, 1)
        oranlar = []
        for tohum in range(20):
            _, _, _, y_t = egitim_test_bol(X, y, test_orani=0.2, tohum=tohum)
            oranlar.append(float(np.mean(y_t)))
        assert min(oranlar) == 0.0, "bazı bölmelerde test setinde hiç pozitif yok"

    def test_tohum_tekrarlanabilir(self):
        X = np.arange(30).reshape(-1, 1)
        a = egitim_test_bol(X, test_orani=0.2, tohum=5)[1].tolist()
        b = egitim_test_bol(X, test_orani=0.2, tohum=5)[1].tolist()
        assert a == b

    def test_ucgen_bolme_60_20_20(self):
        """Ders notundaki train/validation/test ayrımı."""
        X = np.arange(100).reshape(-1, 1)
        y = np.array([1] * 50 + [0] * 50)
        X_gecici, X_test, y_gecici, y_test = egitim_test_bol(
            X, y, test_orani=0.2, tohum=0, tabaka=y)
        X_egitim, X_dog, y_egitim, y_dog = egitim_test_bol(
            X_gecici, y_gecici, test_orani=0.25, tohum=0, tabaka=y_gecici)
        assert (len(X_egitim), len(X_dog), len(X_test)) == (60, 20, 20)
        assert len(np.intersect1d(X_egitim, X_test)) == 0
        assert len(np.intersect1d(X_dog, X_test)) == 0

    def test_gecersiz_oran(self):
        X = np.arange(10).reshape(-1, 1)
        with pytest.raises(ValueError, match="test_orani"):
            egitim_test_bol(X, test_orani=1.5)
        with pytest.raises(ValueError, match="test_orani"):
            egitim_test_bol(X, test_orani=0.0)

    def test_uzunluk_uyusmazligi(self):
        with pytest.raises(ValueError, match="uzunluğu"):
            egitim_test_bol(np.arange(10), np.arange(9))

    def test_tek_ornekli_sinifta_tabakalama_hata_verir(self):
        y = np.array([1] + [0] * 9)
        X = np.arange(10).reshape(-1, 1)
        with pytest.raises(ValueError, match="en az 2 örnek"):
            egitim_test_bol(X, y, test_orani=0.2, tabaka=y)


class TestSklearnUyumluTakmaAdlar:
    """Bölücüler scikit-learn fonksiyonlarına cv= olarak geçirilebilmeli."""

    @pytest.mark.parametrize("bolucu", [KKat(3), TabakaliKKat(3),
                                        ZamanSerisiBolme(3), BirDisarida()])
    def test_split_ve_get_n_splits_var(self, bolucu):
        X = np.arange(30).reshape(-1, 1)
        y = np.array([0, 1] * 15)
        assert list(bolucu.split(X, y)) is not None
        assert bolucu.get_n_splits(X, y) == bolucu.kat_sayisi(X, y)

    def test_repr_okunabilir(self):
        assert "KKat(" in repr(KKat(5, karistir=True, tohum=1))
        assert "karistir=True" in repr(KKat(5, karistir=True, tohum=1))
