"""Çapraz doğrulama çalıştırıcılarının testleri.

Bu dosyanın en önemli kısmı `TestVeriSizintisi`: yanlış yapılan çapraz
doğrulamanın nasıl gerçek dışı skorlar ürettiğini ÖLÇEREK gösterir. Bir
kütüphanenin "sızıntıyı önler" demesi yeterli değildir; sızıntının olduğu ve
olmadığı iki kurulumun skor farkı test edilebilir olmalıdır.
"""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.dogrulama.bolme import (GrupKKat, KKat, TabakaliKKat,
                                       ZamanSerisiBolme)
from gozetimli.dogrulama.capraz import (capraz_dogrula, capraz_tahmin,
                                        ic_ice_capraz_dogrula, klonla,
                                        ogrenme_egrisi)
from gozetimli.metrikler.regresyon import mse, r2
from gozetimli.metrikler.siniflandirma import dogruluk, f1_skoru, roc_auc

sklearn = pytest.importorskip("sklearn", reason="Bu testler scikit-learn gerektirir")

from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


@pytest.fixture
def siniflandirma_verisi():
    """200 örnek, 4 özellik, ayrılabilir ama mükemmel değil."""
    uretec = np.random.default_rng(0)
    X = uretec.normal(size=(200, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + uretec.normal(0, 0.5, 200) > 0).astype(int)
    return X, y


@pytest.fixture
def regresyon_verisi():
    uretec = np.random.default_rng(1)
    X = uretec.normal(size=(150, 3))
    y = 2 * X[:, 0] - X[:, 1] + uretec.normal(0, 0.3, 150)
    return X, y


class TestKlonla:
    def test_hiperparametreler_korunur(self):
        model = DecisionTreeClassifier(max_depth=5, random_state=3)
        kopya = klonla(model)
        assert kopya.max_depth == 5
        assert kopya.random_state == 3
        assert kopya is not model

    def test_ogrenilmis_durum_tasinmaz(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        model = DecisionTreeClassifier(random_state=0).fit(X, y)
        assert hasattr(model, "tree_")
        assert not hasattr(klonla(model), "tree_")


class TestCaprazDogrula:
    def test_kat_basina_skor_uretir(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        sonuc = capraz_dogrula(LogisticRegression(), X, y,
                               bolucu=TabakaliKKat(5),
                               metrikler={"dogruluk": dogruluk, "f1": f1_skoru})
        assert set(sonuc.kat_skorlari) == {"dogruluk", "f1"}
        assert len(sonuc.kat_skorlari["dogruluk"]) == 5
        assert all(0 <= s <= 1 for s in sonuc.kat_skorlari["dogruluk"])

    def test_ozet_ortalama_ve_std_verir(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        ozet = capraz_dogrula(LogisticRegression(), X, y, bolucu=TabakaliKKat(5),
                              metrikler={"dogruluk": dogruluk}).ozet()
        d = ozet["dogruluk"]
        assert {"ortalama", "std", "min", "max", "katlar"} <= set(d)
        assert d["min"] <= d["ortalama"] <= d["max"]
        assert len(d["katlar"]) == 5
        assert d["ortalama"] > 0.75

    def test_std_kararsizligi_gorunur_kilar(self):
        """Tek ortalama sayı yanıltıcıdır; küçük veride katlar çok saçılır."""
        uretec = np.random.default_rng(5)
        X = uretec.normal(size=(30, 2))
        y = uretec.integers(0, 2, 30)          # hedef tamamen rastgele
        ozet = capraz_dogrula(DecisionTreeClassifier(random_state=0), X, y,
                              bolucu=TabakaliKKat(5),
                              metrikler={"dogruluk": dogruluk}).ozet()
        assert ozet["dogruluk"]["std"] > 0.05
        assert ozet["dogruluk"]["max"] - ozet["dogruluk"]["min"] > 0.15

    def test_egitim_skoru_asiri_ogrenmeyi_gosterir(self, siniflandirma_verisi):
        """Budanmamış karar ağacı eğitimde %100, testte belirgin daha düşük."""
        X, y = siniflandirma_verisi
        ozet = capraz_dogrula(DecisionTreeClassifier(random_state=0), X, y,
                              bolucu=TabakaliKKat(5),
                              metrikler={"dogruluk": dogruluk},
                              egitim_skoru=True).ozet()
        assert ozet["dogruluk"]["egitim_ortalama"] == pytest.approx(1.0)
        assert ozet["dogruluk"]["asiri_ogrenme_farki"] > 0.1

    def test_duzenlilestirme_farki_kapatir(self, siniflandirma_verisi):
        """Derinliği sınırlamak eğitim-test farkını küçültür."""
        X, y = siniflandirma_verisi
        derin = capraz_dogrula(DecisionTreeClassifier(random_state=0), X, y,
                               bolucu=TabakaliKKat(5),
                               metrikler={"dogruluk": dogruluk},
                               egitim_skoru=True).ozet()
        sig = capraz_dogrula(DecisionTreeClassifier(max_depth=3, random_state=0),
                             X, y, bolucu=TabakaliKKat(5),
                             metrikler={"dogruluk": dogruluk},
                             egitim_skoru=True).ozet()
        assert (sig["dogruluk"]["asiri_ogrenme_farki"]
                < derin["dogruluk"]["asiri_ogrenme_farki"])

    def test_olasilik_metrikleri(self, siniflandirma_verisi):
        """roc_auc sert tahmini değil olasılığı ister."""
        X, y = siniflandirma_verisi
        sonuc = capraz_dogrula(LogisticRegression(), X, y, bolucu=TabakaliKKat(5),
                               metrikler={"auc": roc_auc, "dogruluk": dogruluk},
                               olasilik_metrikleri=("auc",))
        assert float(np.mean(sonuc.kat_skorlari["auc"])) > 0.85
        # Sert tahminle hesaplanan AUC daha düşük olurdu (bilgi kaybı).
        sert = capraz_dogrula(LogisticRegression(), X, y, bolucu=TabakaliKKat(5),
                              metrikler={"auc": roc_auc})
        assert float(np.mean(sonuc.kat_skorlari["auc"])) > \
            float(np.mean(sert.kat_skorlari["auc"]))

    def test_regresyon_metrikleri(self, regresyon_verisi):
        X, y = regresyon_verisi
        ozet = capraz_dogrula(LinearRegression(), X, y, bolucu=KKat(5),
                              metrikler={"r2": r2, "mse": mse}).ozet()
        assert ozet["r2"]["ortalama"] > 0.9
        assert ozet["mse"]["ortalama"] < 0.2

    def test_sureler_kaydedilir(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        sonuc = capraz_dogrula(LogisticRegression(), X, y, bolucu=TabakaliKKat(3),
                               metrikler={"dogruluk": dogruluk})
        assert len(sonuc.sureler) == 3
        assert all(sure > 0 for sure in sonuc.sureler)

    def test_repr_okunabilir(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        metin = repr(capraz_dogrula(LogisticRegression(), X, y,
                                    bolucu=TabakaliKKat(3),
                                    metrikler={"dogruluk": dogruluk}))
        assert "dogruluk=" in metin and "±" in metin


class TestVeriSizintisi:
    """Yanlış kurulmuş çapraz doğrulamanın ürettiği sahte başarı."""

    def test_ozellik_secimi_sizintisi(self):
        """En çarpıcı örnek: hedefle İLİŞKİSİZ veride %80+ "doğruluk".

        200 örnek, 2000 tamamen rastgele özellik, rastgele etiket. Gerçek
        sinyal sıfır, dürüst skor %50 olmalı.

        YANLIŞ: en iyi 10 özelliği TÜM veriden seç, sonra CV yap. Seçim
        aşaması test katlarının etiketlerini görmüştür; şans eseri uyan
        özellikler her katta da uyar.
        DOĞRU: seçimi Pipeline'a koy — her katta yalnızca o katın eğitim
        verisinden seçilsin.
        """
        uretec = np.random.default_rng(42)
        X = uretec.normal(size=(200, 2000))
        y = uretec.integers(0, 2, size=200)

        # YANLIŞ: seçim döngü dışında, tüm veriyle
        secici = SelectKBest(f_classif, k=10).fit(X, y)
        X_sizdirilmis = secici.transform(X)
        sizintili = capraz_dogrula(LogisticRegression(max_iter=1000),
                                   X_sizdirilmis, y, bolucu=TabakaliKKat(5),
                                   metrikler={"dogruluk": dogruluk}).ozet()

        # DOĞRU: seçim Pipeline içinde, her katta yeniden
        boru = Pipeline([("sec", SelectKBest(f_classif, k=10)),
                         ("model", LogisticRegression(max_iter=1000))])
        durust = capraz_dogrula(boru, X, y, bolucu=TabakaliKKat(5),
                                metrikler={"dogruluk": dogruluk}).ozet()

        assert sizintili["dogruluk"]["ortalama"] > 0.70, "sızıntı sahte başarı üretir"
        assert durust["dogruluk"]["ortalama"] == pytest.approx(0.5, abs=0.12)
        assert sizintili["dogruluk"]["ortalama"] - durust["dogruluk"]["ortalama"] > 0.15

    def test_olcekleme_sizintisi_kucuk_ama_gercek(self):
        """Ölçekleyiciyi tüm veriye uygulamak test katının ortalamasını sızdırır.

        Etkisi özellik seçimi kadar dramatik değildir (skor farkı küçüktür),
        ama ölçekleme parametreleri test verisine bakarak hesaplanmış olur —
        bu, üretimde tekrarlanamayan bir avantajdır.
        """
        uretec = np.random.default_rng(7)
        X = uretec.normal(0, 10, size=(60, 5))
        y = (X[:, 0] > 0).astype(int)

        olcekleyici = StandardScaler().fit(X)          # TÜM veriden
        sizdirilmis_ortalama = olcekleyici.mean_

        boru = Pipeline([("olcek", StandardScaler()),
                         ("model", KNeighborsClassifier(3))])
        kat_ortalamalari = []
        for egitim_idx, _ in TabakaliKKat(5).bol(X, y):
            kat_ortalamalari.append(StandardScaler().fit(X[egitim_idx]).mean_)

        # Kat bazlı ortalamalar, tüm veriden hesaplanandan farklıdır.
        farklar = [float(np.abs(kat - sizdirilmis_ortalama).max())
                   for kat in kat_ortalamalari]
        assert max(farklar) > 0.1

        sonuc = capraz_dogrula(boru, X, y, bolucu=TabakaliKKat(5),
                               metrikler={"dogruluk": dogruluk})
        assert len(sonuc.kat_skorlari["dogruluk"]) == 5

    def test_grup_sizintisi_skoru_sisirir(self):
        """Aynı denek hem eğitimde hem testteyse model ezberi "genelleme" sanılır.

        20 denek, her birinden 10 neredeyse aynı ölçüm. Denek kimliği hedefi
        belirliyor; gerçek genelleme, GÖRÜLMEMİŞ denekte ölçülür.
        """
        uretec = np.random.default_rng(3)
        denek_sayisi, olcum = 20, 10
        gruplar = np.repeat(np.arange(denek_sayisi), olcum)
        denek_etiketi = uretec.integers(0, 2, denek_sayisi)
        y = np.repeat(denek_etiketi, olcum)
        denek_imzasi = uretec.normal(size=(denek_sayisi, 3))
        X = (np.repeat(denek_imzasi, olcum, axis=0)
             + uretec.normal(0, 0.01, size=(denek_sayisi * olcum, 3)))

        model = KNeighborsClassifier(1)
        satir_bazli = capraz_dogrula(model, X, y, bolucu=KKat(5, karistir=True,
                                                             tohum=0),
                                     metrikler={"dogruluk": dogruluk}).ozet()
        grup_bazli = capraz_dogrula(model, X, y, bolucu=GrupKKat(5),
                                    gruplar=gruplar,
                                    metrikler={"dogruluk": dogruluk}).ozet()

        assert satir_bazli["dogruluk"]["ortalama"] > 0.98, "ezber mükemmel görünür"
        assert grup_bazli["dogruluk"]["ortalama"] < 0.80
        assert (satir_bazli["dogruluk"]["ortalama"]
                - grup_bazli["dogruluk"]["ortalama"]) > 0.20

    def test_zaman_serisinde_kfold_gelecegi_sizdirir(self):
        """Trend içeren seride K-Fold, geleceği görerek geçmişi tahmin eder."""
        n = 200
        zaman = np.arange(n, dtype=float)
        uretec = np.random.default_rng(11)
        # Rejim değişimi: 100. adımdan sonra ilişki tersine dönüyor.
        y = np.where(zaman < 100, 0.5 * zaman, 50 - 0.5 * (zaman - 100))
        y = y + uretec.normal(0, 1, n)
        X = zaman.reshape(-1, 1)

        kfold = capraz_dogrula(LinearRegression(), X, y,
                               bolucu=KKat(5, karistir=True, tohum=0),
                               metrikler={"r2": r2}).ozet()
        zaman_serisi = capraz_dogrula(LinearRegression(), X, y,
                                      bolucu=ZamanSerisiBolme(5),
                                      metrikler={"r2": r2}).ozet()

        assert kfold["r2"]["ortalama"] > zaman_serisi["r2"]["ortalama"]
        assert zaman_serisi["r2"]["ortalama"] < 0, \
            "Dürüst değerlendirme modelin rejim değişimini kaçırdığını gösterir"


class TestCaprazTahmin:
    def test_her_ornek_icin_tahmin(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        tahminler = capraz_tahmin(LogisticRegression(), X, y,
                                  bolucu=TabakaliKKat(5))
        assert tahminler.shape == y.shape
        assert set(np.unique(tahminler).tolist()) <= {0, 1}

    def test_oof_skoru_kat_ortalamasina_yakin(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        bolucu = TabakaliKKat(5)
        oof = capraz_tahmin(LogisticRegression(), X, y, bolucu=bolucu)
        oof_skoru = dogruluk(y, oof)
        kat_ortalamasi = float(np.mean(capraz_dogrula(
            LogisticRegression(), X, y, bolucu=bolucu,
            metrikler={"d": dogruluk}).kat_skorlari["d"]))
        assert oof_skoru == pytest.approx(kat_ortalamasi, abs=0.02)

    def test_olasilik_tahmini(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        olasiliklar = capraz_tahmin(LogisticRegression(), X, y,
                                    bolucu=TabakaliKKat(5), olasilik=True)
        assert np.all((olasiliklar >= 0) & (olasiliklar <= 1))
        assert roc_auc(y, olasiliklar) > 0.85

    def test_zaman_serisinde_hata_verir(self, siniflandirma_verisi):
        """ZamanSerisiBolme ilk bloğu hiç test etmez -> OOF eksik kalır."""
        X, y = siniflandirma_verisi
        with pytest.raises(ValueError, match="hiçbir test katına düşmedi"):
            capraz_tahmin(LogisticRegression(), X, y, bolucu=ZamanSerisiBolme(5))


class TestIcIceCaprazDogrulama:
    def test_secilen_parametreler_kat_basina_dondurulur(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        skorlar, secilen = ic_ice_capraz_dogrula(
            lambda **p: DecisionTreeClassifier(random_state=0, **p), X, y,
            dis_bolucu=TabakaliKKat(3), ic_bolucu=TabakaliKKat(3),
            izgara=[{"max_depth": 1}, {"max_depth": 3}, {"max_depth": None}],
            metrik=dogruluk)
        assert len(skorlar) == len(secilen) == 3
        assert all("max_depth" in p for p in secilen)

    def test_nested_cv_iyimser_sapmayi_onler(self):
        """Sinyalsiz veride ızgara araması CV skorunu şişirir; nested şişirmez.

        60 örnek, 50 rastgele özellik, rastgele etiket. 8 aday arasından
        "en iyisini" aynı CV ile seçip o skoru raporlamak, seçim şansını
        performans sanmaktır.
        """
        uretec = np.random.default_rng(13)
        X = uretec.normal(size=(60, 50))
        y = uretec.integers(0, 2, 60)
        izgara = [{"max_depth": d} for d in (1, 2, 3, 4, 5, 8, 12, None)]

        # YANLIŞ: tüm veride ızgara ara, en iyi CV skorunu rapor et
        en_iyi = max(
            float(np.mean(capraz_dogrula(
                DecisionTreeClassifier(random_state=0, **p), X, y,
                bolucu=TabakaliKKat(5),
                metrikler={"d": dogruluk}).kat_skorlari["d"]))
            for p in izgara)

        # DOĞRU: seçim iç döngüde, skor dış döngüde
        nested, _ = ic_ice_capraz_dogrula(
            lambda **p: DecisionTreeClassifier(random_state=0, **p), X, y,
            dis_bolucu=TabakaliKKat(5), ic_bolucu=TabakaliKKat(4),
            izgara=izgara, metrik=dogruluk)

        assert en_iyi > float(np.mean(nested)), \
            "Izgaranın en iyisi, nested tahminden iyimserdir"
        assert float(np.mean(nested)) == pytest.approx(0.5, abs=0.15)

    def test_kucuk_iyi_metrikle_calisir(self, regresyon_verisi):
        """buyuk_iyi=False ile MSE gibi küçüğü iyi metrikler kullanılabilir."""
        X, y = regresyon_verisi
        from sklearn.tree import DecisionTreeRegressor
        skorlar, secilen = ic_ice_capraz_dogrula(
            lambda **p: DecisionTreeRegressor(random_state=0, **p), X, y,
            dis_bolucu=KKat(3), ic_bolucu=KKat(3),
            izgara=[{"max_depth": 1}, {"max_depth": 5}],
            metrik=mse, buyuk_iyi=False)
        assert len(skorlar) == 3
        assert all(s > 0 for s in skorlar)


class TestOgrenmeEgrisi:
    def test_bicim(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        boyutlar, egitim, dogrulama = ogrenme_egrisi(
            LogisticRegression(), X, y, bolucu=TabakaliKKat(4),
            metrik=dogruluk, oranlar=(0.2, 0.5, 1.0))
        assert boyutlar.shape == (3,)
        assert egitim.shape == dogrulama.shape == (3, 4)
        assert boyutlar.tolist() == sorted(boyutlar.tolist())

    def test_dogrulama_skoru_veriyle_artar(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        _, _, dogrulama = ogrenme_egrisi(
            LogisticRegression(), X, y, bolucu=TabakaliKKat(4),
            metrik=dogruluk, oranlar=(0.05, 1.0), tohum=0)
        assert dogrulama[-1].mean() > dogrulama[0].mean()

    def test_yuksek_varyans_teshisi(self, siniflandirma_verisi):
        """Budanmamış ağaç: eğitim ~1.0, doğrulama düşük -> kalıcı boşluk."""
        X, y = siniflandirma_verisi
        _, egitim, dogrulama = ogrenme_egrisi(
            DecisionTreeClassifier(random_state=0), X, y,
            bolucu=TabakaliKKat(4), metrik=dogruluk, oranlar=(0.5, 1.0))
        assert egitim.mean() == pytest.approx(1.0, abs=0.01)
        assert (egitim.mean(axis=1) - dogrulama.mean(axis=1)).min() > 0.1

    def test_yuksek_bias_teshisi(self, siniflandirma_verisi):
        """Derinliği 1 olan ağaç (stump): iki eğri de düşük ve birbirine yakın."""
        X, y = siniflandirma_verisi
        _, egitim, dogrulama = ogrenme_egrisi(
            DecisionTreeClassifier(max_depth=1, random_state=0), X, y,
            bolucu=TabakaliKKat(4), metrik=dogruluk, oranlar=(0.5, 1.0))
        bosluk = float((egitim.mean(axis=1) - dogrulama.mean(axis=1)).max())
        assert bosluk < 0.1, "eksik öğrenmede eğitim-test boşluğu küçüktür"
        assert egitim.mean() < 0.95, "ama iki eğri de tavana ulaşmaz"

    def test_gecersiz_oran(self, siniflandirma_verisi):
        X, y = siniflandirma_verisi
        with pytest.raises(ValueError, match="oranlar"):
            ogrenme_egrisi(LogisticRegression(), X, y, bolucu=TabakaliKKat(3),
                           metrik=dogruluk, oranlar=(0.5, 1.5))
