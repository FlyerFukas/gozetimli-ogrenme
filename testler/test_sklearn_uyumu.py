"""Bu kütüphanenin sayılarının scikit-learn ile birebir aynı olduğunu doğrular.

Neden gerekli?
    Metrik formülleri kâğıt üzerinde basit görünür ama ayrıntıda ayrışırlar:
    makro F1'in nasıl ortalandığı, PR eğrisinin hangi sırada döndüğü, sıfıra
    bölmede ne yapıldığı, AP'nin trapez mi basamak mı toplandığı. Bu dosya,
    "bizim precision" ile "herkesin precision"ının aynı sayı olduğunu her
    koşuda kanıtlar.

Rastgele üretilmiş onlarca senaryo üzerinde çalışır (tohumlar sabittir, yani
tekrarlanabilir). Bir uyuşmazlık çıkarsa hangi senaryoda çıktığı bellidir.
"""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.dogrulama.bolme import (BirDisarida, KKat, TabakaliKKat,
                                       ZamanSerisiBolme)
from gozetimli.metrikler import regresyon as bizim_reg
from gozetimli.metrikler import siniflandirma as bizim_sinif
from gozetimli.saflik import entropi, gini

sklearn = pytest.importorskip("sklearn")
pytestmark = pytest.mark.sklearn

from sklearn import metrics as sk  # noqa: E402
from sklearn.model_selection import (KFold, LeaveOneOut,  # noqa: E402
                                     StratifiedKFold, TimeSeriesSplit)


def _ikili_senaryolar(adet=25):
    """Farklı denge, boyut ve zorluktaki ikili sınıflandırma senaryoları."""
    senaryolar = []
    for tohum in range(adet):
        uretec = np.random.default_rng(tohum)
        n = int(uretec.integers(20, 300))
        pozitif_orani = float(uretec.uniform(0.05, 0.95))
        gercek = (uretec.random(n) < pozitif_orani).astype(int)
        if gercek.sum() in (0, n):          # tek sınıflı senaryoyu atla
            continue
        gurultu = float(uretec.uniform(0.0, 1.5))
        skor = gercek + uretec.normal(0, gurultu, n)
        tahmin = (skor > 0.5).astype(int)
        senaryolar.append((f"tohum={tohum},n={n},oran={pozitif_orani:.2f}",
                           gercek, tahmin, skor))
    return senaryolar


def _cok_sinifli_senaryolar(adet=15):
    senaryolar = []
    for tohum in range(adet):
        uretec = np.random.default_rng(100 + tohum)
        n = int(uretec.integers(30, 200))
        k = int(uretec.integers(3, 6))
        gercek = uretec.integers(0, k, n)
        tahmin = np.where(uretec.random(n) < 0.6, gercek, uretec.integers(0, k, n))
        if np.unique(gercek).size < k:
            continue
        senaryolar.append((f"tohum={tohum},n={n},k={k}", gercek, tahmin))
    return senaryolar


IKILI = _ikili_senaryolar()
COK_SINIFLI = _cok_sinifli_senaryolar()


class TestIkiliSiniflandirma:
    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_karmasiklik_matrisi(self, ad, gercek, tahmin, skor):
        assert np.array_equal(bizim_sinif.karmasiklik_matrisi(gercek, tahmin),
                              sk.confusion_matrix(gercek, tahmin)), ad

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_ikili_bilesenler(self, ad, gercek, tahmin, skor):
        assert bizim_sinif.ikili_bilesenler(gercek, tahmin) == \
            tuple(int(v) for v in sk.confusion_matrix(gercek, tahmin).ravel()), ad

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_temel_metrikler(self, ad, gercek, tahmin, skor):
        assert bizim_sinif.dogruluk(gercek, tahmin) == pytest.approx(
            sk.accuracy_score(gercek, tahmin)), ad
        assert bizim_sinif.kesinlik(gercek, tahmin) == pytest.approx(
            sk.precision_score(gercek, tahmin, zero_division=0)), ad
        assert bizim_sinif.duyarlilik(gercek, tahmin) == pytest.approx(
            sk.recall_score(gercek, tahmin, zero_division=0)), ad
        assert bizim_sinif.f1_skoru(gercek, tahmin) == pytest.approx(
            sk.f1_score(gercek, tahmin, zero_division=0)), ad

    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 3.0])
    def test_f_beta(self, beta):
        for ad, gercek, tahmin, _ in IKILI[:10]:
            assert bizim_sinif.f_beta_skoru(gercek, tahmin, beta=beta) == \
                pytest.approx(sk.fbeta_score(gercek, tahmin, beta=beta,
                                             zero_division=0)), f"{ad}, beta={beta}"

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_dengeli_dogruluk_mcc_kappa(self, ad, gercek, tahmin, skor):
        assert bizim_sinif.dengeli_dogruluk(gercek, tahmin) == pytest.approx(
            sk.balanced_accuracy_score(gercek, tahmin)), ad
        assert bizim_sinif.matthews_korelasyonu(gercek, tahmin) == pytest.approx(
            sk.matthews_corrcoef(gercek, tahmin), abs=1e-10), ad
        assert bizim_sinif.cohen_kappa(gercek, tahmin) == pytest.approx(
            sk.cohen_kappa_score(gercek, tahmin), abs=1e-10), ad

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_roc_auc(self, ad, gercek, tahmin, skor):
        assert bizim_sinif.roc_auc(gercek, skor) == pytest.approx(
            sk.roc_auc_score(gercek, skor)), ad

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_roc_egrisi(self, ad, gercek, tahmin, skor):
        bizim_fpr, bizim_tpr, _ = bizim_sinif.roc_egrisi(gercek, skor)
        sk_fpr, sk_tpr, _ = sk.roc_curve(gercek, skor, drop_intermediate=False)
        assert bizim_fpr.tolist() == pytest.approx(sk_fpr.tolist()), ad
        assert bizim_tpr.tolist() == pytest.approx(sk_tpr.tolist()), ad

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_ortalama_kesinlik(self, ad, gercek, tahmin, skor):
        assert bizim_sinif.ortalama_kesinlik(gercek, skor) == pytest.approx(
            sk.average_precision_score(gercek, skor)), ad

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_pr_egrisi(self, ad, gercek, tahmin, skor):
        bizim_k, bizim_d, bizim_e = bizim_sinif.kesinlik_duyarlilik_egrisi(
            gercek, skor)
        sk_k, sk_d, sk_e = sk.precision_recall_curve(gercek, skor)
        assert bizim_k.tolist() == pytest.approx(sk_k.tolist()), ad
        assert bizim_d.tolist() == pytest.approx(sk_d.tolist()), ad
        assert bizim_e.tolist() == pytest.approx(sk_e.tolist()), ad

    @pytest.mark.parametrize("ad,gercek,tahmin,skor", IKILI, ids=lambda v: None)
    def test_log_kaybi(self, ad, gercek, tahmin, skor):
        olasilik = 1 / (1 + np.exp(-skor))
        assert bizim_sinif.log_kaybi(gercek, olasilik) == pytest.approx(
            sk.log_loss(gercek, olasilik, labels=[0, 1])), ad


class TestCokSinifliOrtalamalar:
    ESLESME = {"makro": "macro", "mikro": "micro", "agirlikli": "weighted"}

    @pytest.mark.parametrize("bizimki,sklearninki", list(ESLESME.items()))
    def test_kesinlik_duyarlilik_f1(self, bizimki, sklearninki):
        for ad, gercek, tahmin in COK_SINIFLI:
            assert bizim_sinif.kesinlik(gercek, tahmin, ortalama=bizimki) == \
                pytest.approx(sk.precision_score(gercek, tahmin,
                                                 average=sklearninki,
                                                 zero_division=0)), ad
            assert bizim_sinif.duyarlilik(gercek, tahmin, ortalama=bizimki) == \
                pytest.approx(sk.recall_score(gercek, tahmin,
                                              average=sklearninki,
                                              zero_division=0)), ad
            assert bizim_sinif.f1_skoru(gercek, tahmin, ortalama=bizimki) == \
                pytest.approx(sk.f1_score(gercek, tahmin, average=sklearninki,
                                          zero_division=0)), ad

    def test_sinif_basina_skorlar(self):
        for ad, gercek, tahmin in COK_SINIFLI:
            assert bizim_sinif.kesinlik(gercek, tahmin, ortalama=None) == \
                pytest.approx(sk.precision_score(gercek, tahmin, average=None,
                                                 zero_division=0)), ad
            assert bizim_sinif.duyarlilik(gercek, tahmin, ortalama=None) == \
                pytest.approx(sk.recall_score(gercek, tahmin, average=None,
                                              zero_division=0)), ad

    def test_karmasiklik_matrisi(self):
        for ad, gercek, tahmin in COK_SINIFLI:
            assert np.array_equal(
                bizim_sinif.karmasiklik_matrisi(gercek, tahmin),
                sk.confusion_matrix(gercek, tahmin)), ad

    def test_mcc_ve_kappa(self):
        for ad, gercek, tahmin in COK_SINIFLI:
            assert bizim_sinif.matthews_korelasyonu(gercek, tahmin) == \
                pytest.approx(sk.matthews_corrcoef(gercek, tahmin), abs=1e-10), ad
            assert bizim_sinif.cohen_kappa(gercek, tahmin) == \
                pytest.approx(sk.cohen_kappa_score(gercek, tahmin), abs=1e-10), ad

    def test_normalize_secenekleri(self):
        for ad, gercek, tahmin in COK_SINIFLI[:5]:
            for bizimki, sklearninki in (("gercek", "true"), ("tahmin", "pred"),
                                         ("tumu", "all")):
                # Matris karşılaştırması: pytest.approx iç içe diziyi desteklemez.
                np.testing.assert_allclose(
                    bizim_sinif.karmasiklik_matrisi(gercek, tahmin,
                                                    normalize=bizimki),
                    sk.confusion_matrix(gercek, tahmin, normalize=sklearninki),
                    rtol=1e-12, err_msg=f"{ad}, normalize={bizimki}")

    def test_rapor_sklearn_raporuyla_ayni_sayilar(self):
        for ad, gercek, tahmin in COK_SINIFLI[:5]:
            bizim = bizim_sinif.siniflandirma_raporu(gercek, tahmin)
            sk_rapor = sk.classification_report(gercek, tahmin, output_dict=True,
                                                zero_division=0)
            for etiket in np.unique(gercek).tolist():
                assert bizim[etiket]["kesinlik"] == pytest.approx(
                    sk_rapor[str(etiket)]["precision"], abs=1e-4), ad
                assert bizim[etiket]["duyarlilik"] == pytest.approx(
                    sk_rapor[str(etiket)]["recall"], abs=1e-4), ad
                assert bizim[etiket]["destek"] == sk_rapor[str(etiket)]["support"], ad
            assert bizim["makro ortalama"]["f1"] == pytest.approx(
                sk_rapor["macro avg"]["f1-score"], abs=1e-4), ad


class TestRegresyon:
    @staticmethod
    def _senaryolar(adet=25):
        cikti = []
        for tohum in range(adet):
            uretec = np.random.default_rng(200 + tohum)
            n = int(uretec.integers(20, 300))
            olcek = float(10 ** uretec.integers(0, 4))
            gercek = uretec.normal(50, 20, n) * olcek
            tahmin = gercek + uretec.normal(0, float(uretec.uniform(1, 30)), n) * olcek
            cikti.append((f"tohum={tohum},n={n},olcek={olcek}", gercek, tahmin))
        return cikti

    SENARYOLAR = _senaryolar.__func__()

    def test_mse_rmse_mae(self):
        for ad, gercek, tahmin in self.SENARYOLAR:
            assert bizim_reg.mse(gercek, tahmin) == pytest.approx(
                sk.mean_squared_error(gercek, tahmin)), ad
            assert bizim_reg.rmse(gercek, tahmin) == pytest.approx(
                sk.root_mean_squared_error(gercek, tahmin)), ad
            assert bizim_reg.mae(gercek, tahmin) == pytest.approx(
                sk.mean_absolute_error(gercek, tahmin)), ad

    def test_r2_ve_aciklanan_varyans(self):
        for ad, gercek, tahmin in self.SENARYOLAR:
            assert bizim_reg.r2(gercek, tahmin) == pytest.approx(
                sk.r2_score(gercek, tahmin)), ad
            assert bizim_reg.aciklanan_varyans(gercek, tahmin) == pytest.approx(
                sk.explained_variance_score(gercek, tahmin)), ad

    def test_medyan_ve_maksimum(self):
        for ad, gercek, tahmin in self.SENARYOLAR:
            assert bizim_reg.medyan_mutlak_hata(gercek, tahmin) == pytest.approx(
                sk.median_absolute_error(gercek, tahmin)), ad
            assert bizim_reg.maksimum_hata(gercek, tahmin) == pytest.approx(
                sk.max_error(gercek, tahmin)), ad

    def test_mape(self):
        for ad, gercek, tahmin in self.SENARYOLAR:
            if np.any(np.isclose(gercek, 0)):
                continue
            assert bizim_reg.mape(gercek, tahmin) == pytest.approx(
                sk.mean_absolute_percentage_error(gercek, tahmin)), ad

    def test_msle(self):
        for ad, gercek, tahmin in self.SENARYOLAR[:10]:
            g, t = np.abs(gercek), np.abs(tahmin)
            assert bizim_reg.msle(g, t) == pytest.approx(
                sk.mean_squared_log_error(g, t)), ad

    def test_ornek_agirlikli_mse(self):
        uretec = np.random.default_rng(7)
        gercek, tahmin = uretec.normal(size=50), uretec.normal(size=50)
        agirliklar = uretec.uniform(0.1, 5, 50)
        assert bizim_reg.mse(gercek, tahmin, ornek_agirliklari=agirliklar) == \
            pytest.approx(sk.mean_squared_error(gercek, tahmin,
                                                sample_weight=agirliklar))


class TestBolucuUyumu:
    """Bölücülerimiz scikit-learn karşılıklarıyla aynı katları üretmeli."""

    def test_kfold_karistirmadan(self):
        X = np.arange(37).reshape(-1, 1)
        bizim = [(e.tolist(), t.tolist()) for e, t in KKat(5).bol(X)]
        sk_bolucu = [(e.tolist(), t.tolist()) for e, t in KFold(5).split(X)]
        assert bizim == sk_bolucu

    @pytest.mark.parametrize("n,k", [(10, 2), (20, 3), (37, 5), (100, 7)])
    def test_kfold_kat_boyutlari(self, n, k):
        X = np.arange(n).reshape(-1, 1)
        bizim = [len(t) for _, t in KKat(k).bol(X)]
        sk_boyutlar = [len(t) for _, t in KFold(k).split(X)]
        assert bizim == sk_boyutlar

    def test_leave_one_out(self):
        X = np.arange(15).reshape(-1, 1)
        bizim = [t.tolist() for _, t in BirDisarida().bol(X)]
        sk_bolucu = [t.tolist() for _, t in LeaveOneOut().split(X)]
        assert bizim == sk_bolucu

    def test_time_series_split(self):
        """Varsayılan ayarlarda TimeSeriesSplit ile aynı blokları üretmeli."""
        X = np.arange(50).reshape(-1, 1)
        bizim = [(e.tolist(), t.tolist()) for e, t in ZamanSerisiBolme(5).bol(X)]
        sk_bolucu = [(e.tolist(), t.tolist()) for e, t in TimeSeriesSplit(5).split(X)]
        assert bizim == sk_bolucu

    def test_stratified_kfold_oranlari(self):
        """İndeksler birebir aynı olmayabilir, ama sınıf oranları aynı korunmalı."""
        y = np.array([0] * 70 + [1] * 30)
        X = np.arange(100).reshape(-1, 1)
        bizim_oranlar = sorted(round(float(np.mean(y[t])), 4)
                               for _, t in TabakaliKKat(5).bol(X, y))
        sk_oranlar = sorted(round(float(np.mean(y[t])), 4)
                            for _, t in StratifiedKFold(5).split(X, y))
        assert bizim_oranlar == sk_oranlar

    def test_bolucu_sklearn_fonksiyonuna_gecirilebilir(self):
        """cv= parametresi olarak doğrudan kullanılabilmeli."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        uretec = np.random.default_rng(0)
        X = uretec.normal(size=(100, 3))
        y = (X[:, 0] > 0).astype(int)
        skorlar = cross_val_score(LogisticRegression(), X, y,
                                  cv=TabakaliKKat(5, karistir=True, tohum=0))
        assert len(skorlar) == 5
        assert skorlar.mean() > 0.8


class TestSaflikOlcutleri:
    """Entropi ve Gini'nin scipy/sklearn tanımlarıyla uyumu."""

    def test_entropi_scipy_ile_ayni(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        uretec = np.random.default_rng(0)
        for _ in range(20):
            y = uretec.integers(0, 4, size=int(uretec.integers(10, 100)))
            _, sayimlar = np.unique(y, return_counts=True)
            beklenen = float(scipy_stats.entropy(sayimlar, base=2))
            assert entropi(y) == pytest.approx(beklenen)

    def test_gini_sklearn_agaciyla_ayni(self):
        """sklearn'ün karar ağacı kök düğümündeki impurity değeri ile karşılaştır."""
        from sklearn.tree import DecisionTreeClassifier
        uretec = np.random.default_rng(1)
        for _ in range(10):
            n = int(uretec.integers(20, 100))
            X = uretec.normal(size=(n, 2))
            y = uretec.integers(0, 3, n)
            if np.unique(y).size < 2:
                continue
            agac = DecisionTreeClassifier(criterion="gini", max_depth=1,
                                          random_state=0).fit(X, y)
            assert gini(y) == pytest.approx(float(agac.tree_.impurity[0]))

    def test_entropi_sklearn_agaciyla_ayni(self):
        from sklearn.tree import DecisionTreeClassifier
        uretec = np.random.default_rng(2)
        for _ in range(10):
            n = int(uretec.integers(20, 100))
            X = uretec.normal(size=(n, 2))
            y = uretec.integers(0, 3, n)
            if np.unique(y).size < 2:
                continue
            agac = DecisionTreeClassifier(criterion="entropy", max_depth=1,
                                          random_state=0).fit(X, y)
            assert entropi(y) == pytest.approx(float(agac.tree_.impurity[0]))
