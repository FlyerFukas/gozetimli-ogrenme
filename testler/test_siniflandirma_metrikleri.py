"""Sınıflandırma metriklerinin birim testleri.

Beklenen değerler elle hesaplanmıştır; her testin docstring'inde hesabın
kendisi yazılıdır ki bir test düştüğünde "acaba beklenen mi yanlış?" sorusu
kodun içinden yanıtlanabilsin.
"""

from __future__ import annotations

import numpy as np
import pytest

from gozetimli.metrikler.siniflandirma import (cohen_kappa, dengeli_dogruluk,
                                               dogruluk, esik_tara,
                                               f1_skoru, f_beta_skoru,
                                               ikili_bilesenler,
                                               karmasiklik_matrisi, kesinlik,
                                               kesinlik_duyarlilik_egrisi,
                                               log_kaybi,
                                               matthews_korelasyonu,
                                               ortalama_kesinlik, ozgulluk,
                                               roc_auc, roc_egrisi,
                                               siniflandirma_raporu,
                                               duyarlilik)


@pytest.fixture
def ikili_ornek():
    """10 örnek: TP=3, TN=4, FP=2, FN=1.

        gerçek: 1 1 1 1 0 0 0 0 0 0
        tahmin: 1 1 1 0 1 1 0 0 0 0

    Buradan: accuracy = 7/10, precision = 3/5, recall = 3/4,
             F1 = 2·0.6·0.75/(0.6+0.75) = 2/3, specificity = 4/6.
    """
    gercek = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    tahmin = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    return gercek, tahmin


class TestKarmasiklikMatrisi:
    def test_temel_matris(self, ikili_ornek):
        """[[TN, FP], [FN, TP]] = [[4, 2], [1, 3]]."""
        gercek, tahmin = ikili_ornek
        matris = karmasiklik_matrisi(gercek, tahmin)
        assert matris.tolist() == [[4, 2], [1, 3]]
        assert matris.sum() == 10

    def test_ikili_bilesenler(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        assert ikili_bilesenler(gercek, tahmin) == (4, 2, 1, 3)

    def test_kosegen_dogru_tahminler(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        matris = karmasiklik_matrisi(gercek, tahmin)
        assert np.trace(matris) == int(np.sum(gercek == tahmin)) == 7

    def test_etiket_sirasi_korunur(self):
        """etiketler parametresi satır/sütun sırasını belirler."""
        gercek = np.array(["kedi", "kopek", "kedi"])
        tahmin = np.array(["kedi", "kedi", "kedi"])
        m1 = karmasiklik_matrisi(gercek, tahmin, etiketler=["kedi", "kopek"])
        m2 = karmasiklik_matrisi(gercek, tahmin, etiketler=["kopek", "kedi"])
        assert m1.tolist() == [[2, 0], [1, 0]]
        assert m2.tolist() == [[0, 1], [0, 2]]

    def test_normalize_gercek_kosegen_recall_verir(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        matris = karmasiklik_matrisi(gercek, tahmin, normalize="gercek")
        assert matris[1, 1] == pytest.approx(duyarlilik(gercek, tahmin))
        assert matris.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

    def test_normalize_tahmin_kosegen_precision_verir(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        matris = karmasiklik_matrisi(gercek, tahmin, normalize="tahmin")
        assert matris[1, 1] == pytest.approx(kesinlik(gercek, tahmin))

    def test_gecersiz_normalize_hata_verir(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        with pytest.raises(ValueError, match="normalize"):
            karmasiklik_matrisi(gercek, tahmin, normalize="yanlis")

    def test_uzunluk_uyusmazligi_hata_verir(self):
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            karmasiklik_matrisi([0, 1, 1], [0, 1])


class TestTemelMetrikler:
    def test_dogruluk(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        assert dogruluk(gercek, tahmin) == pytest.approx(0.7)
        assert dogruluk(gercek, tahmin, normalize=False) == 7

    def test_kesinlik(self, ikili_ornek):
        """TP/(TP+FP) = 3/5 = 0.6."""
        gercek, tahmin = ikili_ornek
        assert kesinlik(gercek, tahmin) == pytest.approx(0.6)

    def test_duyarlilik(self, ikili_ornek):
        """TP/(TP+FN) = 3/4 = 0.75."""
        gercek, tahmin = ikili_ornek
        assert duyarlilik(gercek, tahmin) == pytest.approx(0.75)

    def test_ozgulluk(self, ikili_ornek):
        """TN/(TN+FP) = 4/6 ≈ 0.667."""
        gercek, tahmin = ikili_ornek
        assert ozgulluk(gercek, tahmin) == pytest.approx(2 / 3)

    def test_f1_harmonik_ortalamadir(self, ikili_ornek):
        """F1 = 2PR/(P+R) = 2(0.6)(0.75)/1.35 = 0.6667."""
        gercek, tahmin = ikili_ornek
        p, r = 0.6, 0.75
        assert f1_skoru(gercek, tahmin) == pytest.approx(2 * p * r / (p + r))
        assert f1_skoru(gercek, tahmin) == pytest.approx(2 / 3)

    def test_f1_aritmetik_ortalamadan_kucuktur(self, ikili_ornek):
        """Harmonik ortalama her zaman aritmetikten küçük veya eşittir.

        F1'in varlık sebebi budur: P=1.0, R=0.01 olan bir model aritmetik
        ortalamada 0.5 alırken F1'de 0.0198 alır.
        """
        gercek, tahmin = ikili_ornek
        assert f1_skoru(gercek, tahmin) < (0.6 + 0.75) / 2

        p, r = 1.0, 0.01
        assert 2 * p * r / (p + r) == pytest.approx(0.0198, abs=1e-4)

    @pytest.mark.parametrize("beta,beklenen", [
        (0.5, 0.625),      # precision ağırlıklı: (1.25·0.6·0.75)/(0.25·0.6+0.75)
        (1.0, 2 / 3),
        (2.0, 0.7143),     # recall ağırlıklı: (5·0.6·0.75)/(4·0.6+0.75)
    ])
    def test_f_beta_agirliklari(self, ikili_ornek, beta, beklenen):
        gercek, tahmin = ikili_ornek
        assert f_beta_skoru(gercek, tahmin, beta=beta) == pytest.approx(
            beklenen, abs=1e-4)

    def test_f_beta_sinirlari(self, ikili_ornek):
        """beta -> 0 precision'a, beta -> ∞ recall'a yakınsar."""
        gercek, tahmin = ikili_ornek
        assert f_beta_skoru(gercek, tahmin, beta=0.001) == pytest.approx(0.6, abs=1e-5)
        assert f_beta_skoru(gercek, tahmin, beta=1000) == pytest.approx(0.75, abs=1e-5)


class TestDengesizVeri:
    """Accuracy'nin yalan söylediği, diğer metriklerin söylemediği durum."""

    @pytest.fixture
    def cogunluk_tahmincisi(self):
        """100 örnek, 2 pozitif. Model her şeye 0 diyor."""
        gercek = np.array([1, 1] + [0] * 98)
        tahmin = np.zeros(100, dtype=int)
        return gercek, tahmin

    def test_accuracy_yaniltici(self, cogunluk_tahmincisi):
        """Hiçbir pozitifi yakalamayan model %98 accuracy alır."""
        gercek, tahmin = cogunluk_tahmincisi
        assert dogruluk(gercek, tahmin) == pytest.approx(0.98)

    def test_recall_gercegi_soyler(self, cogunluk_tahmincisi):
        gercek, tahmin = cogunluk_tahmincisi
        assert duyarlilik(gercek, tahmin) == 0.0
        assert f1_skoru(gercek, tahmin) == 0.0

    def test_dengeli_dogruluk_gercegi_soyler(self, cogunluk_tahmincisi):
        """Sınıf recall'larının ortalaması: (1.0 + 0.0)/2 = 0.5 = şans seviyesi."""
        gercek, tahmin = cogunluk_tahmincisi
        assert dengeli_dogruluk(gercek, tahmin) == pytest.approx(0.5)

    def test_mcc_ve_kappa_sifir_verir(self, cogunluk_tahmincisi):
        """Şanstan farksız model için MCC = 0, kappa = 0."""
        gercek, tahmin = cogunluk_tahmincisi
        assert matthews_korelasyonu(gercek, tahmin) == pytest.approx(0.0)
        assert cohen_kappa(gercek, tahmin) == pytest.approx(0.0)

    def test_mukemmel_tahmin(self):
        gercek = np.array([0, 1, 1, 0, 1])
        assert matthews_korelasyonu(gercek, gercek) == pytest.approx(1.0)
        assert cohen_kappa(gercek, gercek) == pytest.approx(1.0)
        assert f1_skoru(gercek, gercek) == pytest.approx(1.0)

    def test_tam_ters_tahmin_mcc_eksi_bir(self):
        gercek = np.array([0, 1, 1, 0, 1, 0])
        assert matthews_korelasyonu(gercek, 1 - gercek) == pytest.approx(-1.0)


class TestCokSinifliOrtalamalar:
    @pytest.fixture
    def uc_sinif(self):
        """3 sınıf, 9 örnek. Sınıf 0: 4 örnek, sınıf 1: 3, sınıf 2: 2.

        gerçek: 0 0 0 0 1 1 1 2 2
        tahmin: 0 0 0 1 1 1 2 2 0

        Sınıf 0: TP=3, FP=1, FN=1  -> P=0.75,  R=0.75,  F1=0.75
        Sınıf 1: TP=2, FP=1, FN=1  -> P=2/3,   R=2/3,   F1=2/3
        Sınıf 2: TP=1, FP=1, FN=1  -> P=0.5,   R=0.5,   F1=0.5
        """
        gercek = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
        tahmin = np.array([0, 0, 0, 1, 1, 1, 2, 2, 0])
        return gercek, tahmin

    def test_sinif_basina_skorlar(self, uc_sinif):
        gercek, tahmin = uc_sinif
        assert kesinlik(gercek, tahmin, ortalama=None) == pytest.approx(
            [0.75, 2 / 3, 0.5])
        assert duyarlilik(gercek, tahmin, ortalama=None) == pytest.approx(
            [0.75, 2 / 3, 0.5])

    def test_makro_duz_ortalamadir(self, uc_sinif):
        """(0.75 + 0.6667 + 0.5)/3 = 0.6389."""
        gercek, tahmin = uc_sinif
        assert kesinlik(gercek, tahmin, ortalama="makro") == pytest.approx(
            0.6389, abs=1e-4)

    def test_agirlikli_destekle_ortalar(self, uc_sinif):
        """(4·0.75 + 3·0.6667 + 2·0.5)/9 = 0.6667."""
        gercek, tahmin = uc_sinif
        assert kesinlik(gercek, tahmin, ortalama="agirlikli") == pytest.approx(
            (4 * 0.75 + 3 * (2 / 3) + 2 * 0.5) / 9)

    def test_mikro_accuracy_ile_ayni(self, uc_sinif):
        """Çok sınıflı tek etiketli problemde mikro P = mikro R = accuracy."""
        gercek, tahmin = uc_sinif
        beklenen = dogruluk(gercek, tahmin)
        assert kesinlik(gercek, tahmin, ortalama="mikro") == pytest.approx(beklenen)
        assert duyarlilik(gercek, tahmin, ortalama="mikro") == pytest.approx(beklenen)
        assert f1_skoru(gercek, tahmin, ortalama="mikro") == pytest.approx(beklenen)

    def test_makro_f1_makro_p_r_den_hesaplanamaz(self, uc_sinif):
        """Sık yapılan hata: makro F1 ≠ 2·makroP·makroR/(makroP+makroR).

        Bu örnekte P = R olduğu için iki değer çakışır; farkı göstermek için
        P ve R'nin ayrıştığı ikinci bir örnek kullanılır.
        """
        gercek = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        tahmin = np.array([0, 0, 0, 1, 1, 1, 0, 2])
        makro_p = kesinlik(gercek, tahmin, ortalama="makro")
        makro_r = duyarlilik(gercek, tahmin, ortalama="makro")
        makro_f = f1_skoru(gercek, tahmin, ortalama="makro")
        yanlis_yol = 2 * makro_p * makro_r / (makro_p + makro_r)
        assert makro_f != pytest.approx(yanlis_yol, abs=1e-6)

    def test_ikili_ortalama_cok_siniflida_hata_verir(self, uc_sinif):
        gercek, tahmin = uc_sinif
        with pytest.raises(ValueError, match="iki sınıflı"):
            kesinlik(gercek, tahmin, ortalama="ikili")

    def test_bilinmeyen_pozitif_etiket_hata_verir(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        with pytest.raises(ValueError, match="veride yok"):
            kesinlik(gercek, tahmin, pozitif_etiket=7)


class TestSifiraBolme:
    def test_hic_pozitif_tahmin_yoksa_precision_sifir(self):
        """TP+FP = 0 -> precision tanımsız; sifir_bolme ile ne döneceği seçilir."""
        gercek = np.array([0, 1, 1])
        tahmin = np.array([0, 0, 0])
        assert kesinlik(gercek, tahmin) == 0.0
        assert kesinlik(gercek, tahmin, sifir_bolme=1.0) == 1.0

    def test_hic_pozitif_gercek_yoksa_recall_sifir(self):
        gercek = np.array([0, 0, 0])
        tahmin = np.array([0, 1, 0])
        assert duyarlilik(gercek, tahmin, ortalama=None)[1] == 0.0


class TestLogKaybi:
    def test_mukemmel_tahmin_sifira_yakin(self):
        gercek = np.array([0, 1, 1, 0])
        olasilik = np.array([0.0, 1.0, 1.0, 0.0])
        assert log_kaybi(gercek, olasilik) == pytest.approx(0.0, abs=1e-12)

    def test_kararsiz_tahmin_ln2_verir(self):
        """Her şeye 0.5 diyen model: -ln(0.5) = 0.6931."""
        gercek = np.array([0, 1, 1, 0])
        assert log_kaybi(gercek, np.full(4, 0.5)) == pytest.approx(
            np.log(2), abs=1e-9)

    def test_emin_ve_yanlis_agir_cezalanir(self):
        """%99 güvenle yanlış olmak, %51 güvenle yanlış olmaktan çok pahalı."""
        gercek = np.array([1])
        az_yanlis = log_kaybi(gercek, np.array([0.49]))
        cok_yanlis = log_kaybi(gercek, np.array([0.01]))
        assert cok_yanlis > az_yanlis * 6

    def test_accuracy_ayni_ama_log_kaybi_farkli(self):
        """İki model aynı accuracy'ye sahip olup farklı kalibrasyona sahip olabilir."""
        gercek = np.array([0, 0, 1, 1])
        emin = np.array([0.05, 0.05, 0.95, 0.95])
        kararsiz = np.array([0.45, 0.45, 0.55, 0.55])
        sert_emin = (emin > 0.5).astype(int)
        sert_kararsiz = (kararsiz > 0.5).astype(int)
        assert dogruluk(gercek, sert_emin) == dogruluk(gercek, sert_kararsiz) == 1.0
        assert log_kaybi(gercek, emin) < log_kaybi(gercek, kararsiz)

    def test_matris_bicimi_cok_sinifli(self):
        gercek = np.array([0, 1, 2])
        olasiliklar = np.array([[0.8, 0.1, 0.1],
                                [0.1, 0.8, 0.1],
                                [0.1, 0.1, 0.8]])
        assert log_kaybi(gercek, olasiliklar) == pytest.approx(-np.log(0.8))

    def test_bicim_hatalari(self):
        with pytest.raises(ValueError, match="satır sayısı"):
            log_kaybi([0, 1], np.array([[0.5, 0.5]]))
        with pytest.raises(ValueError, match="sütun sayısı"):
            log_kaybi([0, 1, 2], np.array([[0.5, 0.5]] * 3))


class TestEgriler:
    @pytest.fixture
    def skorlar(self):
        """4 pozitif, 4 negatif; skorlar kısmen örtüşüyor."""
        gercek = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        skor = np.array([0.9, 0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.1])
        return gercek, skor

    def test_roc_sifirdan_baslar_bire_biter(self, skorlar):
        gercek, skor = skorlar
        fpr, tpr, _ = roc_egrisi(gercek, skor)
        assert (fpr[0], tpr[0]) == (0.0, 0.0)
        assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
        assert np.all(np.diff(fpr) >= 0)
        assert np.all(np.diff(tpr) >= 0)

    def test_mukemmel_ayrim_auc_bir(self):
        gercek = np.array([0, 0, 1, 1])
        skor = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc(gercek, skor) == pytest.approx(1.0)

    def test_tam_ters_ayrim_auc_sifir(self):
        gercek = np.array([0, 0, 1, 1])
        skor = np.array([0.9, 0.8, 0.2, 0.1])
        assert roc_auc(gercek, skor) == pytest.approx(0.0)

    def test_rastgele_skor_auc_yarim(self):
        """Tüm skorlar eşitse eğri köşegendir -> AUC = 0.5."""
        gercek = np.array([0, 1, 0, 1])
        assert roc_auc(gercek, np.full(4, 0.5)) == pytest.approx(0.5)

    def test_auc_siralamaya_duyarli_degere_degil(self, skorlar):
        """AUC yalnızca sıralamaya bakar: monoton dönüşüm sonucu değiştirmez."""
        gercek, skor = skorlar
        assert roc_auc(gercek, skor) == pytest.approx(roc_auc(gercek, skor * 100))
        assert roc_auc(gercek, skor) == pytest.approx(roc_auc(gercek, np.log(skor)))

    def test_auc_mann_whitney_ile_ayni(self, skorlar):
        """AUC = P(pozitif skoru > negatif skoru), beraberlikler yarım sayılır."""
        gercek, skor = skorlar
        poz, neg = skor[gercek == 1], skor[gercek == 0]
        fark = poz[:, None] - neg[None, :]
        beklenen = float((np.sum(fark > 0) + 0.5 * np.sum(fark == 0))
                         / (poz.size * neg.size))
        assert roc_auc(gercek, skor) == pytest.approx(beklenen)

    def test_pr_egrisi_sonu_bir_sifir(self, skorlar):
        gercek, skor = skorlar
        kesinlikler, duyarliliklar, esikler = kesinlik_duyarlilik_egrisi(gercek, skor)
        assert kesinlikler[-1] == 1.0
        assert duyarliliklar[-1] == 0.0
        assert len(esikler) == len(kesinlikler) - 1

    def test_ortalama_kesinlik_taban_cizgisi_pozitif_oranidir(self):
        """Rastgele skorlarda AP ≈ pozitif sınıf oranı. ROC-AUC ise 0.5 kalır."""
        uretec = np.random.default_rng(0)
        gercek = np.zeros(2000, dtype=int)
        gercek[:100] = 1                      # %5 pozitif
        uretec.shuffle(gercek)
        skor = uretec.random(2000)
        assert ortalama_kesinlik(gercek, skor) == pytest.approx(0.05, abs=0.02)
        assert roc_auc(gercek, skor) == pytest.approx(0.5, abs=0.05)

    def test_mukemmel_ayrimda_ap_bir(self):
        gercek = np.array([0, 0, 1, 1])
        assert ortalama_kesinlik(gercek, np.array([0.1, 0.2, 0.8, 0.9])) == \
            pytest.approx(1.0)

    def test_tek_sinifli_veride_hata(self):
        with pytest.raises(ValueError, match="hem pozitif hem negatif"):
            roc_auc(np.ones(5, dtype=int), np.linspace(0, 1, 5))


class TestEsikTarama:
    @pytest.fixture
    def olasiliklar(self):
        gercek = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        skor = np.array([0.1, 0.2, 0.25, 0.3, 0.45, 0.55, 0.35, 0.6, 0.7, 0.9])
        return gercek, skor

    def test_f1_maksimize_eden_esik(self, olasiliklar):
        gercek, skor = olasiliklar
        esik, en_iyi_f1, _, _ = esik_tara(gercek, skor, metrik="f1")
        varsayilan_f1 = f1_skoru(gercek, (skor >= 0.5).astype(int))
        secilen_f1 = f1_skoru(gercek, (skor >= esik).astype(int))
        assert secilen_f1 == pytest.approx(en_iyi_f1)
        assert secilen_f1 >= varsayilan_f1

    def test_0_5_esigi_her_zaman_en_iyi_degil(self, olasiliklar):
        """Varsayılan 0.5 eşiği bu veride F1'i düşürüyor."""
        gercek, skor = olasiliklar
        esik, en_iyi, _, _ = esik_tara(gercek, skor, metrik="f1")
        assert esik != pytest.approx(0.5)
        assert en_iyi > f1_skoru(gercek, (skor >= 0.5).astype(int))

    def test_recall_kisiti_altinda_precision(self, olasiliklar):
        """"Recall en az %100 olsun" kısıtı, eşiği aşağı çeker."""
        gercek, skor = olasiliklar
        esik, _, _, _ = esik_tara(gercek, skor, metrik="kesinlik",
                                  en_az_duyarlilik=1.0)
        tahmin = (skor >= esik).astype(int)
        assert duyarlilik(gercek, tahmin) == pytest.approx(1.0)

    def test_saglanamayan_kisit_hata_verir(self, olasiliklar):
        gercek, skor = olasiliklar
        with pytest.raises(ValueError, match="sağlayan eşik yok"):
            esik_tara(gercek, skor, en_az_duyarlilik=1.0, en_az_kesinlik=1.0)

    def test_f2_esigi_f0_5_esiginden_kucuk(self, olasiliklar):
        """Recall'a ağırlık veren F2 daha düşük eşik seçer (daha çok pozitif)."""
        gercek, skor = olasiliklar
        esik_f2, _, _, _ = esik_tara(gercek, skor, metrik="f2")
        esik_f05, _, _, _ = esik_tara(gercek, skor, metrik="f0.5")
        assert esik_f2 <= esik_f05

    def test_youden_indeksi(self, olasiliklar):
        gercek, skor = olasiliklar
        esik, skor_degeri, _, _ = esik_tara(gercek, skor, metrik="youden")
        tahmin = (skor >= esik).astype(int)
        beklenen = duyarlilik(gercek, tahmin) - (1 - ozgulluk(gercek, tahmin))
        assert skor_degeri == pytest.approx(beklenen)

    def test_gecersiz_metrik_adi(self, olasiliklar):
        gercek, skor = olasiliklar
        with pytest.raises(ValueError):
            esik_tara(gercek, skor, metrik="bilinmeyen")


class TestRapor:
    def test_rapor_anahtarlari(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        rapor = siniflandirma_raporu(gercek, tahmin)
        assert set(rapor) == {0, 1, "dogruluk", "makro ortalama", "agirlikli ortalama"}
        assert rapor[1]["kesinlik"] == pytest.approx(0.6)
        assert rapor[1]["duyarlilik"] == pytest.approx(0.75)
        assert rapor[1]["destek"] == 4
        assert rapor[0]["destek"] == 6

    def test_destek_toplami_ornek_sayisi(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        rapor = siniflandirma_raporu(gercek, tahmin)
        assert rapor[0]["destek"] + rapor[1]["destek"] == len(gercek)

    def test_metin_bicimi(self, ikili_ornek):
        gercek, tahmin = ikili_ornek
        metin = siniflandirma_raporu(gercek, tahmin, metin=True)
        assert "kesinlik" in metin and "duyarlilik" in metin
        assert "makro ortalama" in metin
        assert len(metin.splitlines()) == 7  # başlık + 2 sınıf + boş + 3 özet
