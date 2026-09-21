"""Çapraz doğrulama çalıştırıcıları: skor toplama, OOF tahmin, iç içe CV.

Buradaki fonksiyonlar `bolme.py` içindeki bölücüleri alır ve bir modeli
kat kat eğitip değerlendirir. Model arayüzü scikit-learn sözleşmesidir:
`fit(X, y)` ve `predict(X)` (olasılık gerekiyorsa `predict_proba`).

Neden kendi çalıştırıcımız var?
    scikit-learn'ün `cross_validate`'i zaten iyi. Bu modül onu taklit etmek
    için değil, ÇAPRAZ DOĞRULAMANIN İÇİNİ görünür kılmak için var: her katın
    skoru, eğitim skoru (aşırı öğrenme farkı), kat kat dağılım. `ozet()`
    çıktısı ortalamanın yanında standart sapmayı da verir — tek bir ortalama
    sayı, modelin kararlı olup olmadığını gizler.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

__all__ = [
    "klonla",
    "CaprazSonuc",
    "capraz_dogrula",
    "capraz_tahmin",
    "ic_ice_capraz_dogrula",
    "ogrenme_egrisi",
]


def klonla(model):
    """Modelin eğitilmemiş bir kopyasını üretir.

    scikit-learn varsa `sklearn.base.clone` kullanılır (hiperparametreleri
    kopyalar, öğrenilmiş durumu atar). Yoksa derin kopyaya düşülür.
    """
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        return deepcopy(model)


def _indeksle(veri, indeksler):
    """NumPy dizisi veya pandas DataFrame/Series fark etmeksizin satır seçer."""
    if hasattr(veri, "iloc"):
        return veri.iloc[indeksler]
    return np.asarray(veri)[indeksler]


class CaprazSonuc:
    """Kat kat skorları tutan basit kap."""

    def __init__(self, kat_skorlari: dict, egitim_skorlari: dict | None = None,
                 sureler: list | None = None):
        self.kat_skorlari = {ad: np.asarray(d, dtype=float)
                             for ad, d in kat_skorlari.items()}
        self.egitim_skorlari = ({ad: np.asarray(d, dtype=float)
                                 for ad, d in egitim_skorlari.items()}
                                if egitim_skorlari else {})
        self.sureler = sureler or []

    def ozet(self, basamak: int = 4) -> dict:
        """Metrik başına ortalama, standart sapma ve min/max."""
        cikti = {}
        for ad, skorlar in self.kat_skorlari.items():
            cikti[ad] = {
                "ortalama": round(float(np.mean(skorlar)), basamak),
                "std": round(float(np.std(skorlar, ddof=1))
                             if skorlar.size > 1 else 0.0, basamak),
                "min": round(float(np.min(skorlar)), basamak),
                "max": round(float(np.max(skorlar)), basamak),
                "katlar": [round(float(s), basamak) for s in skorlar],
            }
            if ad in self.egitim_skorlari:
                egitim_ort = float(np.mean(self.egitim_skorlari[ad]))
                cikti[ad]["egitim_ortalama"] = round(egitim_ort, basamak)
                # Pozitif fark = eğitimde daha iyi = aşırı öğrenme sinyali.
                cikti[ad]["asiri_ogrenme_farki"] = round(
                    egitim_ort - float(np.mean(skorlar)), basamak)
        return cikti

    def __repr__(self):
        parcalar = [f"{ad}={np.mean(s):.4f}±{np.std(s, ddof=1) if s.size > 1 else 0:.4f}"
                    for ad, s in self.kat_skorlari.items()]
        return f"CaprazSonuc({', '.join(parcalar)})"


def capraz_dogrula(model, X, y, *, bolucu, metrikler, gruplar=None,
                   egitim_skoru: bool = False, olasilik_metrikleri=()):
    """Modeli kat kat eğitip her katta verilen metrikleri hesaplar.

    metrikler: {"ad": fonksiyon(y_gercek, y_tahmin)} sözlüğü.
    olasilik_metrikleri: sert tahmin yerine `predict_proba`nın pozitif sınıf
        sütununu isteyen metriklerin adları (ör. roc_auc, log_kaybi).

    KRİTİK: ölçekleme, doldurma, özellik seçimi gibi tüm ÖĞRENEN adımlar
    `model` nesnesinin içinde (Pipeline olarak) olmalıdır. Bunları döngü
    dışında tüm veriye uygularsan test katının bilgisi eğitime sızar ve
    skorlar gerçekte olmayan bir başarıyı gösterir.
    """
    import time

    kat_skorlari = {ad: [] for ad in metrikler}
    egitim_skorlari = {ad: [] for ad in metrikler} if egitim_skoru else None
    sureler = []

    for egitim_idx, test_idx in bolucu.bol(X, y, gruplar):
        X_egitim, X_test = _indeksle(X, egitim_idx), _indeksle(X, test_idx)
        y_egitim, y_test = _indeksle(y, egitim_idx), _indeksle(y, test_idx)

        kat_modeli = klonla(model)
        baslangic = time.perf_counter()
        kat_modeli.fit(X_egitim, y_egitim)
        sureler.append(time.perf_counter() - baslangic)

        tahmin_test = kat_modeli.predict(X_test)
        olasilik_test = None
        if olasilik_metrikleri and hasattr(kat_modeli, "predict_proba"):
            olasilik_test = kat_modeli.predict_proba(X_test)[:, 1]

        for ad, fonksiyon in metrikler.items():
            girdi = (olasilik_test if ad in olasilik_metrikleri
                     and olasilik_test is not None else tahmin_test)
            kat_skorlari[ad].append(float(fonksiyon(y_test, girdi)))

        if egitim_skoru:
            tahmin_egitim = kat_modeli.predict(X_egitim)
            olasilik_egitim = (kat_modeli.predict_proba(X_egitim)[:, 1]
                               if olasilik_metrikleri
                               and hasattr(kat_modeli, "predict_proba") else None)
            for ad, fonksiyon in metrikler.items():
                girdi = (olasilik_egitim if ad in olasilik_metrikleri
                         and olasilik_egitim is not None else tahmin_egitim)
                egitim_skorlari[ad].append(float(fonksiyon(y_egitim, girdi)))

    return CaprazSonuc(kat_skorlari, egitim_skorlari, sureler)


def capraz_tahmin(model, X, y, *, bolucu, gruplar=None, olasilik: bool = False):
    """Her örnek için, o örneği GÖRMEMİŞ bir modelden gelen tahmin (out-of-fold).

    Kullanım alanları:
      - Dürüst bir karmaşıklık matrisi / ROC eğrisi çizmek.
      - Eşik seçimi için tüm veriyi kullanabilmek (test setini harcamadan).
      - Stacking (yığınlama) için ikinci seviye modelin girdisini üretmek.

    Uyarı: dönen tahminler tek bir modele ait değildir; k farklı modelin
    birleşimidir. Bu yüzden OOF skoru, `capraz_dogrula`nın kat ortalamasından
    biraz farklı çıkabilir.
    """
    n = np.asarray(X).shape[0] if not hasattr(X, "iloc") else len(X)
    tahminler = None
    dolduruldu = np.zeros(n, dtype=bool)

    for egitim_idx, test_idx in bolucu.bol(X, y, gruplar):
        kat_modeli = klonla(model)
        kat_modeli.fit(_indeksle(X, egitim_idx), _indeksle(y, egitim_idx))
        if olasilik:
            kat_tahmin = kat_modeli.predict_proba(_indeksle(X, test_idx))[:, 1]
        else:
            kat_tahmin = kat_modeli.predict(_indeksle(X, test_idx))
        kat_tahmin = np.asarray(kat_tahmin)
        if tahminler is None:
            tahminler = np.empty(n, dtype=kat_tahmin.dtype)
        tahminler[test_idx] = kat_tahmin
        dolduruldu[test_idx] = True

    if not dolduruldu.all():
        eksik = int((~dolduruldu).sum())
        raise ValueError(
            f"{eksik} örnek hiçbir test katına düşmedi; bu bölücüyle "
            "out-of-fold tahmin üretilemez (ör. ZamanSerisiBolme ilk bloğu "
            "asla test etmez)."
        )
    return tahminler


def ic_ice_capraz_dogrula(model_kurucu, X, y, *, dis_bolucu, ic_bolucu,
                          izgara, metrik, gruplar=None, buyuk_iyi: bool = True):
    """İç içe (nested) çapraz doğrulama — hiperparametre seçimi dahil dürüst skor.

    Neden gerekli?
        Hiperparametreyi tüm veri üzerinde CV ile seçip sonra aynı CV skorunu
        "modelin performansı" diye raporlamak, seçim sürecinin bilgisini skora
        sızdırır. İyimser sapma, çok denenen ızgaralarda birkaç puana çıkar.

    Nasıl çalışır?
        Dış döngü: veriyi eğitim/test diye böler — skor buradan gelir.
        İç döngü: yalnızca dış eğitim parçasında en iyi hiperparametreyi seçer.

    model_kurucu: parametre sözlüğü alıp model döndüren çağrılabilir nesne.
    izgara: denenecek parametre sözlüklerinin listesi.

    Dönüş: (dis_skorlar, secilen_parametreler)
    """
    dis_skorlar, secilen = [], []

    for egitim_idx, test_idx in dis_bolucu.bol(X, y, gruplar):
        X_egitim, X_test = _indeksle(X, egitim_idx), _indeksle(X, test_idx)
        y_egitim, y_test = _indeksle(y, egitim_idx), _indeksle(y, test_idx)
        ic_gruplar = _indeksle(gruplar, egitim_idx) if gruplar is not None else None

        en_iyi_skor, en_iyi_param = None, None
        for parametreler in izgara:
            ic_skorlar = []
            for ic_egitim, ic_test in ic_bolucu.bol(X_egitim, y_egitim, ic_gruplar):
                aday = model_kurucu(**parametreler)
                aday.fit(_indeksle(X_egitim, ic_egitim), _indeksle(y_egitim, ic_egitim))
                ic_skorlar.append(float(metrik(
                    _indeksle(y_egitim, ic_test),
                    aday.predict(_indeksle(X_egitim, ic_test)))))
            ortalama = float(np.mean(ic_skorlar))
            daha_iyi = (en_iyi_skor is None
                        or (ortalama > en_iyi_skor if buyuk_iyi
                            else ortalama < en_iyi_skor))
            if daha_iyi:
                en_iyi_skor, en_iyi_param = ortalama, parametreler

        nihai = model_kurucu(**en_iyi_param)
        nihai.fit(X_egitim, y_egitim)
        dis_skorlar.append(float(metrik(y_test, nihai.predict(X_test))))
        secilen.append(en_iyi_param)

    return np.asarray(dis_skorlar), secilen


def ogrenme_egrisi(model, X, y, *, bolucu, metrik, oranlar=(0.1, 0.25, 0.5, 0.75, 1.0),
                   gruplar=None, tohum: int | None = 0):
    """Eğitim seti büyüdükçe eğitim ve doğrulama skorlarının değişimi.

    Teşhis tablosu:
      - İki eğri de düşük ve birbirine yakın  -> yüksek bias (eksik öğrenme):
        daha karmaşık model, daha iyi özellik. Veri eklemek işe yaramaz.
      - Eğitim yüksek, doğrulama düşük, arada kalıcı boşluk -> yüksek varyans
        (aşırı öğrenme): düzenlileştirme (Ridge/Lasso), budama, daha çok veri.
      - Boşluk veri arttıkça kapanıyorsa -> veri eklemek gerçekten işe yarar.

    Dönüş: (kullanilan_ornek_sayilari, egitim_skorlari, dogrulama_skorlari)
    her skor dizisi (oran_sayisi, kat_sayisi) şeklindedir.
    """
    uretec = np.random.default_rng(tohum)
    oranlar = np.asarray(oranlar, dtype=float)
    if np.any((oranlar <= 0) | (oranlar > 1)):
        raise ValueError("oranlar (0, 1] aralığında olmalı.")

    egitim_skorlari, dogrulama_skorlari, boyutlar = [], [], []
    bolmeler = list(bolucu.bol(X, y, gruplar))

    for oran in oranlar:
        satir_egitim, satir_dogrulama, kullanilan = [], [], []
        for egitim_idx, test_idx in bolmeler:
            adet = max(2, int(round(len(egitim_idx) * oran)))
            secilen = uretec.choice(egitim_idx, size=adet, replace=False)
            kat_modeli = klonla(model)
            kat_modeli.fit(_indeksle(X, secilen), _indeksle(y, secilen))
            satir_egitim.append(float(metrik(_indeksle(y, secilen),
                                             kat_modeli.predict(_indeksle(X, secilen)))))
            satir_dogrulama.append(float(metrik(_indeksle(y, test_idx),
                                                kat_modeli.predict(_indeksle(X, test_idx)))))
            kullanilan.append(adet)
        egitim_skorlari.append(satir_egitim)
        dogrulama_skorlari.append(satir_dogrulama)
        boyutlar.append(int(np.mean(kullanilan)))

    return (np.asarray(boyutlar), np.asarray(egitim_skorlari),
            np.asarray(dogrulama_skorlari))
