"""Sınıflandırma metrikleri, karmaşıklık matrisinden eşik taramasına.

Hepsi tek bir karmaşıklık matrisinden türetilir; böylece ikili ve çok sınıflı
durum aynı kodu paylaşır. Ortalama stratejileri scikit-learn ile birebir aynı
tanımı kullanır (`testler/test_sklearn_uyumu.py` bunu her koşuda doğrular),
fakat isimler ve hata mesajları Türkçedir.

Ortalama stratejileri
---------------------
``"ikili"``      : yalnızca `pozitif_etiket` sınıfının skoru. Varsayılan.
``"makro"``      : sınıf skorlarının düz ortalaması. Her sınıf eşit ağırlıkta:
                   azınlık sınıfı önemliyse bunu kullan.
``"agirlikli"``  : destek (support) ile ağırlıklı ortalama. Dengesiz veride
                   accuracy'ye yakınsar, yani azınlık sınıfını gizler.
``"mikro"``      : tüm TP/FP/FN'leri havuzlayıp tek skor. Çok sınıflı tek
                   etiketli problemde accuracy'ye eşittir.
``None``         : sınıf başına dizi döner.
"""

from __future__ import annotations

import numpy as np

from ..ortak import ciftleri_dogrula, diziye_cevir, etiketleri_bul, guvenli_bolme

__all__ = [
    "karmasiklik_matrisi",
    "ikili_bilesenler",
    "dogruluk",
    "kesinlik",
    "duyarlilik",
    "ozgulluk",
    "f_beta_skoru",
    "f1_skoru",
    "dengeli_dogruluk",
    "matthews_korelasyonu",
    "cohen_kappa",
    "log_kaybi",
    "roc_egrisi",
    "roc_auc",
    "kesinlik_duyarlilik_egrisi",
    "ortalama_kesinlik",
    "siniflandirma_raporu",
    "esik_tara",
]

_ORTALAMALAR = ("ikili", "makro", "mikro", "agirlikli", None)


# --------------------------------------------------------------------------
# Temel: karmaşıklık matrisi
# --------------------------------------------------------------------------
def karmasiklik_matrisi(y_gercek, y_tahmin, *, etiketler=None,
                        normalize: str | None = None) -> np.ndarray:
    """Karmaşıklık (confusion) matrisi: satır = gerçek, sütun = tahmin.

    ``C[i, j]`` = gerçekte `etiketler[i]` olup `etiketler[j]` tahmin edilen
    örnek sayısı. Köşegen doğru tahminlerdir.

    normalize:
        None      ham sayılar
        "gercek"  her satır kendi toplamına bölünür (köşegen = recall)
        "tahmin"  her sütun kendi toplamına bölünür (köşegen = precision)
        "tumu"    toplam örnek sayısına bölünür
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin)
    etiketler = etiketleri_bul(gercek, tahmin, etiketler)

    indeks = {etiket: i for i, etiket in enumerate(etiketler.tolist())}
    n = len(etiketler)
    matris = np.zeros((n, n), dtype=np.int64)
    for g, t in zip(gercek.tolist(), tahmin.tolist()):
        # `etiketler` dışarıdan kısıtlanmış olabilir; kapsam dışı örnekler
        # scikit-learn'de olduğu gibi sessizce atlanır.
        if g in indeks and t in indeks:
            matris[indeks[g], indeks[t]] += 1

    if normalize is None:
        return matris
    if normalize not in ("gercek", "tahmin", "tumu"):
        raise ValueError(
            "normalize 'gercek', 'tahmin', 'tumu' ya da None olmalı; "
            f"'{normalize}' geldi."
        )
    matris = matris.astype(float)
    if normalize == "gercek":
        return guvenli_bolme(matris, matris.sum(axis=1, keepdims=True))
    if normalize == "tahmin":
        return guvenli_bolme(matris, matris.sum(axis=0, keepdims=True))
    return guvenli_bolme(matris, matris.sum())


def ikili_bilesenler(y_gercek, y_tahmin, *, pozitif_etiket=1):
    """İkili problemde (TN, FP, FN, TP) dörtlüsünü döndürür.

    Ders notlarındaki tanımlar:
        TP, gerçek pozitif, tahmin pozitif
        TN, gerçek negatif, tahmin negatif
        FP, gerçek negatif, tahmin pozitif  (I. tip hata, "yanlış alarm")
        FN, gerçek pozitif, tahmin negatif  (II. tip hata, "kaçırılan vaka")
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin)
    g_poz = gercek == pozitif_etiket
    t_poz = tahmin == pozitif_etiket
    tp = int(np.sum(g_poz & t_poz))
    tn = int(np.sum(~g_poz & ~t_poz))
    fp = int(np.sum(~g_poz & t_poz))
    fn = int(np.sum(g_poz & ~t_poz))
    return tn, fp, fn, tp


# --------------------------------------------------------------------------
# Matristen türeyen sınıf bazlı sayımlar
# --------------------------------------------------------------------------
def _sinif_sayimlari(matris: np.ndarray):
    """Karmaşıklık matrisinden sınıf başına (TP, FP, FN, destek) çıkarır."""
    tp = np.diag(matris).astype(float)
    destek = matris.sum(axis=1).astype(float)          # gerçekte o sınıf olanlar
    tahmin_edilen = matris.sum(axis=0).astype(float)   # o sınıf tahmin edilenler
    fp = tahmin_edilen - tp
    fn = destek - tp
    return tp, fp, fn, destek


def _ortalamayi_uygula(skorlar, destek, ortalama, etiketler, pozitif_etiket,
                       *, mikro_skor=None):
    if ortalama is None:
        return skorlar
    if ortalama == "makro":
        return float(np.mean(skorlar))
    if ortalama == "agirlikli":
        toplam = destek.sum()
        if toplam == 0:
            return 0.0
        return float(np.sum(skorlar * destek) / toplam)
    if ortalama == "mikro":
        if mikro_skor is None:
            raise ValueError("Bu metrik için 'mikro' ortalama tanımlı değil.")
        return float(mikro_skor)
    if ortalama == "ikili":
        etiket_listesi = etiketler.tolist()
        if len(etiket_listesi) > 2:
            raise ValueError(
                "ortalama='ikili' yalnızca iki sınıflı problemlerde kullanılır; "
                f"{len(etiket_listesi)} sınıf bulundu: {etiket_listesi}. "
                "'makro', 'mikro' veya 'agirlikli' seçin."
            )
        if pozitif_etiket not in etiket_listesi:
            raise ValueError(
                f"pozitif_etiket={pozitif_etiket!r} veride yok. "
                f"Mevcut etiketler: {etiket_listesi}."
            )
        return float(skorlar[etiket_listesi.index(pozitif_etiket)])
    raise ValueError(
        f"ortalama {_ORTALAMALAR} degerlerinden biri olmalı; '{ortalama}' geldi."
    )


def _kesinlik_duyarlilik_fbeta(y_gercek, y_tahmin, *, beta=1.0, etiketler=None,
                               ortalama="ikili", pozitif_etiket=1,
                               sifir_bolme=0.0):
    """Precision / recall / F-beta üçlüsünü tek geçişte hesaplar."""
    etiketler = etiketleri_bul(y_gercek, y_tahmin, etiketler)
    matris = karmasiklik_matrisi(y_gercek, y_tahmin, etiketler=etiketler)
    tp, fp, fn, destek = _sinif_sayimlari(matris)

    kesinlikler = guvenli_bolme(tp, tp + fp, sifir_bolme=sifir_bolme)
    duyarliliklar = guvenli_bolme(tp, tp + fn, sifir_bolme=sifir_bolme)

    beta2 = float(beta) ** 2
    pay = (1 + beta2) * kesinlikler * duyarliliklar
    payda = beta2 * kesinlikler + duyarliliklar
    fbetalar = guvenli_bolme(pay, payda, sifir_bolme=sifir_bolme)

    # Mikro ortalama: TP/FP/FN havuzlanır, sonra tek skor hesaplanır.
    mikro_k = float(guvenli_bolme(tp.sum(), tp.sum() + fp.sum(),
                                  sifir_bolme=sifir_bolme))
    mikro_d = float(guvenli_bolme(tp.sum(), tp.sum() + fn.sum(),
                                  sifir_bolme=sifir_bolme))
    mikro_f = float(
        guvenli_bolme((1 + beta2) * mikro_k * mikro_d,
                      beta2 * mikro_k + mikro_d, sifir_bolme=sifir_bolme)
    )

    def uygula(skor, mikro):
        return _ortalamayi_uygula(skor, destek, ortalama, etiketler,
                                  pozitif_etiket, mikro_skor=mikro)

    return (uygula(kesinlikler, mikro_k), uygula(duyarliliklar, mikro_d),
            uygula(fbetalar, mikro_f), destek)


def dogruluk(y_gercek, y_tahmin, *, normalize: bool = True) -> float:
    """Accuracy = (TP + TN) / toplam.

    DİKKAT: dengesiz veride yanıltıcıdır. %99'u negatif olan bir veri setinde
    "her şeye negatif de" diyen model %99 accuracy alır ama hiçbir pozitifi
    yakalayamaz. Bu durumda `dengeli_dogruluk`, `f1_skoru` veya
    `matthews_korelasyonu` kullan.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin)
    dogru = int(np.sum(gercek == tahmin))
    return float(dogru / gercek.size) if normalize else float(dogru)


def kesinlik(y_gercek, y_tahmin, *, etiketler=None, ortalama="ikili",
             pozitif_etiket=1, sifir_bolme=0.0):
    """Precision = TP / (TP + FP), "pozitif dediklerimin kaçı gerçekten pozitif?"

    FP'nin pahalı olduğu yerde yükseltilir: spam filtresinde gerçek postayı
    spam'e atmak, bir dolandırıcılık modelinde masum işlemi bloklamak gibi.
    """
    k, _, _, _ = _kesinlik_duyarlilik_fbeta(
        y_gercek, y_tahmin, beta=1.0, etiketler=etiketler, ortalama=ortalama,
        pozitif_etiket=pozitif_etiket, sifir_bolme=sifir_bolme)
    return k


def duyarlilik(y_gercek, y_tahmin, *, etiketler=None, ortalama="ikili",
               pozitif_etiket=1, sifir_bolme=0.0):
    """Recall (duyarlılık, sensitivity, TPR) = TP / (TP + FN).

    "Gerçek pozitiflerin kaçını yakaladım?" FN'nin pahalı olduğu yerde
    yükseltilir: hastalık teşhisi, kanser taraması, arıza öngörüsü.
    """
    _, d, _, _ = _kesinlik_duyarlilik_fbeta(
        y_gercek, y_tahmin, beta=1.0, etiketler=etiketler, ortalama=ortalama,
        pozitif_etiket=pozitif_etiket, sifir_bolme=sifir_bolme)
    return d


def ozgulluk(y_gercek, y_tahmin, *, pozitif_etiket=1) -> float:
    """Specificity (seçicilik, TNR) = TN / (TN + FP). ROC'un x ekseni 1 - bu değerdir."""
    tn, fp, _, _ = ikili_bilesenler(y_gercek, y_tahmin,
                                    pozitif_etiket=pozitif_etiket)
    return float(guvenli_bolme(tn, tn + fp))


def f_beta_skoru(y_gercek, y_tahmin, *, beta=1.0, etiketler=None,
                 ortalama="ikili", pozitif_etiket=1, sifir_bolme=0.0):
    """F-beta = (1 + b²)·P·R / (b²·P + R).

    beta = 1   precision ve recall eşit ağırlıkta (F1)
    beta < 1   precision daha önemli (ör. F0.5: spam filtresi)
    beta > 1   recall daha önemli   (ör. F2 : hastalık taraması)

    Not: makro F-beta, sınıf başına F-beta'ların ortalamasıdır; makro P ile
    makro R'den yeniden hesaplanan değer DEĞİLDİR (ikisi farklı sayılardır).
    """
    _, _, f, _ = _kesinlik_duyarlilik_fbeta(
        y_gercek, y_tahmin, beta=beta, etiketler=etiketler, ortalama=ortalama,
        pozitif_etiket=pozitif_etiket, sifir_bolme=sifir_bolme)
    return f


def f1_skoru(y_gercek, y_tahmin, *, etiketler=None, ortalama="ikili",
             pozitif_etiket=1, sifir_bolme=0.0):
    """F1 = precision ve recall'ın harmonik ortalaması."""
    return f_beta_skoru(y_gercek, y_tahmin, beta=1.0, etiketler=etiketler,
                        ortalama=ortalama, pozitif_etiket=pozitif_etiket,
                        sifir_bolme=sifir_bolme)


def dengeli_dogruluk(y_gercek, y_tahmin) -> float:
    """Sınıf başına recall'ların ortalaması. Dengesiz veride accuracy'nin yerine geçer."""
    return float(duyarlilik(y_gercek, y_tahmin, ortalama="makro"))


def matthews_korelasyonu(y_gercek, y_tahmin) -> float:
    """MCC, karmaşıklık matrisinin dört hücresini birden kullanan tek sayı.

    [-1, +1] aralığında: +1 kusursuz, 0 rastgele, -1 tam ters tahmin.
    Dengesiz veride F1'den daha dürüsttür çünkü TN'yi de hesaba katar.
    """
    etiketler = etiketleri_bul(y_gercek, y_tahmin)
    matris = karmasiklik_matrisi(y_gercek, y_tahmin,
                                 etiketler=etiketler).astype(float)
    t = matris.sum(axis=1)   # gerçek sayıları
    p = matris.sum(axis=0)   # tahmin sayıları
    c = np.trace(matris)     # doğru tahminler
    s = matris.sum()         # toplam örnek
    pay = c * s - np.dot(t, p)
    payda = np.sqrt((s**2 - np.dot(p, p)) * (s**2 - np.dot(t, t)))
    return float(pay / payda) if payda != 0 else 0.0


def cohen_kappa(y_gercek, y_tahmin) -> float:
    """Cohen's kappa = (p_gozlenen - p_sans) / (1 - p_sans).

    "Model, sınıf dağılımını bilen rastgele tahminden ne kadar iyi?" sorusunu
    yanıtlar. 0 = şanstan farksız.
    """
    etiketler = etiketleri_bul(y_gercek, y_tahmin)
    matris = karmasiklik_matrisi(y_gercek, y_tahmin,
                                 etiketler=etiketler).astype(float)
    n = matris.sum()
    p_gozlenen = np.trace(matris) / n
    p_sans = np.dot(matris.sum(axis=0), matris.sum(axis=1)) / (n * n)
    if p_sans == 1:
        return 0.0
    return float((p_gozlenen - p_sans) / (1 - p_sans))


def log_kaybi(y_gercek, olasiliklar, *, etiketler=None, epsilon=1e-15) -> float:
    """Log-loss (çapraz entropi), lojistik regresyonun optimize ettiği kayıp.

    L = -(1/n) Σ Σ y_ik · log(p_ik)

    Sert etiketi değil, olasılığı cezalandırır: %99 güvenle yanlış tahmin,
    %51 güvenle yanlış tahminden çok daha pahalıdır. Kalibrasyonu ölçmek için
    accuracy'den kıyas kabul etmez şekilde daha bilgilendiricidir.
    """
    gercek = diziye_cevir(y_gercek, ad="y_gercek")
    olasiliklar = np.asarray(olasiliklar, dtype=float)
    etiketler = etiketleri_bul(gercek, None, etiketler)

    if olasiliklar.ndim == 1:
        if etiketler.size > 2:
            raise ValueError(
                "Tek boyutlu olasılık yalnızca ikili problemde kullanılabilir; "
                f"{etiketler.size} sınıf var. (n_ornek, n_sinif) matrisi verin."
            )
        if etiketler.size < 2:
            # y'de tek sınıf görünüyor (ör. tek örneklik kayıp hesabı).
            # Tek boyutlu olasılık "pozitif sınıf olasılığı" demektir, yani
            # etiket uzayı {0, 1}'dir; onu geri koyuyoruz.
            gecersiz = set(etiketler.tolist()) - {0, 1}
            if gecersiz:
                raise ValueError(
                    f"y_gercek yalnızca {sorted(etiketler.tolist())} içeriyor ve "
                    "bu değerler {0, 1} değil. Tek boyutlu olasılıkla "
                    "kullanabilmek için `etiketler` parametresini açıkça verin."
                )
            etiketler = np.array([0, 1])
        olasiliklar = np.column_stack([1.0 - olasiliklar, olasiliklar])
    if olasiliklar.shape[0] != gercek.size:
        raise ValueError(
            f"olasılık satır sayısı ({olasiliklar.shape[0]}) örnek sayısına "
            f"({gercek.size}) eşit olmalı."
        )
    if olasiliklar.shape[1] != etiketler.size:
        raise ValueError(
            f"olasılık sütun sayısı ({olasiliklar.shape[1]}) sınıf sayısına "
            f"({etiketler.size}) eşit olmalı."
        )

    olasiliklar = np.clip(olasiliklar, epsilon, 1 - epsilon)
    olasiliklar = olasiliklar / olasiliklar.sum(axis=1, keepdims=True)
    indeks = {etiket: i for i, etiket in enumerate(etiketler.tolist())}
    satir = np.arange(gercek.size)
    sutun = np.array([indeks[e] for e in gercek.tolist()])
    return float(-np.mean(np.log(olasiliklar[satir, sutun])))


# --------------------------------------------------------------------------
# Eşikten bağımsız metrikler: ROC ve PR eğrileri
# --------------------------------------------------------------------------
def _siralanmis_skorlar(y_gercek, skorlar, pozitif_etiket):
    """Skora göre azalan sıralayıp her farklı eşikte kümülatif (TP, FP) üretir."""
    gercek, skorlar = ciftleri_dogrula(y_gercek, skorlar)
    skorlar = skorlar.astype(float)
    pozitif = (gercek == pozitif_etiket).astype(int)
    if pozitif.sum() == 0 or pozitif.sum() == pozitif.size:
        raise ValueError(
            "ROC/PR eğrisi için veride hem pozitif hem negatif örnek olmalı. "
            f"Pozitif sayısı: {int(pozitif.sum())}, toplam: {pozitif.size}."
        )
    sira = np.argsort(-skorlar, kind="mergesort")  # kararlı sıralama
    skorlar = skorlar[sira]
    pozitif = pozitif[sira]
    # Eşit skorlar tek eşik noktası sayılmalı, yoksa eğri kendi içinde zıplar.
    farkli = np.where(np.diff(skorlar))[0]
    esik_indeksleri = np.r_[farkli, pozitif.size - 1]
    tp = np.cumsum(pozitif)[esik_indeksleri]
    fp = 1 + esik_indeksleri - tp
    return tp.astype(float), fp.astype(float), skorlar[esik_indeksleri]


def roc_egrisi(y_gercek, skorlar, *, pozitif_etiket=1):
    """ROC eğrisi: (fpr, tpr, esikler).

    Skor, modelin pozitif sınıfa verdiği olasılık/karar değeridir. Eğri her
    olası eşik için FPR = FP/(FP+TN) ve TPR = TP/(TP+FN) çiftini verir.
    """
    tp, fp, esikler = _siralanmis_skorlar(y_gercek, skorlar, pozitif_etiket)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    # Eğri (0,0)'dan başlamalı: hiçbir örneği pozitif saymayan eşik.
    return (np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[esikler[0] + 1.0, esikler])


def roc_auc(y_gercek, skorlar, *, pozitif_etiket=1) -> float:
    """ROC eğrisinin altındaki alan.

    Yorum: rastgele seçilen bir pozitif örneğe, rastgele seçilen bir negatiften
    daha yüksek skor verme olasılığı. 0.5 = yazı tura. Sınıf dağılımından
    etkilenmez, dengesiz veride bu hem avantaj (kararlı) hem tuzaktır
    (çok dengesiz veride PR-AUC daha bilgilendiricidir).
    """
    fpr, tpr, _ = roc_egrisi(y_gercek, skorlar, pozitif_etiket=pozitif_etiket)
    return float(np.trapezoid(tpr, fpr))


def kesinlik_duyarlilik_egrisi(y_gercek, skorlar, *, pozitif_etiket=1):
    """PR eğrisi: (kesinlikler, duyarliliklar, esikler).

    scikit-learn ile aynı sözleşme:
      - `esikler` ARTAN sırada, her farklı skor değeri için bir eşik,
      - `duyarliliklar` AZALAN (yüksek eşik = az pozitif tahmin = düşük recall),
      - eğri sonuna eşiği olmayan (kesinlik=1, duyarlilik=0) noktası eklenir,
        yani len(kesinlikler) == len(duyarliliklar) == len(esikler) + 1.
    """
    tp, fp, esikler = _siralanmis_skorlar(y_gercek, skorlar, pozitif_etiket)
    kesinlikler = guvenli_bolme(tp, tp + fp, sifir_bolme=1.0)
    duyarliliklar = tp / tp[-1]
    # `_siralanmis_skorlar` skoru azalan sıralar; sözleşmeye uymak için ters çevir.
    return (np.r_[kesinlikler[::-1], 1.0],
            np.r_[duyarliliklar[::-1], 0.0],
            esikler[::-1])


def ortalama_kesinlik(y_gercek, skorlar, *, pozitif_etiket=1) -> float:
    """Average Precision. PR eğrisinin basamak (step) toplamıyla alanı.

    AP = Σ (R_n - R_{n-1}) · P_n

    Trapez yerine basamak kullanılır; trapez PR eğrisinde iyimser sapma yapar.
    Çok dengesiz verilerde (ör. %1 pozitif) ROC-AUC'den daha ayırt edicidir:
    taban çizgisi pozitif sınıf oranına eşittir.
    """
    kesinlikler, duyarliliklar, _ = kesinlik_duyarlilik_egrisi(
        y_gercek, skorlar, pozitif_etiket=pozitif_etiket)
    return float(-np.sum(np.diff(duyarliliklar) * np.asarray(kesinlikler)[:-1]))


# --------------------------------------------------------------------------
# Rapor ve eşik seçimi
# --------------------------------------------------------------------------
def siniflandirma_raporu(y_gercek, y_tahmin, *, etiketler=None,
                         basamak: int = 4, metin: bool = False):
    """Sınıf başına kesinlik/duyarlılık/F1/destek + genel ortalamalar.

    metin=True verilirse konsola basılabilir tablo döner.
    """
    etiketler = etiketleri_bul(y_gercek, y_tahmin, etiketler)
    k, d, f, destek = _kesinlik_duyarlilik_fbeta(
        y_gercek, y_tahmin, etiketler=etiketler, ortalama=None)

    rapor: dict = {}
    for i, etiket in enumerate(etiketler.tolist()):
        rapor[etiket] = {
            "kesinlik": round(float(k[i]), basamak),
            "duyarlilik": round(float(d[i]), basamak),
            "f1": round(float(f[i]), basamak),
            "destek": int(destek[i]),
        }
    rapor["dogruluk"] = round(dogruluk(y_gercek, y_tahmin), basamak)
    for ad, strateji in (("makro ortalama", "makro"),
                         ("agirlikli ortalama", "agirlikli")):
        rapor[ad] = {
            "kesinlik": round(float(kesinlik(y_gercek, y_tahmin,
                                             etiketler=etiketler,
                                             ortalama=strateji)), basamak),
            "duyarlilik": round(float(duyarlilik(y_gercek, y_tahmin,
                                                 etiketler=etiketler,
                                                 ortalama=strateji)), basamak),
            "f1": round(float(f1_skoru(y_gercek, y_tahmin, etiketler=etiketler,
                                       ortalama=strateji)), basamak),
            "destek": int(destek.sum()),
        }
    if not metin:
        return rapor

    satirlar = [f"{'sinif':>18} {'kesinlik':>10} {'duyarlilik':>11} "
                f"{'f1':>8} {'destek':>8}"]
    for etiket in etiketler.tolist():
        s = rapor[etiket]
        satirlar.append(f"{str(etiket):>18} {s['kesinlik']:>10.4f} "
                        f"{s['duyarlilik']:>11.4f} {s['f1']:>8.4f} "
                        f"{s['destek']:>8d}")
    satirlar.append("")
    satirlar.append(f"{'dogruluk':>18} {'':>10} {'':>11} "
                    f"{rapor['dogruluk']:>8.4f} {int(destek.sum()):>8d}")
    for ad in ("makro ortalama", "agirlikli ortalama"):
        s = rapor[ad]
        satirlar.append(f"{ad:>18} {s['kesinlik']:>10.4f} "
                        f"{s['duyarlilik']:>11.4f} {s['f1']:>8.4f} "
                        f"{s['destek']:>8d}")
    return "\n".join(satirlar)


def esik_tara(y_gercek, skorlar, *, metrik="f1", pozitif_etiket=1,
              en_az_duyarlilik=None, en_az_kesinlik=None):
    """Karar eşiğini veriye göre seçer, 0.5 kutsal bir sayı değildir.

    Sigmoid çıktısını 0.5'te kesmek yalnızca sınıflar dengeliyse ve FP ile FN
    aynı maliyetteyse mantıklıdır. Bu fonksiyon tüm eşikleri tarar ve seçilen
    metriği en yükseğe çıkaran eşiği döndürür.

    metrik: "f1", "f2", "f0.5", "youden" (TPR - FPR), "kesinlik", "duyarlilik"
    en_az_duyarlilik / en_az_kesinlik: kısıt koyar, ör. "recall en az 0.90
    olsun, o kısıt altında precision'ı maksimize et".

    Dönüş: (en_iyi_esik, en_iyi_skor, tum_esikler, tum_skorlar)

    UYARI: eşik, test setinde değil doğrulama (validation) setinde seçilmelidir.
    Test setinde seçilen eşik, test skorunu iyimser yönde bozar.
    """
    tp, fp, esikler = _siralanmis_skorlar(y_gercek, skorlar, pozitif_etiket)
    toplam_pozitif, toplam_negatif = tp[-1], fp[-1]
    duyarliliklar = tp / toplam_pozitif
    kesinlikler = guvenli_bolme(tp, tp + fp, sifir_bolme=0.0)
    fpr = fp / toplam_negatif

    if metrik == "youden":
        skor_dizisi = duyarliliklar - fpr
    elif metrik == "kesinlik":
        skor_dizisi = kesinlikler
    elif metrik == "duyarlilik":
        skor_dizisi = duyarliliklar
    elif metrik.startswith("f"):
        try:
            beta = float(metrik[1:]) if len(metrik) > 1 else 1.0
        except ValueError as hata:
            raise ValueError(f"Metrik adı çözülemedi: '{metrik}'.") from hata
        b2 = beta**2
        skor_dizisi = guvenli_bolme((1 + b2) * kesinlikler * duyarliliklar,
                                    b2 * kesinlikler + duyarliliklar)
    else:
        raise ValueError(
            "metrik 'f1', 'f2', 'f0.5', 'youden', 'kesinlik' veya 'duyarlilik' "
            f"olmalı; '{metrik}' geldi."
        )

    uygun = np.ones_like(skor_dizisi, dtype=bool)
    if en_az_duyarlilik is not None:
        uygun &= duyarliliklar >= en_az_duyarlilik
    if en_az_kesinlik is not None:
        uygun &= kesinlikler >= en_az_kesinlik
    if not uygun.any():
        raise ValueError(
            "Verilen kısıtları sağlayan eşik yok "
            f"(en_az_duyarlilik={en_az_duyarlilik}, "
            f"en_az_kesinlik={en_az_kesinlik})."
        )

    maskeli = np.where(uygun, skor_dizisi, -np.inf)
    en_iyi = int(np.argmax(maskeli))
    return float(esikler[en_iyi]), float(skor_dizisi[en_iyi]), esikler, skor_dizisi
