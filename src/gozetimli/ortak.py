"""Modüller arasında paylaşılan küçük yardımcılar.

Buradaki fonksiyonlar bilerek "aptal" tutulmuştur: girdiyi NumPy dizisine
çevirir, boyut uyuşmazlığını erken yakalar ve sıfıra bölmeyi tek bir yerden
yönetir. Metrik modülleri bu davranışı paylaştığı için hata mesajları da
her yerde aynı dili konuşur.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "diziye_cevir",
    "ciftleri_dogrula",
    "guvenli_bolme",
    "etiketleri_bul",
]


def diziye_cevir(deger, *, dtype=None, ad: str = "dizi") -> np.ndarray:
    """Girdiyi 1 boyutlu NumPy dizisine çevirir.

    Listeler, pandas Series/Index, tek sütunlu (n, 1) diziler kabul edilir.
    2 boyutlu ve gerçekten çok sütunlu bir şey gelirse hata verir; sessizce
    düzleştirmek (ravel) sınıf sayısı yanlış hesaplandığında fark edilmesi
    çok zor hatalar üretiyor.
    """
    dizi = np.asarray(deger, dtype=dtype)
    if dizi.ndim == 2 and dizi.shape[1] == 1:
        dizi = dizi.ravel()
    if dizi.ndim != 1:
        raise ValueError(
            f"{ad} 1 boyutlu olmalı, {dizi.ndim} boyutlu geldi (şekil={dizi.shape})."
        )
    if dizi.size == 0:
        raise ValueError(f"{ad} boş olamaz.")
    return dizi


def ciftleri_dogrula(y_gercek, y_tahmin, *, dtype=None):
    """(y_gercek, y_tahmin) çiftini diziye çevirip uzunluklarını karşılaştırır."""
    gercek = diziye_cevir(y_gercek, dtype=dtype, ad="y_gercek")
    tahmin = diziye_cevir(y_tahmin, dtype=dtype, ad="y_tahmin")
    if gercek.shape[0] != tahmin.shape[0]:
        raise ValueError(
            "y_gercek ve y_tahmin aynı uzunlukta olmalı: "
            f"{gercek.shape[0]} != {tahmin.shape[0]}."
        )
    return gercek, tahmin


def guvenli_bolme(pay, payda, *, sifir_bolme: float = 0.0):
    """payda == 0 olan yerlerde `sifir_bolme` döndüren eleman bazlı bölme.

    Precision/recall hesabında payda sıfır olabilir (model hiç pozitif tahmin
    etmediyse TP+FP = 0). scikit-learn bu durumda uyarı basıp 0 döndürür;
    burada davranış açıkça parametreyle seçilir.
    """
    pay = np.asarray(pay, dtype=float)
    payda = np.asarray(payda, dtype=float)
    sonuc = np.full(np.broadcast(pay, payda).shape, float(sifir_bolme))
    gecerli = payda != 0
    np.divide(pay, payda, out=sonuc, where=gecerli)
    return sonuc


def etiketleri_bul(y_gercek, y_tahmin=None, etiketler=None) -> np.ndarray:
    """Kullanılacak sınıf etiketlerini belirler ve sıralı döndürür.

    `etiketler` verilmişse ona saygı duyulur (sıra korunur), karmaşıklık
    matrisinin satır/sütun sırasını kullanıcının seçebilmesi için.
    """
    if etiketler is not None:
        etiketler = np.asarray(etiketler)
        if etiketler.ndim != 1 or etiketler.size == 0:
            raise ValueError("etiketler boş olmayan 1 boyutlu bir dizi olmalı.")
        if np.unique(etiketler).size != etiketler.size:
            raise ValueError("etiketler listesi tekrar eden değer içeremez.")
        return etiketler
    birlesik = np.concatenate(
        [np.asarray(y_gercek).ravel()]
        + ([np.asarray(y_tahmin).ravel()] if y_tahmin is not None else [])
    )
    return np.unique(birlesik)
