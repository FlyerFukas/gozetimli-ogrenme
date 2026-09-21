"""Regresyon metrikleri.

KAYNAK NOTU: Atıl Samancıoğlu'nun elimizdeki ders notlarında (5-19 arası PDF)
regresyon metrikleri ayrı bir başlık olarak İŞLENMEMİŞTİR. MSE yalnızca
doğrusal/Ridge/Lasso maliyet fonksiyonunun içinde geçer, varyans azaltımı ise
karar ağacı bölme ölçütü olarak anlatılır. Elimizdeki klasörde 1-4 numaralı
PDF'ler eksik; regresyon metriklerinin orada anlatılmış olması muhtemel.
Bu modüldeki tanımlar standart istatistik/ML literatüründen alınmıştır ve
`testler/test_sklearn_uyumu.py` içinde scikit-learn'e karşı doğrulanır.
"""

from __future__ import annotations

import numpy as np

from ..ortak import ciftleri_dogrula, guvenli_bolme

__all__ = [
    "mse",
    "rmse",
    "mae",
    "medyan_mutlak_hata",
    "mape",
    "smape",
    "msle",
    "rmsle",
    "r2",
    "duzeltilmis_r2",
    "aciklanan_varyans",
    "maksimum_hata",
    "regresyon_raporu",
]


def mse(y_gercek, y_tahmin, *, ornek_agirliklari=None) -> float:
    """Ortalama Kare Hata = (1/n) Σ (y - ŷ)².

    Büyük hataları karesiyle cezalandırır, yani aykırı değerlere duyarlıdır.
    Doğrusal regresyonun ve karar ağacı regresörünün varsayılan kaybıdır.
    Birimi hedefin birimi DEĞİLDİR (kare birimdir), raporlarken RMSE tercih et.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    kareler = (gercek - tahmin) ** 2
    if ornek_agirliklari is None:
        return float(np.mean(kareler))
    agirliklar = np.asarray(ornek_agirliklari, dtype=float)
    if agirliklar.shape != gercek.shape:
        raise ValueError("ornek_agirliklari, y_gercek ile aynı şekilde olmalı.")
    return float(np.average(kareler, weights=agirliklar))


def rmse(y_gercek, y_tahmin, *, ornek_agirliklari=None) -> float:
    """Kök Ortalama Kare Hata = √MSE. Hedefle aynı birimdedir, bu yüzden raporlanabilir."""
    return float(np.sqrt(mse(y_gercek, y_tahmin,
                             ornek_agirliklari=ornek_agirliklari)))


def mae(y_gercek, y_tahmin, *, ornek_agirliklari=None) -> float:
    """Ortalama Mutlak Hata = (1/n) Σ |y - ŷ|.

    Aykırı değerlere MSE'den dayanıklıdır (hatayı karesine almaz). MAE'yi
    minimize eden sabit tahmin medyandır; MSE'yi minimize eden ortalamadır:
    ikisi arasındaki seçim aslında "hangi merkezi eğilim doğru?" sorusudur.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    mutlak = np.abs(gercek - tahmin)
    if ornek_agirliklari is None:
        return float(np.mean(mutlak))
    return float(np.average(mutlak, weights=np.asarray(ornek_agirliklari,
                                                       dtype=float)))


def medyan_mutlak_hata(y_gercek, y_tahmin) -> float:
    """Mutlak hataların medyanı. Aykırı değerlere en dayanıklı temel metrik."""
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    return float(np.median(np.abs(gercek - tahmin)))


def mape(y_gercek, y_tahmin, *, epsilon: float = 1e-10) -> float:
    """Ortalama Mutlak Yüzde Hata = (1/n) Σ |y - ŷ| / |y|.

    Oran döner (0.12 = %12). Yorumlaması kolay olduğu için iş birimlerinin
    sevdiği metriktir ama üç tuzağı vardır:
      1. y = 0 olan örneklerde tanımsızdır (burada epsilon ile korunur).
      2. Asimetriktir: düşük tahmin etmeyi, yüksek tahmin etmekten daha az
         cezalandırır, modelin sistematik olarak düşük tahmin etmesini teşvik eder.
      3. Küçük gerçek değerlerde patlar (y=1, ŷ=2 → %100 hata).
    Talep tahmininde sıfıra yakın değerler varsa `smape` veya `mae` kullan.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    payda = np.maximum(np.abs(gercek), epsilon)
    return float(np.mean(np.abs(gercek - tahmin) / payda))


def smape(y_gercek, y_tahmin) -> float:
    """Simetrik MAPE = (1/n) Σ |y - ŷ| / ((|y| + |ŷ|)/2).

    MAPE'nin asimetri sorununu hafifletir ve [0, 2] aralığında sınırlıdır.
    y ve ŷ birlikte sıfırsa o örneğin katkısı 0 sayılır.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    payda = (np.abs(gercek) + np.abs(tahmin)) / 2.0
    return float(np.mean(guvenli_bolme(np.abs(gercek - tahmin), payda,
                                       sifir_bolme=0.0)))


def msle(y_gercek, y_tahmin) -> float:
    """Ortalama Kare Logaritmik Hata = (1/n) Σ (log(1+y) - log(1+ŷ))².

    Mutlak hatayı değil ORANSAL hatayı cezalandırır. Hedef üstel büyüyorsa
    (gelir, nüfus, ziyaret sayısı) tercih edilir. Negatif değer kabul etmez.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    if np.any(gercek < 0) or np.any(tahmin < 0):
        raise ValueError(
            "msle negatif değerlerle tanımsızdır; y_gercek ve y_tahmin >= 0 olmalı."
        )
    return float(np.mean((np.log1p(gercek) - np.log1p(tahmin)) ** 2))


def rmsle(y_gercek, y_tahmin) -> float:
    """√MSLE."""
    return float(np.sqrt(msle(y_gercek, y_tahmin)))


def r2(y_gercek, y_tahmin) -> float:
    """Belirleme katsayısı R² = 1 - SS_kalinti / SS_toplam.

    "Modelim, her şeye ortalamayı söyleyen modelden ne kadar iyi?" sorusunu
    yanıtlar. 1 = kusursuz, 0 = ortalamayla aynı, NEGATİF = ortalamadan kötü
    (test setinde bunu görmek mümkündür ve gerçek bir uyarıdır).

    Tuzak: R² özellik eklendikçe hiçbir zaman DÜŞMEZ; anlamsız bir sütun bile
    eğitim R²'sini yükseltir. Model karşılaştırırken `duzeltilmis_r2` kullan.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    ss_kalinti = float(np.sum((gercek - tahmin) ** 2))
    ss_toplam = float(np.sum((gercek - np.mean(gercek)) ** 2))
    if ss_toplam == 0:
        # Sabit hedef: model de sabiti tam tutturduysa 1, tutturamadıysa 0.
        return 1.0 if ss_kalinti == 0 else 0.0
    return 1.0 - ss_kalinti / ss_toplam


def duzeltilmis_r2(y_gercek, y_tahmin, *, ozellik_sayisi: int) -> float:
    """Düzeltilmiş R² = 1 - (1 - R²)·(n - 1)/(n - p - 1).

    Her yeni özelliğin bedelini ödetir: özellik gerçekten bilgi katmıyorsa
    düzeltilmiş R² DÜŞER. Aynı veri üzerinde farklı sayıda özellikli modelleri
    karşılaştırırken doğru olan budur.
    """
    gercek, _ = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    n = gercek.size
    p = int(ozellik_sayisi)
    if p < 0:
        raise ValueError("ozellik_sayisi negatif olamaz.")
    if n - p - 1 <= 0:
        raise ValueError(
            f"Düzeltilmiş R² için örnek sayısı (n={n}) özellik sayısından "
            f"(p={p}) en az 2 fazla olmalı."
        )
    return float(1.0 - (1.0 - r2(gercek, y_tahmin)) * (n - 1) / (n - p - 1))


def aciklanan_varyans(y_gercek, y_tahmin) -> float:
    """Açıklanan varyans = 1 - Var(y - ŷ) / Var(y).

    R²'den farkı: kalıntıların ORTALAMASINI cezalandırmaz. Model sistematik
    olarak sabit bir miktar sapıyorsa (bias) R² düşer ama açıklanan varyans
    yüksek kalır. İkisinin arasındaki fark, modelin yanlılığını ele verir.
    """
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    kalinti_var = float(np.var(gercek - tahmin))
    toplam_var = float(np.var(gercek))
    if toplam_var == 0:
        return 1.0 if kalinti_var == 0 else 0.0
    return 1.0 - kalinti_var / toplam_var


def maksimum_hata(y_gercek, y_tahmin) -> float:
    """En kötü tek örnek hatası: max |y - ŷ|. En kötü durum garantisi gerektiğinde."""
    gercek, tahmin = ciftleri_dogrula(y_gercek, y_tahmin, dtype=float)
    return float(np.max(np.abs(gercek - tahmin)))


def regresyon_raporu(y_gercek, y_tahmin, *, ozellik_sayisi=None,
                     basamak: int = 4) -> dict:
    """Tek çağrıda temel regresyon metrikleri sözlüğü."""
    rapor = {
        "mse": round(mse(y_gercek, y_tahmin), basamak),
        "rmse": round(rmse(y_gercek, y_tahmin), basamak),
        "mae": round(mae(y_gercek, y_tahmin), basamak),
        "medyan_mutlak_hata": round(medyan_mutlak_hata(y_gercek, y_tahmin), basamak),
        "maksimum_hata": round(maksimum_hata(y_gercek, y_tahmin), basamak),
        "r2": round(r2(y_gercek, y_tahmin), basamak),
        "aciklanan_varyans": round(aciklanan_varyans(y_gercek, y_tahmin), basamak),
    }
    gercek = np.asarray(y_gercek, dtype=float)
    if not np.any(np.isclose(gercek, 0.0)):
        rapor["mape"] = round(mape(y_gercek, y_tahmin), basamak)
        rapor["smape"] = round(smape(y_gercek, y_tahmin), basamak)
    if ozellik_sayisi is not None:
        rapor["duzeltilmis_r2"] = round(
            duzeltilmis_r2(y_gercek, y_tahmin, ozellik_sayisi=ozellik_sayisi),
            basamak)
    return rapor
