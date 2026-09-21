"""Boosting hesapları: AdaBoost ağırlıkları, gradient boosting ve XGBoost formülleri.

Ders notlarındaki (16-AdaBoost, 17-Gradient_Boosting, 18-XGBoost) el
hesaplarını çalıştırılabilir hale getirir. Amaç kütüphane yazmak değil;
"bu sayı nereden geldi?" sorusunu koda çevirip test edilebilir kılmaktır.
`testler/test_ders_ornekleri.py` buradaki fonksiyonlarla PDF'lerdeki her
ara değeri yeniden üretir.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigmoid",
    "log_odds",
    "log_loss_gradyan",
    "log_loss_hessian",
    "adaboost_hata_orani",
    "adaboost_agirlik_katsayisi",
    "adaboost_agirliklari_guncelle",
    "adaboost_bin_araliklari",
    "adaboost_nihai_skor",
    "gb_baslangic_degeri",
    "xgb_benzerlik",
    "xgb_kazanc",
    "xgb_yaprak_degeri",
    "xgb_cover",
]


# --------------------------------------------------------------------------
# Ortak dönüşümler
# --------------------------------------------------------------------------
def sigmoid(z):
    """σ(z) = 1 / (1 + e^-z) — log-odds'u [0, 1] olasılığına sıkıştırır.

    Taşma güvenli biçimde hesaplanır: z çok negatifken naif formül exp(+800)
    üretip taşar, burada pozitif ve negatif dallar ayrı ele alınır.
    """
    z = np.asarray(z, dtype=float)
    cikti = np.empty_like(z)
    pozitif = z >= 0
    cikti[pozitif] = 1.0 / (1.0 + np.exp(-z[pozitif]))
    ust = np.exp(z[~pozitif])
    cikti[~pozitif] = ust / (1.0 + ust)
    return cikti if cikti.ndim else float(cikti)


def log_odds(p, *, epsilon: float = 1e-15):
    """logit(p) = log(p / (1-p)) — sigmoid'in tersi.

    Gradient boosting ve XGBoost sınıflandırmada tahminleri bu ölçekte tutar:
    log-odds uzayında toplama yapmak, olasılık uzayında çarpmaya denk gelir ve
    [0,1] sınırından taşma sorununu ortadan kaldırır.
    """
    p = np.clip(np.asarray(p, dtype=float), epsilon, 1 - epsilon)
    sonuc = np.log(p / (1 - p))
    return sonuc if sonuc.ndim else float(sonuc)


def log_loss_gradyan(y, p):
    """Log-loss'un 1. türevi: g = p - y.

    XGBoost'un kullandığı gradyandır. Ders notlarındaki "residual" R = y - p
    bunun negatifidir (negatif gradyan). Benzerlik/yaprak formüllerinde kare
    ya da işaretli toplam olarak girdiği için hangi işaretin kullanıldığına
    dikkat etmek gerekir.
    """
    return np.asarray(p, dtype=float) - np.asarray(y, dtype=float)


def log_loss_hessian(p):
    """Log-loss'un 2. türevi: h = p·(1-p).

    "Hatanın değişim hızı". p 0 veya 1'e yaklaştıkça sıfıra gider — model emin
    olduğu yerlerde küçük adım atar. XGBoost'ta bu aynı zamanda `cover`dır.
    """
    p = np.asarray(p, dtype=float)
    return p * (1.0 - p)


# --------------------------------------------------------------------------
# AdaBoost
# --------------------------------------------------------------------------
def adaboost_hata_orani(agirliklar, dogru_mu) -> float:
    """ε = (yanlış sınıflanan örneklerin ağırlık toplamı) / (toplam ağırlık).

    dogru_mu: her örnek için True/False dizisi.
    """
    agirliklar = np.asarray(agirliklar, dtype=float)
    dogru_mu = np.asarray(dogru_mu, dtype=bool)
    if agirliklar.shape != dogru_mu.shape:
        raise ValueError("agirliklar ve dogru_mu aynı uzunlukta olmalı.")
    toplam = agirliklar.sum()
    if toplam <= 0:
        raise ValueError("Ağırlıkların toplamı pozitif olmalı.")
    return float(agirliklar[~dogru_mu].sum() / toplam)


def adaboost_agirlik_katsayisi(hata_orani: float, *, epsilon: float = 1e-12) -> float:
    """α = ½ · ln((1 - ε) / ε) — zayıf öğrenicinin nihai oylamadaki ağırlığı.

    Davranış:
        ε = 0.5  -> α = 0     : yazı tura kadar iyi, hiç söz hakkı yok.
        ε < 0.5  -> α > 0     : hata küçüldükçe söz hakkı hızla büyür.
        ε > 0.5  -> α < 0     : rastgeleden kötü; AdaBoost tahmini ters çevirip
                                yine de kullanır (bu matematiksel olarak tutarlıdır).
        ε = 0    -> +∞        : epsilon ile kırpılır, yoksa sayısal taşma olur.

    Ders notundaki örnek: 7 örnekte 1 hata -> ε = 1/7 ≈ 0.143, α ≈ 0.895.
    """
    hata_orani = float(np.clip(hata_orani, epsilon, 1 - epsilon))
    return float(0.5 * np.log((1 - hata_orani) / hata_orani))


def adaboost_agirliklari_guncelle(agirliklar, dogru_mu, alfa: float, *,
                                  normalize: bool = True):
    """w_i ← w_i · e^(∓α); doğru tahminlerde eksi, yanlışlarda artı üs.

    Yanlış sınıflanan örneklerin ağırlığı büyür, böylece bir sonraki zayıf
    öğrenici bu örneklere odaklanmak zorunda kalır. normalize=True ise toplam
    1'e çekilir — ders notundaki "bin aralığı" mantığı bunu gerektirir.
    """
    agirliklar = np.asarray(agirliklar, dtype=float)
    dogru_mu = np.asarray(dogru_mu, dtype=bool)
    if agirliklar.shape != dogru_mu.shape:
        raise ValueError("agirliklar ve dogru_mu aynı uzunlukta olmalı.")
    isaret = np.where(dogru_mu, -1.0, 1.0)
    yeni = agirliklar * np.exp(isaret * float(alfa))
    if normalize:
        toplam = yeni.sum()
        if toplam <= 0:
            raise ValueError("Güncellenen ağırlıkların toplamı pozitif olmalı.")
        yeni = yeni / toplam
    return yeni


def adaboost_bin_araliklari(agirliklar):
    """Normalize ağırlıklardan kümülatif [alt, ust) aralıkları üretir.

    Ders notundaki "bin ataması" adımı: sonraki turda örnekler 0-1 arasında
    rastgele sayı çekilerek seçilir; ağırlığı büyük örnek geniş aralık kaplar,
    dolayısıyla birden çok kez seçilme olasılığı yüksektir.

    Dönüş: (n, 2) şeklinde dizi.
    """
    agirliklar = np.asarray(agirliklar, dtype=float)
    toplam = agirliklar.sum()
    if not np.isclose(toplam, 1.0):
        agirliklar = agirliklar / toplam
    ust = np.cumsum(agirliklar)
    alt = np.concatenate([[0.0], ust[:-1]])
    return np.column_stack([alt, ust])


def adaboost_nihai_skor(alfalar, tahminler):
    """F(x) = Σ α_m · h_m(x); sınıf kararı sign(F(x)).

    tahminler ±1 kodlamasında olmalıdır (Yes = +1, No = -1).
    """
    alfalar = np.asarray(alfalar, dtype=float)
    tahminler = np.asarray(tahminler, dtype=float)
    if tahminler.ndim == 1:
        if alfalar.shape != tahminler.shape:
            raise ValueError("alfalar ve tahminler aynı uzunlukta olmalı.")
        return float(np.dot(alfalar, tahminler))
    # (n_model, n_ornek) matrisi -> örnek başına skor
    if tahminler.shape[0] != alfalar.size:
        raise ValueError(
            "tahminler matrisinin satır sayısı model sayısına eşit olmalı."
        )
    return alfalar @ tahminler


# --------------------------------------------------------------------------
# Gradient boosting / XGBoost
# --------------------------------------------------------------------------
def gb_baslangic_degeri(y, *, gorev: str = "regresyon") -> float:
    """F₀ — hiçbir özelliğe bakmadan yapılabilecek en iyi sabit tahmin.

    gorev="regresyon"      -> hedefin ortalaması (MSE'yi minimize eder)
    gorev="siniflandirma"  -> pozitif oranın log-odds'u (log-loss'u minimize eder)

    Ders notundaki örnekler: maaş verisinde F₀ = 70; 5 pozitif / 2 negatifli
    kredi verisinde F₀ = ln(5/2) ≈ 0.916.
    """
    y = np.asarray(y, dtype=float).ravel()
    if gorev == "regresyon":
        return float(np.mean(y))
    if gorev == "siniflandirma":
        oran = float(np.mean(y))
        if oran in (0.0, 1.0):
            raise ValueError(
                "Tek sınıflı veride log-odds tanımsızdır (±sonsuz); "
                "hem pozitif hem negatif örnek gerekir."
            )
        return float(np.log(oran / (1 - oran)))
    raise ValueError(
        f"gorev 'regresyon' veya 'siniflandirma' olmalı; '{gorev}' geldi."
    )


def xgb_benzerlik(gradyan_toplami=None, hessian_toplami=None, *,
                  artiklar=None, olasiliklar=None, lam: float = 1.0) -> float:
    """Similarity score = (Σg)² / (Σh + λ).

    İki kullanım:
      1) Doğrudan toplamlarla:
         xgb_benzerlik(-1.142, 0.612, lam=1)
      2) Ders notundaki gibi artık ve olasılık dizileriyle:
         xgb_benzerlik(artiklar=[...], olasiliklar=[...], lam=1)
         Bu durumda Σh = Σ p(1-p) olarak hesaplanır.

    Sınıflandırmada h = p(1-p), regresyonda h = 1 olduğundan payda örnek
    sayısı + λ olur — ders notundaki regresyon formülü (ΣR)²/N ile aynı şeydir
    (λ = 0 alındığında).

    λ (lambda) yükseldikçe skor küçülür: yaprak değerleri bastırılır, model
    temkinli olur. λ = 0 klasik gradient boosting davranışına döner.
    """
    if artiklar is not None:
        artiklar = np.asarray(artiklar, dtype=float).ravel()
        gradyan_toplami = float(artiklar.sum())
        if olasiliklar is None:
            # Regresyon: her örneğin hessian'ı 1'dir.
            hessian_toplami = float(artiklar.size)
        else:
            olasiliklar = np.asarray(olasiliklar, dtype=float).ravel()
            if olasiliklar.size == 1:
                olasiliklar = np.repeat(olasiliklar, artiklar.size)
            if olasiliklar.shape != artiklar.shape:
                raise ValueError("artiklar ve olasiliklar aynı uzunlukta olmalı.")
            hessian_toplami = float(np.sum(olasiliklar * (1 - olasiliklar)))
    if gradyan_toplami is None or hessian_toplami is None:
        raise ValueError(
            "Ya (gradyan_toplami, hessian_toplami) ya da artiklar verilmelidir."
        )
    payda = float(hessian_toplami) + float(lam)
    if payda == 0:
        raise ValueError("Σh + λ sıfır olamaz; λ > 0 verin.")
    return float(gradyan_toplami ** 2 / payda)


def xgb_kazanc(sol_benzerlik: float, sag_benzerlik: float, kok_benzerlik: float,
               *, gama: float = 0.0, yarim_faktor: bool = False) -> float:
    """Gain = S_sol + S_sag - S_kök  (- γ).

    yarim_faktor:
        False (varsayılan) — ders notundaki biçim. Bölmeleri sıralamak için
            yeterlidir çünkü ½ tüm adaylarda ortaktır.
        True — XGBoost makalesindeki (Chen & Guestrin, 2016, Denklem 7) tam
            biçim: ½·[...] - γ. Gerçek kütüphanenin bastığı `gain` değeriyle
            karşılaştırma yapacaksan bunu kullan, yoksa sayılar 2 kat farklı çıkar.

    γ (gamma), bir bölmenin yapılabilmesi için gereken en küçük kazançtır:
    Gain < γ ise bölme reddedilir. Bu, XGBoost'un ön-budama mekanizmasıdır.
    """
    ham = float(sol_benzerlik) + float(sag_benzerlik) - float(kok_benzerlik)
    if yarim_faktor:
        ham *= 0.5
    return float(ham - float(gama))


def xgb_yaprak_degeri(artiklar=None, olasiliklar=None, *, gradyan_toplami=None,
                      hessian_toplami=None, lam: float = 1.0) -> float:
    """Yaprak çıktısı V = Σg / (Σh + λ).

    Benzerlik skorunun payında kare varken burada yok: benzerlik bölmenin
    kalitesini ölçer (işaret önemsiz), yaprak değeri ise düzeltmenin yönünü ve
    büyüklüğünü verir (işaret önemli).

    Ders notu örneği: sol yaprakta ΣR = -1.142, Σp(1-p) = 0.612, λ = 1
    -> V = -1.142 / 1.612 ≈ -0.709.
    """
    if artiklar is not None:
        artiklar = np.asarray(artiklar, dtype=float).ravel()
        gradyan_toplami = float(artiklar.sum())
        if olasiliklar is None:
            hessian_toplami = float(artiklar.size)
        else:
            olasiliklar = np.asarray(olasiliklar, dtype=float).ravel()
            if olasiliklar.size == 1:
                olasiliklar = np.repeat(olasiliklar, artiklar.size)
            hessian_toplami = float(np.sum(olasiliklar * (1 - olasiliklar)))
    if gradyan_toplami is None or hessian_toplami is None:
        raise ValueError(
            "Ya (gradyan_toplami, hessian_toplami) ya da artiklar verilmelidir."
        )
    payda = float(hessian_toplami) + float(lam)
    if payda == 0:
        raise ValueError("Σh + λ sıfır olamaz; λ > 0 verin.")
    return float(float(gradyan_toplami) / payda)


def xgb_cover(olasiliklar) -> float:
    """Cover = Σ p(1-p) — bir düğümdeki "bilgi ağırlığı" (hessian toplamı).

    `min_child_weight` parametresi bunun alt sınırıdır. Cover küçükse düğümde
    ya çok az örnek vardır ya da model o örneklerden zaten emindir (p ≈ 0 veya
    p ≈ 1); iki durumda da bölmeye devam etmek gürültü öğrenmektir.
    """
    p = np.asarray(olasiliklar, dtype=float).ravel()
    return float(np.sum(p * (1 - p)))
