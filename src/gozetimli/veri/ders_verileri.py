"""Ders notlarındaki örnek veri setleri — el hesaplarını yeniden üretmek için.

Bu tablolar Atıl Samancıoğlu'nun ders notlarındaki (5-19 numaralı PDF'ler)
öğretici mini veri setleridir. Her biri 4-7 satırlıktır ve tek amacı bir
formülü elle takip edebilmektir; hiçbiri gerçek veri değildir, model eğitmek
için kullanılmamalıdır.

Notlardaki sayısal sonuçların doğrulaması `testler/test_ders_ornekleri.py`
içindedir. Notlarda tespit edilen iki tutarsızlık `KAYNAK_NOTLARI` sözlüğünde
açıkça yazılıdır — sessizce düzeltilmemiştir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "VeriSeti",
    "kredi_onay",
    "burs_regresyon",
    "knn_sporcu",
    "naive_bayes_spam",
    "naive_bayes_obezite",
    "gb_maas",
    "gb_kredi_karti",
    "xgb_siniflandirma",
    "xgb_regresyon",
    "adaboost_regresyon",
    "KAYNAK_NOTLARI",
]


@dataclass
class VeriSeti:
    """Küçük bir ders veri setinin taşıyıcısı."""

    ad: str
    X: np.ndarray
    y: np.ndarray
    ozellik_adlari: list[str]
    hedef_adi: str
    kaynak: str
    aciklama: str = ""
    ham_satirlar: list = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def tablo(self) -> str:
        """Notlardaki tabloyu konsolda yeniden basar."""
        basliklar = ["ID"] + self.ozellik_adlari + [self.hedef_adi]
        satirlar = [" | ".join(f"{b:>14}" for b in basliklar)]
        satirlar.append("-" * len(satirlar[0]))
        for i, (ozellikler, hedef) in enumerate(zip(self.X.tolist(),
                                                    self.y.tolist()), start=1):
            hucreler = [f"{i:>14}"]
            hucreler += [f"{d:>14}" for d in ozellikler]
            hucreler.append(f"{hedef:>14}")
            satirlar.append(" | ".join(hucreler))
        return "\n".join(satirlar)


def kredi_onay() -> VeriSeti:
    """Karar ağacı ve AdaBoost bölümlerinin ortak veri seti (7 satır).

    Özellikler sayısallaştırılmıştır:
        gpa_ustu_3    : GPA > 3.0 ise 1, değilse 0
        mulakat       : Düşük = 0, Normal = 1, Yüksek = 2
    Hedef: Onay = 1, Red = 0.
    """
    ham = [
        ("<3.0", "Düşük", "Red"),
        ("<3.0", "Yüksek", "Onay"),
        ("<3.0", "Yüksek", "Onay"),
        (">3.0", "Düşük", "Red"),
        (">3.0", "Yüksek", "Onay"),
        (">3.0", "Normal", "Onay"),
        ("<3.0", "Normal", "Red"),
    ]
    mulakat_kodu = {"Düşük": 0, "Normal": 1, "Yüksek": 2}
    X = np.array([[1 if gpa == ">3.0" else 0, mulakat_kodu[m]] for gpa, m, _ in ham],
                 dtype=float)
    y = np.array([1 if onay == "Onay" else 0 for _, _, onay in ham], dtype=int)
    return VeriSeti(
        ad="kredi_onay",
        X=X, y=y,
        ozellik_adlari=["gpa_ustu_3", "mulakat"],
        hedef_adi="onay",
        kaynak="13-Decision_Tree_Algorithms.pdf Tablo 1 / 16-Adaboost_Algorithm.pdf Tablo 1",
        aciklama="4 Onay, 3 Red. Kök entropi ≈ 0.985, kök Gini ≈ 0.4898.",
        ham_satirlar=ham,
    )


def burs_regresyon() -> VeriSeti:
    """Karar ağacı regresörü için GPA -> burs miktarı (7 satır, bin TL)."""
    gpa = [2.5, 2.7, 3.0, 3.2, 3.5, 3.7, 4.0]
    burs = [10, 12, 18, 22, 30, 35, 40]
    return VeriSeti(
        ad="burs_regresyon",
        X=np.array(gpa, dtype=float).reshape(-1, 1),
        y=np.array(burs, dtype=float),
        ozellik_adlari=["gpa"],
        hedef_adi="burs_bin_tl",
        kaynak="13-Decision_Tree_Algorithms.pdf Tablo 2",
        aciklama="Kök varyans ≈ 124.83. Notta GPA<3.1 bölmesi için varyans "
                 "azaltımı ≈ 94.64 hesaplanmıştır.",
    )


def knn_sporcu() -> VeriSeti:
    """KNN örneği: boy/kilo -> sporcu mu? (6 satır). Test noktası (168, 66)."""
    satirlar = [
        (170, 65, "Evet"),
        (165, 72, "Hayır"),
        (180, 80, "Evet"),
        (175, 85, "Hayır"),
        (160, 60, "Hayır"),
        (172, 70, "Evet"),
    ]
    X = np.array([[boy, kilo] for boy, kilo, _ in satirlar], dtype=float)
    y = np.array([1 if etiket == "Evet" else 0 for *_, etiket in satirlar], dtype=int)
    return VeriSeti(
        ad="knn_sporcu",
        X=X, y=y,
        ozellik_adlari=["boy_cm", "kilo_kg"],
        hedef_adi="sporcu",
        kaynak="12-KNN_Algorithm.pdf",
        aciklama="Yeni kişi (168 cm, 66 kg); k=3 için tahmin 'Evet'. "
                 "Öklid mesafeleri: 2.24, 6.70, 18.44, 20.25, 10.00, 5.66.",
        ham_satirlar=satirlar,
    )


def naive_bayes_spam() -> VeriSeti:
    """Bernoulli Naive Bayes örneği: Free/Win kelimeleri -> Spam mı? (5 satır)."""
    satirlar = [
        ("Yes", "Yes", "Spam"),
        ("Yes", "No", "Spam"),
        ("No", "Yes", "Ham"),
        ("Yes", "Yes", "Spam"),
        ("No", "No", "Ham"),
    ]
    X = np.array([[1 if free == "Yes" else 0, 1 if win == "Yes" else 0]
                  for free, win, _ in satirlar], dtype=int)
    y = np.array([1 if etiket == "Spam" else 0 for *_, etiket in satirlar], dtype=int)
    return VeriSeti(
        ad="naive_bayes_spam",
        X=X, y=y,
        ozellik_adlari=["free", "win"],
        hedef_adi="spam",
        kaynak="11-Naive_Bayes_Theorem_ML_Algorithm.pdf",
        aciklama="P(Spam)=3/5, P(Ham)=2/5. Test: Free=Yes, Win=No -> Spam. "
                 "P(Free=Yes|Ham)=0 olduğu için Laplace düzeltmesi gerekir.",
        ham_satirlar=satirlar,
    )


def naive_bayes_obezite() -> VeriSeti:
    """Gaussian Naive Bayes örneği: yaş/kilo/boy -> obez mi? (4 satır)."""
    satirlar = [
        (25, 80, 170, "Yes"),
        (30, 90, 165, "Yes"),
        (22, 55, 178, "No"),
        (28, 60, 180, "No"),
    ]
    X = np.array([[yas, kilo, boy] for yas, kilo, boy, _ in satirlar], dtype=float)
    y = np.array([1 if etiket == "Yes" else 0 for *_, etiket in satirlar], dtype=int)
    return VeriSeti(
        ad="naive_bayes_obezite",
        X=X, y=y,
        ozellik_adlari=["yas", "kilo_kg", "boy_cm"],
        hedef_adi="obez",
        kaynak="11-Naive_Bayes_Theorem_ML_Algorithm.pdf",
        aciklama="Test: yaş=27, kilo=85, boy=175. Sınıf başına μ ve σ "
                 "(popülasyon, ddof=0) hesaplanır: Yes için yaş μ=27.5, σ=2.5.",
        ham_satirlar=satirlar,
    )


def gb_maas() -> VeriSeti:
    """Gradient boosting regresyon örneği: tecrübe/sertifika -> maaş (4 satır)."""
    satirlar = [
        (1, "Hayır", 40),
        (3, "Evet", 60),
        (5, "Evet", 80),
        (7, "Hayır", 100),
    ]
    X = np.array([[yil, 1 if s == "Evet" else 0] for yil, s, _ in satirlar],
                 dtype=float)
    y = np.array([maas for *_, maas in satirlar], dtype=float)
    return VeriSeti(
        ad="gb_maas",
        X=X, y=y,
        ozellik_adlari=["tecrube_yil", "sertifika"],
        hedef_adi="maas",
        kaynak="17-Gradient_Boosting_Algorithm.pdf Tablo 1",
        aciklama="F0 = 70 (ortalama). İlk artıklar: -30, -10, +10, +30. "
                 "İlk ağaç: tecrübe<4 -> -20, aksi halde +20; η = 0.1.",
        ham_satirlar=satirlar,
    )


def gb_kredi_karti() -> VeriSeti:
    """Gradient boosting sınıflandırma örneği: gelir/kredi skoru -> kart onayı (4 satır)."""
    satirlar = [(30, 0, 0), (40, 0, 0), (50, 1, 1), (60, 2, 1)]
    X = np.array([[gelir, skor] for gelir, skor, _ in satirlar], dtype=float)
    y = np.array([onay for *_, onay in satirlar], dtype=int)
    return VeriSeti(
        ad="gb_kredi_karti",
        X=X, y=y,
        ozellik_adlari=["gelir_bin_tl", "kredi_skoru"],
        hedef_adi="kart_onayi",
        kaynak="17-Gradient_Boosting_Algorithm.pdf Tablo 4",
        aciklama="p = 0.5 -> F0 = log(0.5/0.5) = 0. Artıklar ±0.5. "
                 "Gelir<50 bölmesi, η = 0.1 ile F1 = ∓0.05, p = 0.487 / 0.512.",
        ham_satirlar=satirlar,
    )


def xgb_siniflandirma() -> VeriSeti:
    """XGBoost sınıflandırma örneği: maaş/kredi skoru -> onay (7 satır)."""
    satirlar = [
        (30, "Low", 0),
        (40, "Low", 0),
        (50, "Medium", 1),
        (60, "High", 1),
        (70, "Medium", 1),
        (80, "High", 1),
        (90, "High", 1),
    ]
    skor_kodu = {"Low": 0, "Medium": 1, "High": 2}
    X = np.array([[maas, skor_kodu[s]] for maas, s, _ in satirlar], dtype=float)
    y = np.array([onay for *_, onay in satirlar], dtype=int)
    return VeriSeti(
        ad="xgb_siniflandirma",
        X=X, y=y,
        ozellik_adlari=["maas_bin", "kredi_skoru"],
        hedef_adi="onay",
        kaynak="18-XGBoost_Algorithm.pdf Tablo 1",
        aciklama="5 pozitif / 2 negatif -> F0 = ln(5/2) ≈ 0.916, p0 ≈ 0.714. "
                 "Maaş≤50 bölmesi: S_sol ≈ 0.809, S_sağ ≈ 0.702, Gain ≈ 1.511; "
                 "yaprak değerleri -0.709 ve +0.631 (λ = 1).",
        ham_satirlar=satirlar,
    )


def xgb_regresyon() -> VeriSeti:
    """XGBoost regresyon örneği: maaş/kredi skoru -> harcama (7 satır)."""
    satirlar = [
        (30, "Low", 2000),
        (40, "Low", 3000),
        (50, "Medium", 4000),
        (60, "High", 5000),
        (70, "Medium", 6000),
        (80, "High", 7000),
        (90, "High", 8000),
    ]
    skor_kodu = {"Low": 0, "Medium": 1, "High": 2}
    X = np.array([[maas, skor_kodu[s]] for maas, s, _ in satirlar], dtype=float)
    y = np.array([harcama for *_, harcama in satirlar], dtype=float)
    return VeriSeti(
        ad="xgb_regresyon",
        X=X, y=y,
        ozellik_adlari=["maas_bin", "kredi_skoru"],
        hedef_adi="harcama",
        kaynak="18-XGBoost_Algorithm.pdf Tablo 5",
        aciklama="F0 = 5000. Maaş≤60 bölmesi: Gain = 21.000.000, "
                 "yaprak değerleri -1500 ve +2000 (λ = 0).",
        ham_satirlar=satirlar,
    )


def adaboost_regresyon() -> VeriSeti:
    """AdaBoost regresör örneği: tecrübe -> maaş (6 satır, bin $)."""
    tecrube = [1, 2, 3, 4, 5, 6]
    maas = [40, 45, 50, 58, 60, 65]
    return VeriSeti(
        ad="adaboost_regresyon",
        X=np.array(tecrube, dtype=float).reshape(-1, 1),
        y=np.array(maas, dtype=float),
        ozellik_adlari=["tecrube_yil"],
        hedef_adi="maas_bin_dolar",
        kaynak="16-Adaboost_Algorithm.pdf Tablo 7",
        aciklama="F0 = 53 (ortalama). İlk artıklar: -13, -8, -3, +5, +7, +12.",
    )


# --------------------------------------------------------------------------
# Kaynak notlarında tespit edilen tutarsızlıklar
# --------------------------------------------------------------------------
KAYNAK_NOTLARI = {
    "karar_agaci_kok_varyans": (
        "13-Decision_Tree_Algorithms.pdf: burs verisi için kök varyans 124.83 "
        "verilmiş. Doğrusu 113.2653'tür. Kareler toplamı 792.857, n = 7 -> "
        "792.857/7 = 113.2653 (ddof=1 ile bölünse bile 132.14 çıkar, 124.83 "
        "hiçbir bölmeyle elde edilmiyor). Notun ALT küme varyansları doğrudur "
        "(11.55 ve 44.19), yalnızca kök yanlıştır. "
        "Doğrulama: testler/test_ders_ornekleri.py::test_burs_kok_varyansi_notta_hatali"
    ),
    "karar_agaci_en_iyi_esik": (
        "13-Decision_Tree_Algorithms.pdf: gövde metni GPA<3.1 eşiğinin en yüksek "
        "varyans azaltımını verdiğini söylüyor (94.64). Tüm eşikler hesaplandığında "
        "en iyi eşik 3.35'tir: azaltım 93.12, 3.1 ise 83.06. Notun Tablo 3'ündeki "
        "özet sayılar (3.1 -> 94.6, 3.35 -> 80.3) da yeniden üretilemiyor. "
        "Not: 94.64 rakamı, hatalı kök varyanstan (124.83) türetilmiştir — "
        "124.83 - 30.19 = 94.64. Ağırlıklı varyans 30.19 doğrudur. "
        "Doğrulama: testler/test_ders_ornekleri.py::test_burs_tum_esikler"
    ),
    "adaboost_ikinci_stump_tablosu": (
        "16-Adaboost_Algorithm.pdf, Tablo 6: ikinci stump tablosunda ID 7'nin "
        "GPA'si '>3.0' yazılmış, oysa Tablo 1'de aynı ID '<3.0'. Ayrıca ID 4'ün "
        "mülakat skoru Tablo 1'de 'Düşük', Tablo 6'da 'Normal'. Tablo 6, "
        "ağırlıklı yeniden örnekleme sonrası çekilen satırları gösterdiği için "
        "ID'ler tekrar edebilir; fakat aynı ID'nin özellik değerinin değişmesi "
        "dizgi hatasıdır. Bu repo Tablo 1'i esas alır."
    ),
    "xgboost_sag_benzerlik": (
        "18-XGBoost_Algorithm.pdf, Adım 3: sağ düğüm için S = 0.702 verilmiş. "
        "Notun kendi ara değerleriyle hesaplandığında bile sonuç farklı çıkıyor: "
        "ΣR = 1.144, Σp(1-p) = 0.816, λ = 1 -> 1.144² / 1.816 = 0.7207. "
        "Yuvarlanmamış değerlerle 0.7191. Buna bağlı olarak Gain de 1.511 değil "
        "1.5292'dir. Sol düğüm (0.809) ve her iki yaprak değeri (-0.709, +0.631) "
        "doğrudur; hata yalnızca bu iki sayıdadır ve bölme kararını değiştirmez. "
        "Doğrulama: testler/test_ders_ornekleri.py::test_xgb_sag_benzerlik_notta_hatali"
    ),
    "xgboost_kok_benzerlik_gerekcesi": (
        "18-XGBoost_Algorithm.pdf: S_kök = 0 sonucu 'çünkü kök düğümde tek bir "
        "grup var' diye gerekçelendirilmiş. Gerekçe yanlış — tek grup olması "
        "similarity'yi sıfırlamaz. Doğru sebep artıkların toplamının sıfır "
        "olmasıdır: 2·(-0.7143) + 5·(0.2857) = 0, payda ne olursa olsun pay 0. "
        "Bu bir tesadüf de değildir: F₀ log-odds olarak seçildiğinde Σ(y - p) = 0 "
        "olması, F₀'ın log-loss'u minimize etmesinin birinci derece koşuludur. "
        "Yani her gradient boosting turunda kök similarity'si sıfırdan başlar."
    ),
    "xgboost_gain_formulu": (
        "18-XGBoost_Algorithm.pdf: Gain = S_sol + S_sağ - S_kök olarak verilmiş. "
        "XGBoost makalesindeki (Chen & Guestrin 2016, Denklem 7) tam biçim "
        "½·[G_sol²/(H_sol+λ) + G_sağ²/(H_sağ+λ) - (G_sol+G_sağ)²/(H+λ)] - γ'dır. "
        "½ ve γ ders notunda yok. Bölme SIRALAMASI değişmez (½ ortak çarpan), "
        "ama gerçek xgboost kütüphanesinin bastığı gain değeriyle "
        "karşılaştırırsan sayılar 2 kat farklı çıkar. "
        "gozetimli.boosting.xgb_kazanc(yarim_faktor=True) tam biçimi verir."
    ),
    "regresyon_metrikleri_eksik": (
        "Elimizdeki klasörde 1-4 numaralı PDF'ler yok. Mevcut notlarda "
        "MAE / RMSE / R² gibi regresyon metrikleri tanımlanmıyor; MSE yalnızca "
        "maliyet fonksiyonu içinde geçiyor. gozetimli.metrikler.regresyon "
        "modülündeki tanımlar standart literatürden alınmış ve scikit-learn'e "
        "karşı doğrulanmıştır."
    ),
}
