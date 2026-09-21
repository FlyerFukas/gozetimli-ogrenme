"""Karar ağacı bölme ölçütleri: entropi, Gini, bilgi kazancı, varyans azaltımı.

Ders notu (13-Decision_Tree_Algorithms) şunu söyler: "amaç, veri setini öyle
bölmek ki her bölme sonucunda oluşan alt kümeler mümkün olduğunca homojen
(tek sınıflı) olsun." Bu modül o homojenliği ölçen fonksiyonları ve en iyi
bölmeyi arayan yardımcıları içerir.

Sınıflandırma ölçütleri saflığı ÖLÇER (küçük = saf), bölme kalitesi ise
kazanç (gain) olarak hesaplanır: kök saflıksızlığı eksi ağırlıklı çocuk
saflıksızlığı.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "entropi",
    "gini",
    "siniflandirma_hatasi",
    "varyans",
    "agirlikli_saflik",
    "bilgi_kazanci",
    "kazanc_orani",
    "varyans_azaltimi",
    "esik_adaylari",
    "en_iyi_bolunme",
]

_OLCUTLER = {}


def entropi(y, *, taban: float = 2.0) -> float:
    """Shannon entropisi: H = -Σ p_i · log(p_i).

    Bilgi teorisinden gelir ve "bu kümedeki belirsizlik kaç bit?" sorusunu
    yanıtlar. Tek sınıflı küme 0, iki sınıfın yarı yarıya olduğu küme 1 bit.
    Aralık: [0, log_taban(sınıf sayısı)].

    ID3 ve C4.5 bu ölçütü kullanır. Gini'ye göre biraz daha pahalıdır
    (logaritma) ve dengesiz bölmeleri biraz daha sert cezalandırır.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    _, sayimlar = np.unique(y, return_counts=True)
    olasiliklar = sayimlar / y.size
    # p = 0 olan sınıflar toplama katkı vermez (0·log0 = 0 kabul edilir).
    olasiliklar = olasiliklar[olasiliklar > 0]
    return float(-np.sum(olasiliklar * (np.log(olasiliklar) / np.log(taban))))


def gini(y) -> float:
    """Gini saflıksızlığı: G = 1 - Σ p_i².

    Ders notundaki tanım: "rastgele seçilen bir örneğin yanlış sınıflandırılma
    olasılığı". Aralık: [0, 1 - 1/k]. İki sınıfta en fazla 0.5.

    CART'ın varsayılan ölçütüdür. Logaritma içermediği için entropiden hızlıdır
    ve pratikte neredeyse hep aynı ağacı üretir, ders notunun "genelde benzer
    sonuçlar verir" ifadesi doğrudur.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    _, sayimlar = np.unique(y, return_counts=True)
    olasiliklar = sayimlar / y.size
    return float(1.0 - np.sum(olasiliklar ** 2))


def siniflandirma_hatasi(y) -> float:
    """Yanlış sınıflandırma oranı: 1 - max(p_i).

    En kaba saflık ölçütü. Ağaç BÜYÜTMEK için kötüdür (türevi çoğu bölmede
    sıfırdır, yani iyileşmeyi göremez), fakat BUDAMA (pruning) aşamasında
    kullanılır çünkü doğrudan hatayı temsil eder.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    _, sayimlar = np.unique(y, return_counts=True)
    return float(1.0 - sayimlar.max() / y.size)


def varyans(y) -> float:
    """Popülasyon varyansı (ddof=0), regresyon ağacının saflıksızlık ölçütü.

    Ders notundaki hesap da ddof=0 kullanır: kareler toplamı / n.
    DİKKAT: pandas'ın `.var()` varsayılanı ddof=1'dir ve küçük yapraklarda
    belirgin şekilde farklı sayı verir.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 2))


_OLCUTLER.update({
    "entropi": entropi,
    "gini": gini,
    "hata": siniflandirma_hatasi,
    "varyans": varyans,
})


def _olcut_al(olcut):
    if callable(olcut):
        return olcut
    if olcut not in _OLCUTLER:
        raise ValueError(
            f"olcut {sorted(_OLCUTLER)} içinden biri ya da çağrılabilir olmalı; "
            f"'{olcut}' geldi."
        )
    return _OLCUTLER[olcut]


def agirlikli_saflik(alt_kumeler, *, olcut="entropi") -> float:
    """Σ (|S_i| / |S|) · saflık(S_i), bölme sonrası ağırlıklı saflıksızlık.

    Ağırlık şart: 1 örnekli saf bir yaprak ile 100 örnekli saf bir yaprak
    aynı değerde sayılamaz.
    """
    fonksiyon = _olcut_al(olcut)
    alt_kumeler = [np.asarray(alt).ravel() for alt in alt_kumeler]
    toplam = sum(alt.size for alt in alt_kumeler)
    if toplam == 0:
        return 0.0
    return float(sum(alt.size / toplam * fonksiyon(alt) for alt in alt_kumeler))


def bilgi_kazanci(y, alt_kumeler=None, *, maske=None, olcut="entropi") -> float:
    """Bilgi kazancı = saflık(kök) - Σ ağırlık_i · saflık(çocuk_i).

    İki kullanım biçimi:
        bilgi_kazanci(y, [sol_y, sag_y])         , alt kümeler doğrudan
        bilgi_kazanci(y, maske=bool_dizi)        , ikili bölme maskesiyle

    Ders notundaki örnek (7 örnek: 4 Onay, 3 Red; "Interview Score = Yüksek"
    bölmesi) 0.521 bit kazanç verir ve `testler/test_ders_ornekleri.py`
    bu sayıyı birebir doğrular.
    """
    fonksiyon = _olcut_al(olcut)
    y = np.asarray(y).ravel()
    if alt_kumeler is None:
        if maske is None:
            raise ValueError("alt_kumeler veya maske verilmelidir.")
        maske = np.asarray(maske, dtype=bool).ravel()
        if maske.size != y.size:
            raise ValueError(
                f"maske uzunluğu ({maske.size}) y uzunluğuna ({y.size}) eşit olmalı."
            )
        alt_kumeler = [y[maske], y[~maske]]
    return float(fonksiyon(y) - agirlikli_saflik(alt_kumeler, olcut=olcut))


def kazanc_orani(y, alt_kumeler=None, *, maske=None) -> float:
    """C4.5'in kazanç oranı = bilgi kazancı / bölme bilgisi (split info).

    Neden var? Ham bilgi kazancı, çok değerli kategorik özellikleri kayırır:
    "müşteri kimliği" gibi her satırda farklı olan bir sütun, her yaprağı tek
    örnekli yapar, entropiyi sıfırlar ve en yüksek kazancı alır, ama hiçbir
    şey öğretmez. Bölme bilgisine bölmek bu kayırmayı giderir.

    SplitInfo = -Σ (|S_i|/|S|) · log2(|S_i|/|S|)
    """
    y = np.asarray(y).ravel()
    if alt_kumeler is None:
        if maske is None:
            raise ValueError("alt_kumeler veya maske verilmelidir.")
        maske = np.asarray(maske, dtype=bool).ravel()
        alt_kumeler = [y[maske], y[~maske]]
    kazanc = bilgi_kazanci(y, alt_kumeler, olcut="entropi")
    oranlar = np.array([np.asarray(alt).size / y.size for alt in alt_kumeler])
    oranlar = oranlar[oranlar > 0]
    bolme_bilgisi = float(-np.sum(oranlar * np.log2(oranlar)))
    if bolme_bilgisi == 0:
        return 0.0
    return float(kazanc / bolme_bilgisi)


def varyans_azaltimi(y, alt_kumeler=None, *, maske=None) -> float:
    """Varyans azaltımı = Var(kök) - Σ (|S_i|/|S|) · Var(S_i).

    Regresyon ağacının bölme ölçütü. Ders notundaki formülün aynısıdır ve
    ağırlıklı MSE azalışına denktir: aynı bölmeyi seçerler.
    """
    return bilgi_kazanci(y, alt_kumeler, maske=maske, olcut="varyans")


def esik_adaylari(sutun) -> np.ndarray:
    """Sürekli bir özellik için denenecek eşikler: ardışık farklı değerlerin orta noktaları.

    Ders notundaki GPA örneğinde olduğu gibi: [2.5, 2.7, 3.0, ...] ->
    [2.6, 2.85, ...]. Orta nokta kullanmak, eşiğin iki gözlem arasına
    düşmesini garanti eder; ham değeri eşik yapmak "<= mi < mü" belirsizliği
    yaratır ve aynı bölmeyi iki kez denemeye yol açar.
    """
    degerler = np.unique(np.asarray(sutun, dtype=float))
    if degerler.size < 2:
        return np.array([])
    return (degerler[:-1] + degerler[1:]) / 2.0


def en_iyi_bolunme(X, y, *, olcut="entropi", en_az_yaprak: int = 1,
                   ozellik_adlari=None):
    """Tüm özellikler ve tüm eşikler için kazancı hesaplayıp en iyisini döndürür.

    Karar ağacının tek bir düğümde yaptığı işin tamamı budur; CART bunu
    özyinelemeli olarak tekrarlar.

    Dönüş: {"ozellik", "ozellik_adi", "esik", "kazanc", "sol_sayi", "sag_sayi",
            "tum_adaylar"} sözlüğü. Geçerli bölme yoksa kazanc = 0.0 ve
            esik = None döner.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.size:
        raise ValueError(
            f"X satır sayısı ({X.shape[0]}) y uzunluğuna ({y.size}) eşit olmalı."
        )

    en_iyi = {"ozellik": None, "ozellik_adi": None, "esik": None,
              "kazanc": 0.0, "sol_sayi": 0, "sag_sayi": 0}
    adaylar = []

    for sutun_no in range(X.shape[1]):
        sutun = X[:, sutun_no]
        for esik in esik_adaylari(sutun):
            maske = sutun <= esik
            sol, sag = int(maske.sum()), int((~maske).sum())
            if sol < en_az_yaprak or sag < en_az_yaprak:
                continue
            kazanc = bilgi_kazanci(y, maske=maske, olcut=olcut)
            ad = (ozellik_adlari[sutun_no] if ozellik_adlari is not None
                  else f"ozellik_{sutun_no}")
            adaylar.append({"ozellik": sutun_no, "ozellik_adi": ad,
                            "esik": float(esik), "kazanc": float(kazanc),
                            "sol_sayi": sol, "sag_sayi": sag})
            if kazanc > en_iyi["kazanc"]:
                en_iyi = adaylar[-1].copy()

    en_iyi["tum_adaylar"] = sorted(adaylar, key=lambda a: -a["kazanc"])
    return en_iyi
