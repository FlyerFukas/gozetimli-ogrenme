"""Veri bölme stratejileri, ders notundaki 5 çapraz doğrulama yöntemi + grup/tekrar.

Ders notu (7-Types_of_Cross_Validation) şunları anlatır:
    Leave-One-Out, Leave-P-Out, K-Fold, Stratified K-Fold, Time Series CV.
Hepsi burada var. Ek olarak pratikte gerekli olan `GrupKKat` (aynı kişiye/mağazaya
ait satırlar aynı katta kalsın) ve `TekrarliKKat` (tek bir bölmenin şansına
bağlı kalmamak için) eklenmiştir.

Ortak sözleşme
--------------
Her bölücü `bol(X, y=None, gruplar=None)` çağrısında (egitim_indeksleri,
test_indeksleri) çiftleri üretir ve `kat_sayisi(...)` ile kaç bölme yapacağını
söyler. scikit-learn ile uyum için `split` ve `get_n_splits` takma adları da
tanımlıdır; bu sayede bu bölücüler doğrudan `sklearn.model_selection`
fonksiyonlarına `cv=` olarak geçirilebilir.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = [
    "KKat",
    "TabakaliKKat",
    "BirDisarida",
    "PDisarida",
    "ZamanSerisiBolme",
    "GrupKKat",
    "TekrarliKKat",
    "egitim_test_bol",
]


def _ornek_sayisi(X) -> int:
    dizi = np.asarray(X)
    if dizi.ndim == 0:
        raise ValueError("X en az 1 boyutlu olmalı.")
    return dizi.shape[0]


class _Bolucu:
    """Ortak iskelet: sklearn takma adlarını ve tekrarlı temsili tek yerde tutar."""

    def split(self, X, y=None, groups=None):
        """scikit-learn uyumlu takma ad, `bol` ile aynı."""
        return self.bol(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        """scikit-learn uyumlu takma ad, `kat_sayisi` ile aynı."""
        return self.kat_sayisi(X, y, groups)

    def __repr__(self):
        alanlar = ", ".join(f"{ad}={deger!r}" for ad, deger
                            in sorted(vars(self).items())
                            if not ad.startswith("_"))
        return f"{type(self).__name__}({alanlar})"


class KKat(_Bolucu):
    """K-Fold: veri k eşit parçaya bölünür, her turda bir parça test olur.

    Ders notundaki avantajlar: dengeli doğrulama sağlar, LOOCV'ye göre ucuzdur.
    Her örnek tam olarak bir kez test setine düşer, yani tüm veri hem eğitimde
    hem testte kullanılmış olur.

    karistir=True verildiğinde bölmeden önce satırlar karıştırılır. Verinin
    sıralı geldiği durumlarda (ör. önce tüm 0'lar, sonra tüm 1'ler) karıştırmamak
    katları tamamen tek sınıflı yapar, bu sessiz ve ölümcül bir hatadır.
    ZAMAN SERİSİNDE KARIŞTIRMA: geleceği görüp geçmişi tahmin etmiş olursun.
    """

    def __init__(self, kat_sayisi: int = 5, *, karistir: bool = False,
                 tohum: int | None = None):
        if kat_sayisi < 2:
            raise ValueError(f"kat_sayisi en az 2 olmalı; {kat_sayisi} geldi.")
        self.katlar = int(kat_sayisi)
        self.karistir = bool(karistir)
        self.tohum = tohum
        if not karistir and tohum is not None:
            raise ValueError(
                "tohum yalnızca karistir=True iken anlamlıdır; "
                "karıştırmadan bölme zaten deterministiktir."
            )

    def kat_sayisi(self, X=None, y=None, gruplar=None) -> int:
        return self.katlar

    def bol(self, X, y=None, gruplar=None):
        n = _ornek_sayisi(X)
        if self.katlar > n:
            raise ValueError(
                f"kat_sayisi ({self.katlar}) örnek sayısından ({n}) büyük olamaz."
            )
        indeksler = np.arange(n)
        if self.karistir:
            np.random.default_rng(self.tohum).shuffle(indeksler)

        # n, katlara tam bölünmüyorsa ilk (n % k) kat bir fazla örnek alır.
        boyutlar = np.full(self.katlar, n // self.katlar, dtype=int)
        boyutlar[: n % self.katlar] += 1
        baslangic = 0
        for boyut in boyutlar:
            test = indeksler[baslangic: baslangic + boyut]
            egitim = np.concatenate([indeksler[:baslangic],
                                     indeksler[baslangic + boyut:]])
            baslangic += boyut
            yield egitim, test


class TabakaliKKat(_Bolucu):
    """Stratified K-Fold: her katta sınıf oranları korunur.

    Ders notu: "Katmanlardaki etiket oranlarını (ör. %60 1 ve %40 0) her
    katmanda korumaya çalışır." Dengesiz veride VARSAYILAN seçim budur;
    düz K-Fold, %2 pozitifli bir veri setinde hiç pozitif içermeyen bir test
    katı üretebilir ve o katın recall'u tanımsız olur.

    Sınıflandırmada `KKat` yerine neredeyse her zaman bunu kullan.
    """

    def __init__(self, kat_sayisi: int = 5, *, karistir: bool = False,
                 tohum: int | None = None):
        if kat_sayisi < 2:
            raise ValueError(f"kat_sayisi en az 2 olmalı; {kat_sayisi} geldi.")
        self.katlar = int(kat_sayisi)
        self.karistir = bool(karistir)
        self.tohum = tohum

    def kat_sayisi(self, X=None, y=None, gruplar=None) -> int:
        return self.katlar

    def bol(self, X, y=None, gruplar=None):
        if y is None:
            raise ValueError("TabakaliKKat için y (etiketler) zorunludur.")
        y = np.asarray(y).ravel()
        n = _ornek_sayisi(X)
        if y.shape[0] != n:
            raise ValueError(
                f"y uzunluğu ({y.shape[0]}) örnek sayısına ({n}) eşit olmalı."
            )
        siniflar, sayimlar = np.unique(y, return_counts=True)
        if sayimlar.min() < self.katlar:
            raise ValueError(
                f"En az örnekli sınıfın ({siniflar[np.argmin(sayimlar)]!r}) "
                f"{sayimlar.min()} örneği var; bu kat_sayisi={self.katlar} "
                "için yetersiz. Kat sayısını düşür veya sınıfları birleştir."
            )

        uretec = np.random.default_rng(self.tohum)
        # Her sınıfın örneklerini kendi içinde katlara dağıt; kat kimliklerini
        # tek bir dizide topla. Böylece her kat, sınıf oranlarını korur.
        kat_kimligi = np.empty(n, dtype=int)
        for sinif in siniflar:
            sinif_indeksleri = np.where(y == sinif)[0]
            if self.karistir:
                uretec.shuffle(sinif_indeksleri)
            kat_kimligi[sinif_indeksleri] = np.arange(
                sinif_indeksleri.size) % self.katlar

        for kat in range(self.katlar):
            test = np.where(kat_kimligi == kat)[0]
            egitim = np.where(kat_kimligi != kat)[0]
            yield egitim, test


class BirDisarida(_Bolucu):
    """Leave-One-Out (LOOCV): her turda tek bir gözlem test edilir, n tur yapılır.

    Ders notundaki dezavantajlar aynen geçerli:
      - Büyük veri setlerinde çok yavaş (n kez eğitim).
      - Kat skorları birbirine çok bağımlı olduğundan varyans tahmini güvenilmez;
        tek örnekten hesaplanan accuracy ya 0 ya 1'dir.
    n < ~100 olan küçük veri setleri dışında K-Fold tercih edilir.
    """

    def kat_sayisi(self, X=None, y=None, gruplar=None) -> int:
        if X is None:
            raise ValueError("BirDisarida kat sayısı için X gereklidir.")
        return _ornek_sayisi(X)

    def bol(self, X, y=None, gruplar=None):
        n = _ornek_sayisi(X)
        if n < 2:
            raise ValueError("BirDisarida için en az 2 örnek gerekir.")
        indeksler = np.arange(n)
        for i in range(n):
            yield np.delete(indeksler, i), np.array([i])


class PDisarida(_Bolucu):
    """Leave-P-Out: her turda p gözlem test edilir, TÜM kombinasyonlar denenir.

    Tur sayısı C(n, p), kombinatorik patlama gerçektir:
        n=20, p=2  ->    190 tur
        n=20, p=5  -> 15 504 tur
        n=50, p=5  -> 2 118 760 tur
    Ders notundaki "hesaplama maliyeti yüksektir" uyarısı bu yüzden ciddiye
    alınmalı. `guvenlik_siniri` bu patlamayı erken yakalar.
    """

    def __init__(self, p: int = 2, *, guvenlik_siniri: int = 100_000):
        if p < 1:
            raise ValueError(f"p en az 1 olmalı; {p} geldi.")
        self.p = int(p)
        self.guvenlik_siniri = int(guvenlik_siniri)

    def kat_sayisi(self, X=None, y=None, gruplar=None) -> int:
        if X is None:
            raise ValueError("PDisarida kat sayısı için X gereklidir.")
        from math import comb
        return comb(_ornek_sayisi(X), self.p)

    def bol(self, X, y=None, gruplar=None):
        n = _ornek_sayisi(X)
        if self.p >= n:
            raise ValueError(
                f"p ({self.p}) örnek sayısından ({n}) küçük olmalı."
            )
        toplam = self.kat_sayisi(X)
        if toplam > self.guvenlik_siniri:
            raise ValueError(
                f"Leave-{self.p}-Out bu veri için C({n},{self.p}) = {toplam:,} "
                f"tur demek; güvenlik sınırı {self.guvenlik_siniri:,}. "
                "Kat sayısı sabit olan KKat kullanmayı düşün "
                "veya guvenlik_siniri değerini bilinçli olarak yükselt."
            )
        indeksler = np.arange(n)
        for test_ikilisi in combinations(range(n), self.p):
            test = np.array(test_ikilisi)
            yield np.delete(indeksler, test), test


class ZamanSerisiBolme(_Bolucu):
    """Time Series CV: eğitim her zaman testten ÖNCE gelir.

    Ders notu: "Geçmiş veriler eğitimde, ileri tarihli veriler doğrulamada
    kullanılır. Veri sırası mutlaka korunmalıdır."

    Zaman serisinde K-Fold kullanmak, geleceği görüp geçmişi tahmin etmek
    demektir; skor gerçekte olmayacak bir başarıyı gösterir. Bu, veri
    sızıntısının (data leakage) en sık yapılan biçimidir.

    pencere:
        "genisleyen"  eğitim seti her turda büyür (0..k ile k+1 test edilir)
        "kayan"       eğitim seti sabit uzunlukta kayar (eski veri düşer)
    bosluk: eğitim ile test arasında atlanacak örnek sayısı. Hedef gelecekteki
        h adımı tahmin ediyorsa (ör. 7 gün sonrası) buraya h-1 koymak,
        eğitim penceresinin test hedefine sızmasını engeller.
    """

    def __init__(self, kat_sayisi: int = 5, *, test_boyutu: int | None = None,
                 bosluk: int = 0, pencere: str = "genisleyen",
                 en_az_egitim: int | None = None):
        if kat_sayisi < 1:
            raise ValueError(f"kat_sayisi en az 1 olmalı; {kat_sayisi} geldi.")
        if pencere not in ("genisleyen", "kayan"):
            raise ValueError(
                f"pencere 'genisleyen' veya 'kayan' olmalı; '{pencere}' geldi."
            )
        if bosluk < 0:
            raise ValueError("bosluk negatif olamaz.")
        self.katlar = int(kat_sayisi)
        self.test_boyutu = test_boyutu
        self.bosluk = int(bosluk)
        self.pencere = pencere
        self.en_az_egitim = en_az_egitim

    def kat_sayisi(self, X=None, y=None, gruplar=None) -> int:
        return self.katlar

    def bol(self, X, y=None, gruplar=None):
        n = _ornek_sayisi(X)
        test_boyutu = (self.test_boyutu if self.test_boyutu is not None
                       else n // (self.katlar + 1))
        if test_boyutu < 1:
            raise ValueError(
                f"Örnek sayısı ({n}) {self.katlar} kat için yetersiz: "
                "her katta en az 1 test örneği kalmıyor."
            )
        # İlk test bloğunun başlangıcı: sondan geriye doğru katlar yerleştirilir.
        ilk_test_basi = n - self.katlar * test_boyutu
        if ilk_test_basi - self.bosluk < 1:
            raise ValueError(
                f"Örnek sayısı ({n}) bu ayar için yetersiz: kat_sayisi="
                f"{self.katlar}, test_boyutu={test_boyutu}, bosluk={self.bosluk}. "
                "İlk katta eğitim verisi kalmıyor."
            )
        for kat in range(self.katlar):
            test_bas = ilk_test_basi + kat * test_boyutu
            test_son = test_bas + test_boyutu
            egitim_son = test_bas - self.bosluk
            if self.pencere == "genisleyen":
                egitim_bas = 0
            else:
                uzunluk = (self.en_az_egitim if self.en_az_egitim is not None
                           else ilk_test_basi - self.bosluk)
                egitim_bas = max(0, egitim_son - uzunluk)
            if self.en_az_egitim is not None and \
                    egitim_son - egitim_bas < self.en_az_egitim:
                raise ValueError(
                    f"{kat}. katta eğitim uzunluğu {egitim_son - egitim_bas}, "
                    f"en_az_egitim={self.en_az_egitim} şartını sağlamıyor."
                )
            yield np.arange(egitim_bas, egitim_son), np.arange(test_bas, test_son)


class GrupKKat(_Bolucu):
    """Aynı gruba ait satırlar asla eğitim ve testte birlikte bulunmaz.

    Ders notlarında yok ama gerçek veride en sık karşılaşılan sızıntı kaynağı:
    aynı hastanın 5 ölçümü, aynı mağazanın 300 günü, aynı kullanıcının 40
    tıklaması. Satır bazlı bölersen model "bu hastayı zaten gördüm" der ve
    test skoru gerçek dışı çıkar.

    Gruplar, kat başına toplam boyut dengelenecek şekilde (en büyük grup en boş
    kata) dağıtılır; bu yüzden katlar tam eşit boyutlu olmayabilir.
    """

    def __init__(self, kat_sayisi: int = 5):
        if kat_sayisi < 2:
            raise ValueError(f"kat_sayisi en az 2 olmalı; {kat_sayisi} geldi.")
        self.katlar = int(kat_sayisi)

    def kat_sayisi(self, X=None, y=None, gruplar=None) -> int:
        return self.katlar

    def bol(self, X, y=None, gruplar=None):
        if gruplar is None:
            raise ValueError("GrupKKat için gruplar zorunludur.")
        gruplar = np.asarray(gruplar).ravel()
        n = _ornek_sayisi(X)
        if gruplar.shape[0] != n:
            raise ValueError(
                f"gruplar uzunluğu ({gruplar.shape[0]}) örnek sayısına "
                f"({n}) eşit olmalı."
            )
        benzersiz, sayimlar = np.unique(gruplar, return_counts=True)
        if benzersiz.size < self.katlar:
            raise ValueError(
                f"Grup sayısı ({benzersiz.size}) kat sayısından "
                f"({self.katlar}) küçük olamaz."
            )
        # Açgözlü denge: büyük gruplardan başlayarak en hafif kata yerleştir.
        kat_yukleri = np.zeros(self.katlar, dtype=int)
        grup_kati = {}
        for idx in np.argsort(-sayimlar):
            hedef = int(np.argmin(kat_yukleri))
            grup_kati[benzersiz[idx].item()] = hedef
            kat_yukleri[hedef] += sayimlar[idx]

        kat_kimligi = np.array([grup_kati[g] for g in gruplar.tolist()])
        for kat in range(self.katlar):
            yield np.where(kat_kimligi != kat)[0], np.where(kat_kimligi == kat)[0]


class TekrarliKKat(_Bolucu):
    """K-Fold'u farklı karıştırmalarla n kez tekrarlar (toplam k·n bölme).

    Tek bir 5-kat bölmenin skoru, o rastgele bölmenin şansını da içerir. Küçük
    veri setlerinde 5x2 veya 10x5 tekrarlı CV, ortalamanın etrafındaki
    belirsizliği görmenin en ucuz yoludur.
    """

    def __init__(self, kat_sayisi: int = 5, *, tekrar: int = 3,
                 tohum: int | None = None, tabakali: bool = False):
        self.katlar = int(kat_sayisi)
        self.tekrar = int(tekrar)
        self.tohum = tohum
        self.tabakali = bool(tabakali)
        if self.tekrar < 1:
            raise ValueError("tekrar en az 1 olmalı.")

    def kat_sayisi(self, X=None, y=None, gruplar=None) -> int:
        return self.katlar * self.tekrar

    def bol(self, X, y=None, gruplar=None):
        ana_uretec = np.random.default_rng(self.tohum)
        for _ in range(self.tekrar):
            alt_tohum = int(ana_uretec.integers(0, 2**32 - 1))
            sinif = TabakaliKKat if self.tabakali else KKat
            bolucu = sinif(self.katlar, karistir=True, tohum=alt_tohum)
            yield from bolucu.bol(X, y, gruplar)


def egitim_test_bol(*diziler, test_orani: float = 0.2, tohum: int | None = None,
                    karistir: bool = True, tabaka=None):
    """Tek seferlik eğitim/test ayrımı (train_test_split karşılığı).

    tabaka verilirse (genelde y) sınıf oranları iki tarafta da korunur.

    Ders notundaki train/validation/test ayrımını kurmak için iki kez çağır:
        X_gecici, X_test, y_gecici, y_test = egitim_test_bol(X, y, test_orani=0.2, tabaka=y)
        X_egitim, X_dog, y_egitim, y_dog  = egitim_test_bol(
            X_gecici, y_gecici, test_orani=0.25, tabaka=y_gecici)
    Böylece 60/20/20 olur. Hiperparametre ve eşik seçimi DOĞRULAMA setinde
    yapılır; test seti yalnızca en sonda, bir kez kullanılır.
    """
    if not diziler:
        raise ValueError("En az bir dizi verilmelidir.")
    n = _ornek_sayisi(diziler[0])
    for i, dizi in enumerate(diziler[1:], start=1):
        if _ornek_sayisi(dizi) != n:
            raise ValueError(
                f"{i}. dizinin uzunluğu ({_ornek_sayisi(dizi)}) ilk dizininkinden "
                f"({n}) farklı."
            )
    if not 0.0 < test_orani < 1.0:
        raise ValueError(f"test_orani (0, 1) aralığında olmalı; {test_orani} geldi.")

    test_boyutu = int(round(n * test_orani))
    test_boyutu = max(1, min(n - 1, test_boyutu))
    uretec = np.random.default_rng(tohum)

    if tabaka is None:
        indeksler = np.arange(n)
        if karistir:
            uretec.shuffle(indeksler)
        test_idx, egitim_idx = indeksler[:test_boyutu], indeksler[test_boyutu:]
    else:
        tabaka = np.asarray(tabaka).ravel()
        if tabaka.shape[0] != n:
            raise ValueError("tabaka uzunluğu veri uzunluğuna eşit olmalı.")
        siniflar, sayimlar = np.unique(tabaka, return_counts=True)
        if sayimlar.min() < 2:
            raise ValueError(
                f"Tabakalama için her sınıfta en az 2 örnek gerekir; "
                f"{siniflar[np.argmin(sayimlar)]!r} sınıfında {sayimlar.min()} var."
            )
        test_parcalari = []
        for sinif, sayi in zip(siniflar, sayimlar):
            sinif_idx = np.where(tabaka == sinif)[0]
            if karistir:
                uretec.shuffle(sinif_idx)
            pay = max(1, int(round(sayi * test_orani)))
            pay = min(pay, sayi - 1)  # eğitimde de en az 1 örnek kalsın
            test_parcalari.append(sinif_idx[:pay])
        test_idx = np.concatenate(test_parcalari)
        egitim_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        if karistir:
            uretec.shuffle(test_idx)
            uretec.shuffle(egitim_idx)

    sonuc = []
    for dizi in diziler:
        dizi = np.asarray(dizi)
        sonuc.extend([dizi[egitim_idx], dizi[test_idx]])
    return tuple(sonuc)
