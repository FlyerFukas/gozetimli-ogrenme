"""Gözetimli öğrenme referans kütüphanesi.

Dört parça:
    gozetimli.metrikler   — sınıflandırma ve regresyon metrikleri
    gozetimli.dogrulama   — veri bölme stratejileri ve çapraz doğrulama
    gozetimli.saflik      — karar ağacı bölme ölçütleri
    gozetimli.boosting    — AdaBoost / gradient boosting / XGBoost hesapları
    gozetimli.veri        — ders notlarındaki örnek veri setleri

Hızlı başlangıç:

    >>> from gozetimli.metrikler.siniflandirma import siniflandirma_raporu
    >>> print(siniflandirma_raporu([0, 1, 1, 0], [0, 1, 0, 0], metin=True))

    >>> from gozetimli.dogrulama.bolme import TabakaliKKat
    >>> bolucu = TabakaliKKat(5, karistir=True, tohum=42)
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import boosting, dogrulama, metrikler, saflik, veri  # noqa: F401

__all__ = ["boosting", "dogrulama", "metrikler", "saflik", "veri", "__version__"]
