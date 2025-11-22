# Projekt 7: Anomalia i Uczenie Maszynowe

**Zaawansowane zagadnienia algorytmiki i programowania**
**Rok akademicki**: 2025/2026

## Opis projektu

Projekt zajmuje się kompleksową analizą metod detekcji anomalii w uczeniu maszynowym. Implementujemy i porównujemy trzy główne algorytmy:
- **LOF** (Local Outlier Factor)
- **PCA** (Principal Component Analysis)
- **Isolation Forest**

## Status realizacji

- ✅ **Raport 1**: Teoria i plan (UKOŃCZONY)
- ✅ **Raport 2**: Implementacja podstawowa (UKOŃCZONY)
- ⏳ **Raport 3**: Rozszerzenia i optymalizacja
- ⏳ **Raport 4**: Analiza wyników i wnioski końcowe

## Struktura projektu

```
advancedAlgorithms/
├── src/                        # Kod źródłowy
│   ├── algorithms/             # Implementacje algorytmów
│   │   ├── lof.py             # Local Outlier Factor
│   │   ├── pca_anomaly.py     # PCA dla detekcji anomalii
│   │   └── isolation_forest.py # Isolation Forest 
│   ├── utils/                  # Narzędzia pomocnicze
│   └── evaluation/             # Metryki i benchmarki
├── tests/                      # Testy jednostkowe
│   ├── test_lof.py            # Testy LOF (13 testów)
│   ├── test_pca.py            # Testy PCA (21 testów)
│   └── test_isolation_forest.py # Testy IF (13 testów)
│   └── test_autoencoder.py # Testy auto_enkodera (10 testów)
├── notebooks/                  # Jupyter notebooks
│   ├── raport2_basic_implementation.ipynb
│   ├── raport3_extensions.ipynb
│   └── raport4_final_analysis.ipynb (TODO)
├── data/                       # Zbiory danych
│   ├── kdd_cup_99/
│   ├── credit_card/
│   ├── breast_cancer/
│   └── synthetic/
├── docs/                       # Dokumentacja i raporty
│   ├── Raport1_Anomalia_ML.docx
│   ├── Raport2_Anomalia_ML.docx 
│   └── ...
├── requirements.txt            # Zależności
└── README.md                   # Ten plik
```

## Instalacja

### Wymagania
- Python 3.8+
- pip lub conda

### Kroki instalacji

1. **Sklonuj repozytorium** (lub pobierz pliki)
```bash
cd advancedAlgorithms
```

2. **Utwórz środowisko wirtualne** (opcjonalnie, ale zalecane)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
venv\Scripts\activate  # Windows
```

3. **Zainstaluj zależności**
```bash
pip install -r requirements.txt
```

## Użycie

### Uruchomienie testów

```bash
# Wszystkie testy
python3 -m pytest tests/ -v

# Tylko testy LOF
python3 -m pytest tests/test_lof.py -v

# Tylko testy PCA
python3 -m pytest tests/test_pca.py -v

# Z pokryciem kodu
python3 -m pytest tests/ --cov=src --cov-report=html
```

### Uruchomienie notebooka

```bash
jupyter notebook notebooks/raport2_basic_implementation.ipynb
```

### Przykład użycia w kodzie

#### LOF (Local Outlier Factor)

```python
from src.algorithms.lof import LOF
import numpy as np

# Przygotuj dane
X = np.random.randn(100, 2)
X = np.vstack([X, [[5, 5]]])  # Dodaj outlier

# Fituj model
lof = LOF(n_neighbors=20)
scores = lof.fit_predict(X)

# Wykryj anomalie
threshold = 1.5
outliers = scores > threshold
print(f"Znaleziono {np.sum(outliers)} anomalii")
```

#### PCA Anomaly Detection

```python
from src.algorithms.pca_anomaly import PCAAnomaly
import numpy as np

# Przygotuj dane
X = np.random.randn(100, 5)

# Fituj model (reconstruction error method)
pca = PCAAnomaly(n_components=2, method='reconstruction', contamination=0.1)
pca.fit(X)

# Wykryj anomalie
labels = pca.predict(X)
print(f"Znaleziono {np.sum(labels)} anomalii")

# Alternatywnie: Mahalanobis distance
pca_maha = PCAAnomaly(n_components=2, method='mahalanobis')
pca_maha.fit(X)
distances = pca_maha.mahalanobis_distance(X)
```

## Wyniki testów

### Raport 2

Wszystkie testy jednostkowe przechodzą pomyślnie:

```
tests/test_lof.py::TestLOF  ✓ 13 passed
tests/test_pca.py::TestPCAAnomaly  ✓ 21 passed

Total: 34 passed
```

## Zbiory danych

### Planned datasets (Raport 4):

1. **KDD Cup 99** - Detekcja włamań sieciowych
2. **Credit Card Fraud Detection** - Wykrywanie oszustw kartami kredytowymi
3. **Breast Cancer Wisconsin** - Diagnostyka medyczna
4. **Synthetic Dataset** - Syntetyczne dane z kontrolowanymi anomaliami

## Metryki ewaluacji

- **Precision**: Stosunek prawdziwych anomalii do wszystkich wykrytych
- **Recall**: Stosunek wykrytych anomalii do wszystkich rzeczywistych
- **F1-score**: Średnia harmoniczna precision i recall
- **AUC-ROC**: Pole pod krzywą ROC
- **Czas wykonania**: Czas trenowania i predykcji

## Dokumentacja algorytmów

### LOF (Local Outlier Factor)

**Wzory**:
- `k-distance(p)` = odległość do k-tego najbliższego sąsiada
- `reach-dist(p, o)` = max(k-distance(o), d(p, o))
- `LRD(p)` = 1 / (średnia reach-dist do k sąsiadów)
- `LOF(p)` = (średnia LRD sąsiadów) / LRD(p)

**Złożoność**: O(n²) lub O(n log n) z KD-Tree

### PCA (Principal Component Analysis)

**Metody detekcji**:
1. **Reconstruction Error**: ||x - x_reconstructed||²
2. **Mahalanobis Distance**: √((x - μ)ᵀ Σ⁻¹ (x - μ))

**Złożoność**: O(n × d² + d³)

## Bibliografia

Zobacz plik `docs/Raport1_Anomalia_ML.docx.pdf` dla pełnej bibliografii.

Kluczowe źródła:
- Breunig et al. (2000) - LOF algorithm
- Liu et al. (2008) - Isolation Forest
- Jolliffe & Cadima (2016) - PCA review
- Aggarwal (2017) - Outlier Analysis

## Autorzy

Projekt realizowany w ramach przedmiotu "Zaawansowane zagadnienia algorytmiki i programowania"

## Licencja

Projekt edukacyjny - używaj dowolnie do celów edukacyjnych.

---

**Ostatnia aktualizacja**: 2025-11-22
**Status**: Raport 3 ukończony
