# Wyniki Benchmarków - Raporty 2 i 3

Ten katalog zawiera wszystkie wyniki benchmarków wydajności i profilowania pamięci dla Raportów 2 i 3.

## 📊 Pliki CSV (Dane)

### Performance Benchmarks:
- **lof_performance.csv** - Wyniki benchmarków LOF (Brute-force, KD-Tree, Parallel, kombinacje)
  - Kolumny: method, n_samples, time, speedup

- **isolation_forest_performance.csv** - Wyniki Isolation Forest z różnymi parametrami
  - Kolumny: method, n_samples, n_estimators, time

- **autoencoder_performance.csv** - Wyniki różnych architektur autoenkodera
  - Kolumny: method, n_samples, encoding_dim, time

- **dimensionality_scaling.csv** - Skalowalność algorytmów z wymiarem
  - Kolumny: algorithm, n_features, time

### Memory Profiling:
- **memory_profiling.csv** - Profilowanie zużycia pamięci LOF
  - Kolumny: method, n_samples, baseline_mb, peak_mb, increase_mb

## 📈 Pliki PNG (Wykresy)

### Raport 2 - Demonstracje podstawowe (raport2/):
- **raport2/raport2_lof_simple.png** - Prosty przykład LOF (2D z jednym punktem odstającym)
- **raport2/raport2_lof_gaussian.png** - LOF na danych gaussowskich z metrykami
- **raport2_lof_k_parameter.png** - Wpływ parametru k na wyniki LOF
- **raport2/raport2_pca_reconstruction.png** - PCA z błędem rekonstrukcji
- **raport2/raport2_pca_variance.png** - Analiza wariancji wyjaśnionej przez składowe PCA
- **raport2/raport2_pca_mahalanobis.png** - PCA z odległością Mahalanobisa
- **raport2/raport2_comparison.png** - Porównanie LOF vs PCA

### Raport 3 - LOF Optimization:
- **lof_optimization.png** - Porównanie czasu wykonania i przyspieszenia dla różnych optymalizacji LOF

### Raport 3 - Algorithm Performance:
- **isolation_forest_performance.png** - Wpływ liczby drzew i paralelizacji na wydajność IF
- **autoencoder_performance.png** - Wpływ architektury na czas treningu autoenkodera
- **dimensionality_scaling.png** - Skalowalność algorytmów z wymiarem danych

### Raport 3 - Memory Analysis:
- **memory_profiling.png** - Zużycie pamięci dla różnych wariantów LOF

## 🔄 Reprodukcja Wyników

Aby odtworzyć te wyniki:

```bash
cd notebooks

# Raport 2 - Demonstracje podstawowe
jupyter notebook raport2_basic_implementation.ipynb
# Uruchom wszystkie komórki - zapisze 6 grafów demonstracyjnych

# Raport 3 - Benchmarki wydajności
jupyter notebook raport3_performance_analysis.ipynb
# Uruchom wszystkie komórki - zapisze 5 grafów + 5 plików CSV

# Wszystkie wyniki zostaną zapisane do benchmarks/results/
```

## 📝 Kluczowe Wyniki

### LOF Optimizations (n=5000):
- Brute-force: 1.109s (baseline)
- KD-Tree: 0.213s (5.20x szybsze!)
- KD-Tree + Parallel: 0.320s (3.46x szybsze)

### Memory Usage (n=5000):
- Brute-force: 573 MB
- **KD-Tree: 0 MB** (praktycznie brak overhead!)
- Parallel: 573 MB (podobnie jak brute-force)

### Dimensionality (n=1000, d=5→50):
- **Isolation Forest: 0.076s → 0.079s** (prawie niezależny od wymiaru!)
- LOF KD-Tree: 0.017s → 0.042s (liniowe skalowanie)
- Autoencoder: ~0.23s (stały czas)

## ✅ Weryfikacja

Wszystkie liczby w tym katalogu odpowiadają danym w:
- `docs/Raport3_Optymalizacja_Analiza.md` - główna dokumentacja Raportu 3
- `notebooks/raport2_basic_implementation.ipynb` - demonstracje Raportu 2
- `notebooks/raport3_performance_analysis.ipynb` - źródło danych Raportu 3

Wyniki są w 100% reprodukowalne poprzez uruchomienie notebooków.
