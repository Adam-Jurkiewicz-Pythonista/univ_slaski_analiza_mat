# univ_slaski_analiza_mat
Analiza obrazów dla Uniwersytetu Śląskiego - Python OpenCV

---

## Informacje dla KMeansObraz:

## Informacja dla Canny

Wywołujemy z nazwą pliku, który musi znajdowa się w podkatalogu `edges` lub podajemy taki podkatalog jako parametr - patrz docstring.
`plik = CannyEdge("test_edge.jpg")`

Potem wołamy metodę wyszukiwania brzegów, np:
```
plik.canny_detect(search_type='median', file_overwrite=True)
#
Generuje plik z dodatkiem w nazwie '_edges_found' i zapisuje w katalogu edges
Params:
search_type – 'auto' - treshold 127, 'median' - oblicza medianę
treshold – int 0 ... 255
file_overwrite – bool - domyślnie False | True
```

Po tym w katalogu `edges` pojawia się plik z wykrytymi edges.