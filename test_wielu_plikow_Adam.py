# to będzie przyklad dla Was
from analiza_fn.obrazy import KMeansObraz, wczytaj_pliki_z_katalogu


# sprawdzam pliki wejściowe - katalog ze zdjęciami IMAGES_DIRECTORY
pliki = wczytaj_pliki_z_katalogu()
print(pliki)

# tworzę listę obiektóœ dla każdego pliku
lista_obrazow = [ KMeansObraz(plik) for plik in pliki ]
print(lista_obrazow)

# dla każdego elementu listy o nazwie obrazek obliczam klastry
# ale nie zapisuję wynikowych plików graficznych
for obrazek in lista_obrazow:
    print(obrazek)
    obrazek.run_kmeans()
    obrazek.show_clusters()