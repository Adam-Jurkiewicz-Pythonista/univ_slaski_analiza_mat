from analiza_fn.obrazy import KMeansObraz,wczytaj_pliki_z_katalogu

pliki = wczytaj_pliki_z_katalogu(typy_plikow=("jpg","tif"))
print(pliki)