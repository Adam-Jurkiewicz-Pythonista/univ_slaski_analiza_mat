import os
from pathlib import Path

def wczytaj_pliki_z_katalogu(nazwa_katalogu_in, typy_plikow=("jpg","bmp","jpeg"), min_wielkosc=1000):
    base_dir = Path(__file__).parent.parent
    nazwa_katalogu = str(Path(base_dir, nazwa_katalogu_in))
    print(nazwa_katalogu)
    if not os.path.isdir(nazwa_katalogu):
        return False
    zwracane_pliki = []
    # rekurencyjnie sprawdza podkatalogi
    # https://github.com/abixadamj/helion-python/blob/main/Rozdzial_7/r7_00_walk.py
    for dirpath, dirname, files in os.walk(nazwa_katalogu):

        for each_file in files:
            ext = os.path.splitext(each_file)[1].lower()
            for maska in typy_plikow:
                if maska in ext:
                    plik_z_danymi = dirpath + "/" + each_file
                    if os.path.getsize(plik_z_danymi)>min_wielkosc:
                        zwracane_pliki.append(each_file)

    return zwracane_pliki


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def check_file(file_name):
    return os.path.exists(file_name)