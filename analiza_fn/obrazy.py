import os
import cv2
import logging
import json
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
from pathlib import Path


# ważne! tego nie dotykaj!
base_dir = Path(__file__).parent.parent
RUN_DATE = datetime.now().strftime("%Y%m%d-%H%M%S")

# przykład: ścieżka do podfolderu ze zdjęciami - tu zmieniamy ewentualnie!
IMAGES_DIRECTORY = str(base_dir / "images_in")
IMAGES_CLUSTER_DIRECTORY = str(base_dir / "images_in/clusters-") + RUN_DATE
LOGS_DIRECTORY = str(base_dir / "logs")
#########



if not os.path.exists(IMAGES_CLUSTER_DIRECTORY):
    os.makedirs(IMAGES_CLUSTER_DIRECTORY)

if not os.path.exists(LOGS_DIRECTORY):
    os.makedirs(LOGS_DIRECTORY)
logging.basicConfig(
    filename=f"{LOGS_DIRECTORY}/Obrazy-{RUN_DATE}.log",
    level=logging.DEBUG,
    format="%(levelname)s %(message)s",
)


# funkcje ogólnego przeznaczenia
def wczytaj_pliki_z_katalogu(typy_plikow=("jpg", "bmp", "jpeg"), min_wielkosc=1000):
    zwracane_pliki = []
    # rekurencyjnie sprawdza podkatalogi
    # https://github.com/abixadamj/helion-python/blob/main/Rozdzial_7/r7_00_walk.py
    # patrzymy na pliki tylko w 1 katalogu BEZ podkatalogów
    files = [
        f
        for f in os.listdir(IMAGES_DIRECTORY)
        if os.path.isfile(os.path.join(IMAGES_DIRECTORY, f))
    ]

    for each_file in files:
        ext = os.path.splitext(each_file)[1].lower()
        for maska in typy_plikow:
            if maska in ext:
                plik_z_danymi = IMAGES_DIRECTORY + "/" + each_file
                if os.path.getsize(plik_z_danymi) > min_wielkosc:
                    zwracane_pliki.append(each_file)

    return zwracane_pliki


class Obraz:
    def __init__(self, file_name, additional_images_dir=None):
        """

        :param file_name:
        :param additional_images_dir: dodatkowy, jeśli istnieje, to nadpisuje IMAGES_DIRECTORY
        """

        self.images_directory = (
            additional_images_dir if additional_images_dir is not None else IMAGES_DIRECTORY
        )

        if not os.path.exists(self.images_directory):
            raise Exception(f"Directory {self.images_directory} does not exist!!!")

        self.file_name = file_name
        self.file_name_only = os.path.splitext(os.path.basename(self.file_name))[0]
        self.file_extension = os.path.splitext(os.path.basename(self.file_name))[1]
        self.file_name_bw = None
        self.image_raw = None
        self.file_read_ok = None

        if not os.path.exists(self.images_directory + "/" + file_name):
            logging.error(f"File {file_name} not found in {self.images_directory}")
            raise Exception(
                f"__Class__ Obrazy: File {file_name} not found in {self.images_directory}"
            )
        if not self.read_file():
            self.change_file_color2bw()

        logging.info(
            f"File -- INIT Complete {self.file_name=} in {self.images_directory}"
        )

    def __str__(self):
        return f"{self.file_name} - {self.image_raw.shape} / {self.image_raw.dtype}"

    @staticmethod
    def image_save2file(image_raw, new_file_name, overwrite=False):
        try:
            if not os.path.isfile(new_file_name):
                cv2.imwrite(new_file_name, image_raw)
                logging.info(f"File saved -- {new_file_name=}")
                return True
            else:
                logging.info(f"File already exists -- {new_file_name=}")
                if overwrite:
                    os.remove(new_file_name)
                    cv2.imwrite(new_file_name, image_raw)
                    logging.info(f"File overwritten -- {new_file_name=}")
        except Exception as e:
            logging.error(f"image_save2file {image_raw} - {e=}")
            return False

    def read_file(self):
        if self.file_name is None:
            return False

        logging.info(f"Reading {self.file_name}")
        try:
            if self.file_read_ok is None:
                self.image_raw = cv2.imread(
                    self.images_directory + "/" + self.file_name, cv2.IMREAD_GRAYSCALE
                )
                self.file_read_ok = True
                logging.info(
                    f"Read complete {self.file_name} - {self.image_raw.shape} / {self.image_raw.dtype}"
                )
                return True
            return None
        except Exception as e:
            logging.error(f"Failed to read {self.file_name} => {e=}")
            return False

    def change_file_color2bw(self):
        if self.file_read_ok is None:
            img_tmp = cv2.imread(self.images_directory + "/" + self.file_name)
            self.file_name_bw = self.file_name_only + "_bw" + self.file_extension
            logging.info(
                f"Change {self.file_name} from {img_tmp.shape} / {self.image_raw.dtype} => cv2.IMREAD_GRAYSCALE {self.file_name_bw}"
            )
            try:
                self.image_raw = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
                self.image_save2file(
                    self.image_raw, self.images_directory + "/" + self.file_name_bw
                )
                # new reading
                self.file_read_ok = None
                self.file_name = self.images_directory + "/" + self.file_name_bw
                if self.read_file():
                    return True
            except Exception as e:
                logging.error(f"Failed to change color {self.file_name_bw} => {e=}")
                return False


class KMeansObraz(Obraz):
    def __init__(
        self,
        file_name,
        n_clusters=5,
        lista_klastrow_do_wydzielenia=None,
        clasters_init="k-means++",
    ):
        """

        To jest dokumentacja dla Was
        :param file_name:
        :param n_clusters: liczba klatrów, domyślnie 5
        :param lista_klastrow_do_wydzielenia: np.: (1,3,4) - domyślnie wszystkie
        :param clasters_init: domyślnie 'k-means++', lub 'random'
        """
        super().__init__(file_name)
        self.n_clusters = n_clusters
        self.clusters_init = clasters_init
        self.kmeans = None
        self.img_pixels = None  # total pixels
        self.img_clusters = []  # lista obrazów
        self.centers_names = None
        self.proc_clusters = {"kluster": ["pixeli", "procent"]}  # wartości
        if lista_klastrow_do_wydzielenia is None:
            lista_klastrow_do_wydzielenia = tuple(range(n_clusters))
        self.lista_klastrow_do_wydzielenia = lista_klastrow_do_wydzielenia

    def run_kmeans(self):
        if self.kmeans is True:
            logging.info(f"Already ran KMeans on {self.n_clusters} old_cls")
            return False

        img = self.image_raw
        # Spłaszcz obraz do 1-wymiarowej tablicy (lista pikseli)
        pixels = img.reshape(-1, 1)
        self.img_pixels = len(pixels)
        log = f"Kmean start {self.file_name_only=} {img.shape=} {self.n_clusters=} {self.img_pixels=} "
        logging.info(log)

        # Model KMeans(n_clusters=cluster, init='xxxxxxxx')
        # init{‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
        kmeans = KMeans(
            n_clusters=self.n_clusters, init=self.clusters_init
        )  # Liczba klastrów (odcieni)
        kmeans.fit(pixels)
        # Centroidy - reprezentatywne odcienie - posortowane
        centers = kmeans.cluster_centers_.flatten().astype(np.uint8)

        # Przypisania klastrów dla pikseli
        labels = kmeans.labels_
        self.centers_names = centers
        logging.info(f"Remember {centers=}")

        # Odtworzenie obrazu na podstawie centroidów (redukcja ilości odcieni)
        segmented_img = centers[labels].reshape(img.shape)

        # wydzielenie obrazu
        for cluster in self.lista_klastrow_do_wydzielenia:
            wycinek = centers[cluster]
            background = 0 if wycinek > 126 else 255
            segmented_img_wycinek = segmented_img.copy()
            segmented_img_wycinek[segmented_img_wycinek != wycinek] = background
            # zliczam ilość pixeli
            count = np.count_nonzero(segmented_img_wycinek == wycinek)
            procent = count / self.img_pixels * 100
            logging.info(f"Liczba pixeli dla {wycinek=} {count=} {procent=}")
            # teraz zapis takiego obrazka do self.klastry
            self.img_clusters.append(segmented_img_wycinek)
            self.proc_clusters[int(wycinek)] = [count, round(procent, 2)]
        else:
            self.kmeans = True

    def save_clusters(self):
        if self.kmeans is None:
            logging.info(f"Run first run_kmeans on {self.file_name} ")
        for idx, image in enumerate(self.img_clusters):
            new_file_name = f"{IMAGES_CLUSTER_DIRECTORY}/{self.file_name_only}_{self.centers_names[idx]}_{self.file_extension}"
            self.image_save2file(image, new_file_name)

    def show_clusters(self):
        if self.kmeans is None:
            logging.info(f"Run first run_kmeans on {self.file_name} ")
            return False
        # return self.proc_clusters
        print(self.file_name, json.dumps(self.proc_clusters))


class CannyEdge(Obraz):
    def __init__(self, file_name, edge_directory_last="edges"):
        """

        :param file_name:
        :param edge_directory_last: domyślnie 'edges'
        """
        edge_directory = str(base_dir / edge_directory_last)
        # + "-" + RUN_DATE
        # najpierw spradzamy
        if not os.path.exists(edge_directory):
            raise Exception(f"Directory {edge_directory=} does not exist!!!")

        self.edge_directory = edge_directory
        self.edges_found = None
        logging.info(f"Edge detection for file {file_name=}")
        super().__init__(file_name, additional_images_dir=self.edge_directory)

    def canny_detect(
        self, search_type="auto", treshold=(127, 127), file_overwrite=False
    ):
        """
        Generuje plik z dodatkiem w nazwie '_edges_found' i zapisuje w katalogu edges
        :param search_type: 'auto' - treshold 127, 'median' - oblicza medianę
        :param treshold: tuple (int 0 ... 255, int 0 ... 255)
        :param file_overwrite: bool - domyślnie False | True
        :return:
        """
        logging.info(
            f"Canny Edge detection for file {self.file_name=} / {self.image_raw.shape=}"
        )
        threshold1, threshold2 = treshold
        if search_type == "median":
            med_val = np.median(self.image_raw)
            threshold1 = int(max(0, 0.7 * med_val))
            threshold2 = int(min(255, 1.3 * med_val))
        elif search_type == "custom":
            pass

        self.edges_found = cv2.Canny(
            image=self.image_raw, threshold1=threshold1, threshold2=threshold2
        )
        edges_file_name = (
            self.edge_directory
            + "/"
            + self.file_name_only
            + "_edges_found"
            + self.file_extension
        )
        self.image_save2file(
            self.edges_found, edges_file_name, overwrite=file_overwrite
        )

class X(Obraz):
    def __init__(self, file_name):
        # tu trzeba dać tylko 1 wiersz, by działało


    def dowolna_metoda(self):
        print(f"Jakas metoda: {self.file_name=}")


pliczek = X("plik_graficzny.jpg") # musi być w xx_obrazy
pliczek.dowolna_metoda()