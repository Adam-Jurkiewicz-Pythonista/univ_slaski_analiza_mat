# to będzie przyklad dla Was
from analiza_fn.obrazy import KMeansObraz

# przykład obliczania klastrów odcieni szarości
obrazek1 = KMeansObraz("Fe1i5Cr1i5_obszar2_x1000_x12_y5_R1_C7.jpg") #
obrazek2 = KMeansObraz("Fe1i5Cr1i5_obszar2_x1000_x12_y5_R1_C9.jpg",
                       n_clusters=3) # 3 klastry
print(obrazek1)
print(obrazek2)

# wykonanie analizy
obrazek1.run_kmeans()
obrazek2.run_kmeans()
# zapis wyników analizy do plików graficznych
obrazek1.save_clusters()
obrazek2.save_clusters()
# pokaż wyniki
obrazek1.show_clusters()
obrazek2.show_clusters()