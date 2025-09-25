from analiza_fn.obrazy import Obraz, KMeansObraz

q = Obraz('100MA_300g_2_06.tif')
print(q)
print("inny")
p = Obraz('Fe1i5Cr1i5_obszar2_x1000_x12_y5_R1_C1.jpg')
print(p)

# to jest przykład użycia k-mean do 5 klastrów
q = KMeansObraz('100MA_300g_2_06.tif')
# wykonujemy
q.run_kmeans()
# zapisujemy klastry do plików
q.save_clusters()
