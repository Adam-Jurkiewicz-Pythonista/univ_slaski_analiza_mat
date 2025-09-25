from analiza_fn.obrazy import Obraz, KMeansObraz

q = Obraz('100MA_300g_2_06.tif')
print(q)
print("inny")
p = Obraz('Fe1i5Cr1i5_obszar2_x1000_x12_y5_R1_C1.jpg')
print(p)
r = Obraz('trump.jpg')
print(r)

q = KMeansObraz('100MA_300g_2_06.tif')
print(q)
q.run_kmeans()
q.save_clusters()
