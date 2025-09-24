from analiza_fn.obrazy import Obraz, KMeansObraz

q = Obraz('/home/adasiek/PycharmProjects/univ_slaski_analiza_mat/images_in/100MA_300g_2_06.tif')
print(q)
p = Obraz('/home/adasiek/PycharmProjects/univ_slaski_analiza_mat/images_in/Fe1i5Cr1i5_obszar2_x1000_x12_y5_R1_C1.jpg')
print(p)
r = Obraz('/home/adasiek/PycharmProjects/univ_slaski_analiza_mat/images_in/trump.jpg')
print(r)

q = KMeansObraz('/home/adasiek/PycharmProjects/univ_slaski_analiza_mat/images_in/100MA_300g_2_06.tif')
print(q)
q.run_kmeans()
q.save_clusters()
