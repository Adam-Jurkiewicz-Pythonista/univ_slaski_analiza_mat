from analiza_fn.obrazy import CannyEdge, KMeansObraz

pliczek = CannyEdge("plik_graficzny.jpg",edge_directory="xx_obrazy") # musi być w edges
pliczek.canny_detect()

plik = KMeansObraz("20250829_144656.jpg", kmean_directory= "images_in")
plik.run_kmeans()
plik.show_clusters_values()
plik.save_clusters()
