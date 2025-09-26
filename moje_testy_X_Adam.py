from analiza_fn.obrazy import X, CannyEdge

pliczek = CannyEdge("test_pliku.abc",edge_directory_last="xx_obrazy") # musi być w edges
print(pliczek)
pliczek.show_image()
pliczek.canny_detect()
pliczek.show_image()
# pliczek.blur_image()
# pliczek.show_image()
# pliczek.show_original_image()
# #
# pliczek3 = X("plik_graficzny.jpg")
# print(pliczek3)
# pliczek2 = X("test_not.jpg")
# print(pliczek2)
