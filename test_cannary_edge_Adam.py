# to będzie przyklad dla Was
from analiza_fn.obrazy import CannyEdge

plik = CannyEdge("test_edge.jpg") # musi być w katalogu 'edges'

# Generuje plik z dodatkiem w nazwie '_edges_found' i zapisuje w katalogu edges
# Params:
# treshold – int 0 ... 255
plik.canny_detect(search_type='median', file_overwrite=True)
