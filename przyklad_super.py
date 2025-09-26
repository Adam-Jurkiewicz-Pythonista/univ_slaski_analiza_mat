class Podstawowa:
    def __init__(self, file_name, directory=None):
        print("Podstawowa:", file_name, directory)

class Inna(Podstawowa):
    def __init__(self, file_name, dir="test"):
        print("Inna:", file_name, dir)
        super().__init__(file_name, directory=dir)

plik1 = Podstawowa("a")
plik2 = Podstawowa("b", directory="b_dir")
plik3 = Inna("c")
plik4 = Inna("d", dir="d_dir")
