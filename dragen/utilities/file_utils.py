import pandas as pd

class FileUtils:
    """General file utilities."""

    def __init__(self):
        pass  # LOGGER initialization can be added here

    def read_input(self, file_name):
        """Reads the given input file and returns the volume along with radii list.
        Parameter :
        file_name : String, name of the input file
        """
        data = pd.read_csv(filename)
        radius_a, radius_b, radius_c = ([] for i in range(3))
        data.sort_values(by=['Volumen'], ascending=False, inplace=True)
        if 'a' in data.head(0):
            for rad in data['a']:
                radius_a.append(rad)
        if 'b' in data.head(0):
            for rad in data['b']:
                radius_b.append(rad)
        if 'c' in data.head(0):
            for rad in data['c']:
                radius_c.append(rad)
        # LOGGER: add 'CSV Gesamtvolumen: sum(volume)'
        return radius_a, radius_b, radius_c, data['Volumen']
