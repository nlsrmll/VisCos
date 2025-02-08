import pandas


class ImageAnalyzer:
    def __init__(self, data: pandas.DataFrame, name: str, path: str, mean: float):
        self.data = data
        self.name = name
        self.path = path
        self.mean = mean


class ImageContainer:
    def __init__(self, data: any, name: str):
        self.data = data
        self.name = name
