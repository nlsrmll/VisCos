from dataclasses import dataclass


@dataclass
class File:
    extension: str
    filename: str

    def __str__(self) -> str:
        return self.filename + self.extension

    def __repr__(self) -> str:
        return self.filename + self.extension

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return str(self) == other
        if isinstance(other, File):
            return self.filename == other.filename and self.extension == other.extension
