from typing import Optional, Any, Union, List, Tuple, Dict, Type, NamedTuple
from brah.constants.tokens import *
from brah.constants.scanner import *

__all__ = [
    'Token',
    'Source',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES


class Token(NamedTuple):
    kind: str
    value: str
    line: int
    column: int

    def __str__(self) -> str:
        return f'{self.kind}:´{self._sample()}´'

    def _sample(self, length: int = 24) -> str:
        if len(self.value) <= length:
            return self.value
        return f'{self.value[:length - 3]}...'

    @property
    def len(self) -> int:
        return len(self.value)


class Source:

    @classmethod
    def load(cls, fname: str):
        with open(fname, 'r', encoding='utf8') as source_code:
            return cls(source_code.read(), fname)

    def __init__(self, source: str, filename: str):
        self.filename: str = filename
        self.src: str = source
        self.lin: int = 0
        self.col: int = 0
        self.idx: int = 0

    def __eq__(self, other: str) -> bool:
        return self.char == other

    def __ne__(self, other: str) -> bool:
        return self.char != other

    @property
    def eof(self) -> bool:
        return self.idx >= len(self.src)

    @property
    def pos(self) -> Tuple[int, int]:
        return self.lin, self.col

    @property
    def char(self) -> str:
        if self.eof:
            return '\x00'
        return self.src[self.idx]

    def next(self) -> str:
        if self.eof:
            return self.char
        if self.char == SCN_NEWLINE:
            self.lin += 1
            self.col = 0
        else:
            self.col += 1
        self.idx += 1
        return self.char

    def skip(self, count: int = 1) -> None:
        for i in range(count):
            self.next()

    def in_decdigits(self) -> bool:
        return self.char in SCN_DECDIGITS

    def in_operators(self) -> bool:
        return self.char in SCN_OPERATORS

    def in_delimiters(self) -> bool:
        return self.char in SCN_DELIMITERS

# endregion (classes)
# ---------------------------------------------------------
