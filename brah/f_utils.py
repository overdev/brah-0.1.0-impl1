import sys
from typing import NamedTuple, Any

__all__ = [
    'error',
    'error_if',
    'assertion',
    'Location',
]


# ---------------------------------------------------------
# region CONSTANTS & ENUMS

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS


def error(location: 'Location', kind: str, message: str, *args, **kwargs):
    debug = kwargs.get('debug')
    if debug:
        del kwargs['debug']
    msg = location.format_error(kind, message, *args, **kwargs)
    if debug:
        assert False, message.format(*args, **kwargs)
    else:
        print(msg, file=sys.stderr)
        exit(0)


def error_if(condition: Any, location: 'Location', kind: str, message: str, *args, **kwargs):
    if condition:
        error(location, kind, message, *args, **kwargs)


def assertion(condition: Any, location: 'Location', kind: str, message: str, *args, **kwargs):
    if not condition:
        error(location, kind, message, *args, **kwargs)


# endregion (functions)
# ---------------------------------------------------------
# region CLASSES


class Location(NamedTuple):
    line_start: int
    line_end: int
    start: int
    end: int
    line_index: int
    fname: str
    source: str

    @property
    def line(self) -> str:
        return self.source[self.line_start: self.line_end]

    @property
    def token(self) -> str:
        return self.source[self.start: self.end]

    def format_error(self, kind: str, message: str, *args, **kwargs) -> str:
        line_number = "{: 8}".format(self.line_index)
        err = f"File \"{self.fname}\", line {self.line_index}, in module"
        frame_width = 78
        src_line = self.line
        line_len = len(src_line)
        if line_len > frame_width:
            center = self.start + (len(self.token) // 2)
            start = max(0, center - frame_width // 2)
            end = min(start + frame_width, len(self.line[start:]))
            src_line = self.line[start: end]
            hilite_start = self.start - start
            hilite_end = self.end - start
            st = '...' if start else '   '
            nd = '...' if len(src_line) > frame_width else ''
        else:
            hilite_start = self.start - self.line_start
            hilite_end = self.end - self.line_start
            st = nd = ''
        hilite = (" " * hilite_start) + ("^" * (hilite_end - hilite_start))
        formatted = message.format(*args, **kwargs)

        line = f"\n{err}\n\n{line_number}│{st}{src_line}{nd}\n         {st}{hilite}{nd}\n{kind} ERROR: {formatted}"

        return line

# endregion (classes)
# ---------------------------------------------------------
