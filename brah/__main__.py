import sys, os.path
# from typing import Optional, Any, Union, List, Tuple, Dict, Type

from brah.e_interpreter import Interpreter
from brah.constants.tokens import *

__all__ = [
    '',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES

# endregion (classes)
# ---------------------------------------------------------

# print(' '.join(sys.argv))

if len(sys.argv) >= 2:
    if os.path.exists(sys.argv[-1]):
        print(f'Loading source file at {sys.argv[-1]}')
        Interpreter(sys.argv[1]).execute()
    else:
        print(f'Unable to load "{sys.argv[-1]}"')
else:
    seconds = Interpreter('../examples/arith.brah').execute()
    print(f"Program terminated (executed in {seconds} seconds)")
