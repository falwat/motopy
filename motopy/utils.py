
class ScanError(Exception):
    def __init__(self, text: str, row: int, col: int) -> None:
        msg = f'the text "{text}" started at (row: {row}, col: {col}) scan error.'
        super().__init__(msg)

class ParseError(Exception):
    def __init__(self, text:str, row:int, col:int) -> None:
        msg = f'the token "{text}" at (row: {row}, col: {col}) parse error.'
        super().__init__(msg)

class TransError(Exception):
    def __init__(self, text:str, filename: str, row:int, col:int, more:str='') -> None:
        msg = f'the token "{text}" at (row: {row}, col: {col}) translate error. {more}'
        super().__init__(msg)

class ExecError(Exception):
    def __init__(self, expr:str, filename: str, row: int) -> None:
        msg = f'the statment "{expr}" at line {row} of "{filename}.m" execute error.'
        super().__init__(msg)

