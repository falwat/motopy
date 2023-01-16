"""
Constants used by motopy.

This is a part of motopy.
"""
import numpy as np
# token's types:
# block. if..else, for, while, etc.
TT_BLOCK = "block"
# command. load, save.
TT_CMD = "cmd"
# end of line. "\r" | "\n"
# token.text used to store comment.
TT_EOL = "eol"
# identifier
TT_ID = "id"
# keyword
TT_KW = "kw"
# number. int, float or complex
TT_NUM = "num"
# operation and expression
TT_OP = "op"
# string 
TT_STR = "str"
# format string
TT_FMT_STR = 'fmt_str'
# symbol
TT_SYM = "sym"
# entry
TT_FILE = "file"


# see: MATLAB Operators and Special Characters
op_priors = {
    '()': 0, '[]': 0, '{}': 0,
    '.' : 1,
    '+?': 2, '-?': 2, '~': 2,
    "'" : 3, ".'" : 3, '^' : 3, '.^' : 3,
    '*' : 4, '/': 4, '\\': 4, '.*' : 4, './': 4,'.\\': 4, 
    '+' : 5, '-' : 5,
    ':' : 6,
    '<' : 7, '<=': 7, '>' : 7, '>=' : 7, '==' : 7, '~=' : 7, 
    '&' : 8,
    '|' : 9,
    '&&' : 10, 
    '||' : 11, 
    '=' : 12,
    ',' : 13,
    ';' : 14,
}

op_dict = {
    '~': 'not ',
    '^' : '**', 
    '.^' : '**', 
    '.*' : '*', 
    './': '/',
    '~=' : '!=', 
    '&&' : 'and', 
    '||' : 'or'
}

""" `func_name_dict` is used to replace the function name simply.
"""
func_name_dict = {
    'acos': 'np.arccos',
    'asin': 'np.arcsin',
    'atan': 'np.arctan',
    'ceil': 'np.ceil',
    'cos': 'np.cos',
    'diag': 'np.diag',
    'disp': 'print',
    'eye': 'np.eye',
    'exp': 'np.exp',
    'fft': 'np.fft',
    'fix': 'np.fix',
    'floor': 'np.floor',
    'ifft': 'np.ifft',
    'inv': 'linalg.inv',
    'linspace': 'np.linspace',
    'log': 'np.log',
    'log10': 'np.log10',
    'log2': 'np.log2',
    'ndims': 'np.ndim',
    'numel': 'np.size',
    'pinv': 'linalg.pinv',
    'rank': 'linalg.matrix_rank',
    'round': 'np.round',
    'sin': 'np.sin',
    'sort': 'np.sort',
    'sqrt': 'np.sqrt',
    'unique': 'np.unique',
}

binary_operators = ['.', '^', '.^', '*', '/', '\\', '.*', './', '.\\', '+', '-', ':', '<', '<=', '>', '>=', '==', '~=',  '&', '|', '&&',  '||',  '=']
left_unary_operators = ['+?', '-?', '~']
right_unary_operators = ["'", ".'"]

bracket_pairs = {'(':')', '[':']', '{':'}'}  

matlab_blocks = ["if", "elseif", "else", "switch", "case", "otherwise",
    "for", "while", "try", "catch", "parfor", "function"]

# matlab keywords
matlab_keywords = [
    "if", "elseif", "else", "switch", "case", "otherwise",
    "for", "while", "try", "catch", "break", "return", "continue",
    "pause", "parfor", "end", 'function'
]

matlab_commands = ['clc', 'clf', 'close', 'clear', 'load', 'save', 'global']

value_dict = {
    'true':True,
    'false':False,
    'inf': np.Inf,
    'Inf': np.Inf,
    'nan': np.NaN,
    'NaN': np.NaN
}