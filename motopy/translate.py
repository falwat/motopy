"""
translate tokens.

This is a part of motopy.
"""
import chardet
import json
import logging
import numpy as np
import os
import re
from typing import Union
from .constants import *
from .token import Token
from .scan import scan
from .parse import parse
from .generate import generate


def translate_add_num(token: Token, num: Union[int, float], **kwargs):
    text = generate(token)[0]
    if token.ttype == TT_NUM:
        token.value += num
        token.text = str(token.value)
    else:
        token.set_lchild(token.move())
        token.set_rchild(Token(TT_NUM, str(num), value=num))
        token.text = '+'
        token.ttype = TT_OP
        token.value = token.lchild.value + num
    token.translated = True
    logging.debug(f'translate "{text}" in line {token.row:3} to {generate(token)[0]}')

def translate_subtract_num(token: Token, num: Union[int, float], **kwargs):
    text = generate(token)[0]
    if token.ttype == TT_NUM:
        token.value -= num
        token.text = str(token.value)
    else:
        token.set_lchild(token.move())
        token.set_rchild(Token(TT_NUM, str(num), value=num))
        token.ttype = TT_OP
        token.text = '-'
        token.value = token.lchild.value - num
    token.translated = True
    logging.debug(f'translate "{text}" in line {token.row:3} to {generate(token)[0]}')

def translate_to_int(token: Token, **kwargs):
    text = generate(token)[0]
    root = token.root
    if isinstance(token.value, np.ndarray):
        if issubclass(token.value.dtype.type, np.integer):
            return
        else:
            root.add_import_as('numpy', 'np')
            fname = 'np.int32'
    elif not isinstance(token.value, int):
        fname = 'int'
    else:
        return
    _token = root.build_func(fname, token.move())
    _token.value = eval(generate(_token)[0], root._globals, root._locals)
    token.replaced_with(_token)
    logging.info(f'translate the value of "{text}" in line {token.row:3} to int')

def translate_to_range(token: Token, **kwargs):
    assert token.ttype == TT_OP and token.text == ':'
    text = generate(token)[0]
    root = token.root
    ltoken = token.lchild
    rtoken = token.rchild
    if ltoken.text == ':':
        # a:step:b --> np.arange(a, b, step)
        token.set_children([ltoken.lchild, rtoken, ltoken.rchild])
        if (token.children[1].value - token.children[0].value) % token.children[2].value == 0:
            translate_add_num(token.children[1], token.children[2].value, **kwargs)
    else:
        # a:b
        token.set_children([ltoken, rtoken])
        if (token.children[1].value - token.children[0].value) % 1 == 0:
            translate_add_num(rtoken, 1, **kwargs)
        
    # a:step:b" and "a:b"" in "for" block will be translate to "range()"
    # when the value of a, b, step is integer
    if token.parent.text == 'for..in' and all([type(t.value) == int for t in token.children]):
        token_fname = Token(TT_ID, 'range', root=root)
    else:
        token.root.add_import_as('numpy', 'np')
        token_fname = Token(TT_ID, 'np.arange', root=root)
    token.text = '?()'
    token.set_lchild(token_fname)
    token.set_rchild()
    token.value = eval(generate(token)[0], token.root._globals, token.root._locals)
    token.translated = True
    logging.warning(f'translate "{text}" to "{generate(token)[0]}". ' +
            'Please manually confirm the result.')

def translate_assign_copy(token: Token, **kwargs):
    assert token.ttype == TT_ID and isinstance(token.value, np.ndarray)
    root = token.root
    # x --> x.copy()
    token.translated = True
    token_dot = Token(TT_OP, '.', lchild=token.move(), rchild=Token(TT_ID, 'copy', translated=True), root=root)
    token.ttype = TT_OP
    token.text = '?()'
    token.set_lchild(token_dot)
    token.value = eval(generate(token)[0], root._globals, root._locals)

def translate_format_text(fmt_text: str, tokens: 'list[Token]'):
    re_ph = re.compile(r'%\d*\.?\d*[cdeEfgGiosuxX]')
    placeholders = re_ph.findall(fmt_text)
    for i, ph in enumerate(placeholders):
        pos = fmt_text.find(ph)
        if isinstance(tokens[i].value, np.ndarray):
            opt_text = ''
        else:
            opt_text = ':' + fmt_text[pos+1:pos+len(ph)]
        fmt_text = fmt_text[:pos] + '{' + generate(tokens[i])[0] + opt_text + '}' + fmt_text[pos+len(ph):]
    return fmt_text

def translate_tic_toc(token: Token, **kwargs):
    if token.text == 'tic':
        if token.parent != None and token.parent.text == '=':
            # timerVal = tic
            pass
        else:
            # tic
            token_tic = token.move()
            token_time = token.move()
            token_time.ttype = TT_ID
            token_time.text = 'time()'
            token.ttype = TT_OP
            token.text = '='
            token.set_lchild(token_tic)
            token.set_rchild(token_time)
            token.root.add_imports('time', 'time')
            exec(generate(token)[0], token.root._globals, token.root._locals)
            token.translated = True
    if token.text == 'toc':
        if token.parent != None and token.parent.text == '=':
            # elapsedTime = toc
            token_tid = token.move()
            token_tid.text = 'tic'
            token.ttype = TT_OP
            token.text = '-'
            token.set_lchild(Token(TT_ID, 'time()'))
            token.set_rchild(token_tid)
            token.value = eval(generate(token)[0], token.root._globals, token.root._locals)
            token.translated = True
        elif token.parent != None and token.parent.text == '?()':
            # toc(timerVal) or elapsedTime = toc(timerVal)
            pass
        else:
            # toc
            token_sub = token.move()
            token_sub.ttype = TT_OP
            token_sub.text = '-'
            token_sub.set_lchild(Token(TT_ID, 'time()'))
            token_sub.set_rchild(Token(TT_ID, 'tic'))
            token.ttype = TT_OP
            token.text = '?()'
            token.set_lchild(Token(TT_ID, 'print'))
            token.set_children([Token(TT_STR, 'Elapse'), token_sub, Token(TT_STR, 'second.')])

def trans_func_audioread(token: Token, **kwargs):
    token.root.add_imports('scipy.io', 'wavfile')
    token.lchild.text = 'wavfile.read'
    token_assign = token.parent
    if token.is_rchild() and token_assign.text == '=' and \
        token_assign.lchild.text == ',':
            # data, fs --> fs, data
            token_assign.lchild.set_children(token_assign.lchild.children[::-1])
    token.value = eval(generate(token)[0], token.root._globals, token.root._locals)

def trans_func_cell(token: Token, **kwargs):
    vnames = ['_c', '_r', '_k', '_m', '_n']
    token_list = Token(TT_ID, 'None', value=None, translated=True)
    root = token.root
    if len(token.children) == 1:
        tokens_sz = token.children * 2
    else:
        tokens_sz = token.children
    for i, t in enumerate(tokens_sz[::-1]):
        token_list = root.build_list(root.build_for_in(vnames[i], root.build_func('range', t), token_list))
    token.set_children(token_list.children)
    token.set_lchild()
    token.text = '[]'
    token.root.exec_(f'__ans = {generate(token)[0]}')
    token.value = token.root._locals['__ans']
    token.translated = all([t.translated for t in token.children])
        
def trans_func_dir(token: Token, **kwargs):
    root = token.root
    root.add_imports('os', 'scandir')
    token_list = root.build_list(root.build_for_in('e', root.build_func('scandir', token.children[0]), ts=root.build_dict(
        name=Token(TT_ID, 'e.name'), 
        folder=Token(TT_ID, 'e.path'), 
        isdir=Token(TT_ID, 'e.is_dir()'))))
    token.text = '[]'
    token.set_lchild()
    token.set_children(token_list.children)
    
def trans_func_find(token: Token, **kwargs):
    root = token.root
    root.add_import_as('numpy', 'np')
    cond = token.children[0].value
    if isinstance(cond, np.ndarray) and cond.ndim == 1:
        _token = Token(TT_OP, '?[]', lchild= root.build_func('np.nonzero', token.children[0]), 
            children=[root.build_token(0)], root=root)
        token.replaced_with(_token)
        token = _token
    else:
        token.lchild.text = 'np.nonzero'
    token.value = eval(generate(token)[0], token.root._globals, token.root._locals)
    
def trans_func_sprintf(token: Token, **kwargs):
    text = token.children[0].text
    token.text = translate_format_text(text, token.children[1:])
    token.ttype = TT_FMT_STR
    token.value = token.text
    token.children = []

def trans_func_fprintf(token: Token, **kwargs):
    fmt_token = token.children[0] 
    if isinstance(token.children[0].value, str):
        fmt_token.text = translate_format_text(fmt_token.text, token.children[1:])
        fmt_token.ttype = TT_FMT_STR
        fmt_token.value = fmt_token.text
        token.children = [fmt_token]
        token.lchild.text = 'print'

def trans_func_max(token: Token, **kwargs):
    token.root.add_import_as('numpy', 'np')
    token.lchild.text = 'np.amax'
    token.value = eval(generate(token)[0], token.root._globals, token.root._locals)

def trans_func_length(token: Token, **kwargs):
    root = token.root
    # length(a) --> max(np.shape(a))
    root.add_import_as('numpy', 'np')
    token.lchild.text = 'max'
    token.set_children([root.build_func('np.shape', token.children[0])])
    token.value = eval(generate(token)[0], token.root._globals, token.root._locals)

def trans_func_load(token: Token, **kwargs):
    root = token.root
    _globals = root._globals
    _locals = root._locals
    arg_tokens = token.children
    # If you do not specify filename, the load function searches for a file named matlab.mat.
    if len(token.children) == 0:
        filename = 'matlab.mat'
    elif token.children[0].ttype == TT_STR:
        filename = token.children[0].text
    else:
        filename = token.children[0].value
    if token.parent != None and token.parent.text == '=':
        # v = load(filename)
        if filename.endswith('.mat'):
            root.add_imports('scipy.io', 'loadmat')
            token.lchild.text = 'loadmat'
            token.value = eval(generate(token)[0], _globals, _locals)
        else:
            # If filename has an extension other than .mat, 
            # the load function treats the file as ASCII data.
            root.add_import_as('numpy', 'np')
            token.lchild.text = 'np.loadtxt'
            token.children.append(Token(TT_OP, '=', 
                lchild=Token(TT_ID, 'ndmin'),
                rchild=Token(TT_NUM, '2')))
            text = generate(token, **kwargs)[0]
            # logging.info(f'translate line {token.row} to: "{text}"')
            token.value = eval(text, _globals, _locals)
    else:
        # no returns
        if filename.endswith('.mat'):
            root.add_imports('scipy.io', 'loadmat')
            _token = token.move()
            _token.lchild.text = 'loadmat'
            token.text = '='
            token.set_lchild(Token(TT_ID, '_mat'))
            token.set_rchild(_token)
            token.set_children()
            exec(generate(token)[0], _globals, _locals)
            _mat:dict = _locals['_mat']    
            _token = token.move()
            if len(arg_tokens) > 1:
                varnames = [t.text for t in token.children[1:]]
            else:
                varnames = []
                for n in _mat:
                    if n not in ['__header__', '__version__', '__globals__']:
                        varnames.append(n)
            children = [_token]
            for k in varnames:
                children.append(Token(TT_OP, '=', 
                    lchild=Token(TT_ID, k), 
                    rchild=Token(TT_OP, '?[]', 
                        lchild=Token(TT_ID, '_mat'), 
                        children=[Token(TT_STR, k)])))
                exec(generate(children[-1])[0], _globals, _locals)
            token.set_children(children)  
            token.text = ';'
        else:
            _token = token.move()
            vname = os.path.splitext(os.path.split(filename)[-1])[0]
            token.root.add_import_as('numpy', 'np')
            _token.lchild.text = 'np.loadtxt'
            _token.children.append(Token(TT_OP, '=', 
                lchild=Token(TT_ID, 'ndmin'),
                rchild=Token(TT_NUM, '2')))
            token.text = '='
            token.set_lchild(Token(TT_ID, vname))
            token.set_rchild(_token)
            token.set_children()
            exec(generate(token)[0], _globals, _locals)


def trans_func_size(token: Token, **kwargs):
    if len(token.children) == 2:
        _token = token.move()
        token_func = _token.lchild
        token_func.text = 'shape'
        token_var = _token.children[0]
        token_dim = _token.children[1]
        translate_subtract_num(token_dim, 1, **kwargs)
        _token.set_lchild(token_var)
        _token.set_rchild(token_func)
        _token.text = '.'
        token.set_lchild(_token)
        token.set_children([token_dim])
        token.text = '?[]'
        token.value = eval(generate(token)[0], token.root._globals, token.root._locals)

def trans_func_strcat(token: Token, **kwargs):
    # strcat(s1, s2) --> ''.join([s1, s2])
    root = token.root
    func = root.build_func("''.join", root.build_list(*token.children))
    token.lchild.text = "''.join"
    token.set_children(func.children)
    token.value = eval(generate(token)[0], token.root._globals, token.root._locals)

def trans_func_zeros(token: Token, **kwargs):
    root = token.root
    token.lchild.text = 'np.' + token.lchild.text
    root.add_import_as('numpy', 'np')
    for t in token.children:
        if not isinstance(t.value, int):
            translate_to_int(t, **kwargs)
    if len(token.children) == 1:
        token.set_children([Token(TT_OP, '()', children=[token.children[0], token.children[0]], root=root)])
    elif len(token.children) > 1:
        token.children = [Token(TT_OP, '()', children=token.children, root=root)]
    token.value = eval(generate(token)[0], root._globals, root._locals)
    
func_dict = {
    'audioread': trans_func_audioread,
    'cell': trans_func_cell,
    'dir': trans_func_dir,
    'find': trans_func_find,
    'sprintf': trans_func_sprintf,
    'fprintf': trans_func_fprintf,
    'max': trans_func_max,
    'length': trans_func_length,
    'load': trans_func_load,
    'ones': trans_func_zeros,
    'size': trans_func_size,
    'strcat': trans_func_strcat,
    'zeros': trans_func_zeros,
}

def translate_expr_assign(token: Token, **kwargs):
    _globals = token.root._globals
    _locals = token.root._locals
    ltoken = token.lchild
    rtoken = token.rchild
    if rtoken.ttype == TT_ID and ltoken.ttype == TT_ID and \
        isinstance(rtoken.value, np.ndarray):
        # >> y=x translate to: y = x.copy()
        translate_assign_copy(rtoken, **kwargs)
        text = generate(token)[0]
        exec(text, _globals, _locals)
        token.value = rtoken.value
        token.translated = True
    elif ltoken.text == '.':
        # s.m = v --> s = {'m': v}
        token_s = ltoken.lchild
        token_m = ltoken.rchild
        token_m.ttype = TT_STR
        token_v = token.rchild
        token_dict = Token(TT_OP, '{}', 
            children=[Token(TT_OP, ':', 
                lchild=token_m,
                rchild=token_v)])
        token.set_lchild(token_s)
        token.set_rchild(token_dict)
        if token_s.ttype != TT_ID:
            translate_expr_assign(token, **kwargs)
        else:
            text = generate(token)[0]
            exec(text, _globals, _locals)
            token.value = _locals[token.lchild.text]
            token.translated = True
    else:
        _locals['__ans'] = rtoken.value
        exec(f'{generate(ltoken)[0]} = __ans', _globals, _locals)
        token.value = rtoken.value
        token.translated = ltoken.translated & rtoken.translated

def translate_expr_fcall(token: Token, **kwargs):
    assert token.ttype == TT_OP and token.text == '?()'
    root = token.root
    _globals = root._globals
    _locals = root._locals
    ltoken = token.lchild
    functions: 'dict[str, Token]' = kwargs['functions']
    cur_func:Token = kwargs['cur_func']
    if ltoken.text in _locals:
        if eval(f'callable({ltoken.text})', _globals, _locals):
            # this is a function call
            token.value = eval(generate(token)[0], _globals, _locals)
        elif isinstance(_locals[ltoken.text], (list, np.ndarray)):
            # this is a array or cell slice
            token.text = '?[]' 
            for subtoken in token.children:
                if subtoken.text != ':':
                    translate_subtract_num(subtoken, 1, **kwargs)
                    translate_to_int(subtoken, **kwargs)
            token.value = eval(generate(token)[0], _globals, _locals)
        else:
            msg = f'the type of variable "{ltoken.text}" not allowed called or slice.'
            logging.fatal(msg)
            raise Exception(msg)
    elif ltoken.translated == True:
        token.value = eval(f'{generate(token)[0]}', _globals, _locals)
    elif ltoken.text in kwargs['replaced_functions']:
        m, fn = kwargs['replaced_functions'][ltoken.text]
        exec(f'from {m} import {fn}', _globals, _locals)
        root.add_imports(m, fn)
        token.value = eval(f'{generate(token)[0]}', _globals, _locals)

    func = cur_func
    while func != None:
        if  ltoken.text in func.functions:
            translate_function(func.functions[ltoken.text], token.children, **kwargs)
            token.value = func.functions[ltoken.text].value
            return
        else:
            func = func.parent

    if ltoken.text in functions and functions[ltoken.text].translated == False:
        # this is a local function.
        translate_function(functions[ltoken.text], token.children, **kwargs)
        token.value = functions[ltoken.text].value
    elif ltoken.text in func_dict:
        func_dict[ltoken.text](token, **kwargs)
    elif ltoken.text in func_name_dict:
        ltoken.text = func_name_dict[ltoken.text]
        if ltoken.text.startswith('np.'):
            root.add_import_as('numpy', 'np')
        elif ltoken.text.startswith('linalg.'):
            root.add_imports('scipy', 'linalg')
        elif ltoken.text.startswith('signal.'):
            root.add_imports('scipy', 'signal')
        elif ltoken.text.startswith('integrate.'):
            root.add_imports('scipy', 'integrate')
        elif ltoken.text.startswith('random.'):
            root.add_imports('numpy', 'random')
        ltoken.translated = True
        _argin = [t.value for t in token.children]
        root._locals['_argin'] = _argin
        root.exec_(f'_ans = {ltoken.text}(*_argin)')
        token.value = root._locals['_ans']
        token.translated = all([t.translated for t in token.children])
    elif f'{ltoken.text}.m' in  os.listdir(kwargs['input_path']):
        filename = f'{ltoken.text}.m'
        logging.info(f'found "{filename}" in "{kwargs["input_path"]}"')
        _root = load_file(file_basename=ltoken.text,
            input_path=kwargs['input_path'],
            output_path=kwargs['output_path'],
            logging_level=kwargs['logging_level'])
        functions[ltoken.text] = _root.functions[ltoken.text]
        translate_function(functions[ltoken.text], token.children, **kwargs)
        token.value = functions[ltoken.text].value
        root.add_imports(ltoken.text, ltoken.text, noexe=True)
    else:
        logging.warning(f'no translate rule for {ltoken.text}()')


def translate_expression(token: Token, **kwargs):
    assert token.ttype in [TT_OP, TT_SYM, TT_ID, TT_NUM, TT_STR] or token.text == 'end'
    root = token.root
    _globals = root._globals
    _locals = root._locals
    functions: 'dict[str, Token]' = kwargs['functions']
    ltoken = token.lchild
    rtoken = token.rchild
    if token.ttype == TT_NUM: 
        token.value = eval(token.text, _globals, _locals)
        token.translated = True
    elif token.ttype in [TT_STR, TT_FMT_STR]:
        token.value = token.text
        token.translated = True
    elif token.ttype == TT_ID:
        if token.text in _locals:
            token.value = _locals[token.text]
            token.translated = True
        elif token.text in _globals:
            token.value = _globals[token.text]
            token.translated = True
        elif token.is_lchild() and token.parent.text == '=':
            token.translated = True
        elif token.text in value_dict:
            token.ttype = TT_NUM
            token.text = value_dict[token.text]
            if token.text.startswith('np.'):
                root.add_import_as('numpy', 'np')
            token.value = eval(token.text, _globals, _locals)
            token.translated = True
        elif token.text in ['tic', 'toc']:
            translate_tic_toc(token, **kwargs)
        else:
            token.translated = False

    elif token.ttype == TT_OP:
        # () [] {} . +? -? ~ ' .' ^ .^ * / \ .* ./ .\ : < <= > >= == ~=
        # & | && || = , ;
        if token.text == '=':
            translate_expr_assign(token, **kwargs)
        elif token.text == '*':
            # matrix multiplication
            if isinstance(ltoken.value, np.ndarray) and isinstance(rtoken.value, np.ndarray):
                token.text = '@'
            token.value = eval(f'ltoken.value {token.text} rtoken.value')
            token.translated = True
        elif token.text == '.':
            token_s = token.lchild
            token_m = token.rchild
            if (token_s.ttype == TT_ID and token_s.text in _locals and
                isinstance(_locals[token_s.text], dict)):
                # s.m --> s['m']
                pass
            elif token_s.ttype == TT_OP and token_s.text == '?[]' and \
                eval(f"isinstance({generate(token_s)[0]}, dict)", _globals, _locals):
                # s['a'].m --> s['a']['m']
                pass 
            else:
                return
            token.text = '?[]'
            token.rchild.ttype = TT_STR
            token.set_children([token_m])
            token.set_rchild()
            token.translated = True
        elif token.text == '()':
            if len(token.children) == 1:
                token.value = token.children[0].value
            else:
                pass
        elif token.text == ':':
            if token.parent.text == '?()':
                # slice
                if token.parent.lchild.text in _locals:
                    # token.parent.lvalue is a variable
                    pass
                if ltoken is not None:
                    translate_subtract_num(ltoken, 1, **kwargs)
                if rtoken is not None and rtoken.text == 'end':
                    token.set_rchild(None)
            else:
                # range
                if token.parent.text == ':':
                    # do nothing
                    pass
                else:
                    translate_to_range(token, **kwargs)
        elif token.text == '[]':
            # matlab >> [a, b] = fun()
            if token == token.parent.lchild:
                token.parent.lchild = token.children[0]
            elif all([type(t.value)==str for t in token.children]):
                if token.parent != None and token.parent.lchild.text == "''.join":
                    pass
                else:
                    token.translated = all([t.translated for t in token.children])
                    # ['str1', 'str2']
                    _token = token.move()
                    token.text = '?()' 
                    token.set_lchild(Token(TT_ID, "''.join", translated=True, root=root))
                    token.set_children([_token])
                token.value = eval(generate(token)[0], _globals, _locals)
                token.translated = all([t.translated for t in token.children])
            else:
                # array and matrix
                t = token.move()
                token.text = '?()'
                token.set_lchild(Token(TT_ID, 'np.array', translated=True, root=root))
                token.set_children([t])
                token.root.add_import_as('numpy', 'np')
                token.value = eval(generate(token)[0], _globals, _locals)
                token.translated = all([t.translated for t in token.children])
        elif token.text == '{}':
            # matlab cell translate to python list
            token.text = '[]'
            text = generate(token)[0]
            token.value = eval(text, _globals, _locals)
        elif token.text == ',':
            if token.parent.text in ['[]', '{}'] and len(token.parent.children) > 1:
                token.text = '[]'
        elif token.text == '?{}':
            # cell slice
            token.text = '?[]'
            for subtoken in token.children:
                if subtoken.ttype == TT_NUM:
                    subtoken.value -= 1
                    subtoken.text = str(subtoken.value)
                elif subtoken.text != ':':
                    t = subtoken.move()
                    subtoken.text = '-'
                    subtoken.ttype = TT_OP
                    subtoken.set_lchild(t)
                    subtoken.set_rchild(Token(TT_NUM, '1', value=1))
            text = generate(token)[0]
            token.value = eval(text, _globals, _locals)
        elif token.text == '?()':
            translate_expr_fcall(token, **kwargs)
        elif token.text in binary_operators:
            if token.text in op_dict:
                token.text = op_dict[token.text]
            token.value = eval(f'ltoken.value {token.text} rtoken.value')
            token.translated = ltoken.translated & rtoken.translated
        elif token.text in left_unary_operators:
            if token.text in op_dict:
                token.text = op_dict[token.text]
            token.value = eval(f'{token.text[0]}rtoken.value')
            token.translated = rtoken.translated
        elif token.text in right_unary_operators:
            if token.text == ".'":
                token_t = Token(TT_ID, 'T', root=root)
                token.text = '.'
                token.value = ltoken.value.T
                token.set_rchild(token_t)
                token.translated = ltoken.translated
            elif token.text == "'":
                token_t = Token(TT_ID, 'T', root=root)
                if issubclass(ltoken.value.dtype.type, np.complex_):
                    token_t.text = 'conj().T'
                token.text = '.'
                token.value = eval(f'ltoken.value.{token_t.text}')
                token.set_rchild(token_t)
                token.translated = ltoken.translated

def translate_for_block(token: Token, **kwargs):
    assert token.ttype == TT_BLOCK and token.text in ['for', 'for..in']
    _globals = token.root._globals
    _locals = token.root._locals
    stat = None
    if token.text == 'for..in':
        translate(token.rchild, **kwargs)
        for i in range(len(token.rchild.value)):
            _locals[token.lchild.text] = token.rchild.value[i]
            for t in token.children:
                stat = translate(t, **kwargs)
                if stat in fctrl_keywords:
                    break
            if stat == None:
                pass
            elif stat == 'break':
                break
            elif stat == 'continue':
                continue
            elif stat == 'return':
                return stat
    else:
        token_iter = token.children[0]
        if token_iter.text == '=':
            token.set_lchild(token_iter.lchild)
            token.set_rchild(token_iter.rchild)
            token.text = 'for..in'
            translate(token.rchild, **kwargs)
            token.children = token.children[1:]
            for i in range(len(token.rchild.value)):
                _locals[token.lchild.text] = token.rchild.value[i]
                for t in token.children:
                    stat = translate(t, **kwargs)
                    if stat in fctrl_keywords:
                        break
                if stat == None:
                    pass
                elif stat == 'break':
                    break
                elif stat == 'continue':
                    continue
                elif stat == 'return':
                    return stat
        else:
            logging.error(f'syntax error at: {token.children[0].row}, {token.children[0].col}')

def translate_while_block(token: Token, **kwargs):
    assert token.ttype == TT_BLOCK and token.text == 'while'
    token_cond = token.children[0]
    stat = None
    while True:
        translate(token_cond, **kwargs)
        if token_cond.value == False:
            break
        for t in token.children[1:]:
            stat = translate(t, **kwargs)
            if stat in fctrl_keywords:
                break
        if stat == None:
            pass
        elif stat == 'break':
            break
        elif stat == 'continue':
            continue
        elif stat == 'return':
            return stat

def translate_if_else_block(token: Token, **kwargs):
    assert token.ttype == TT_BLOCK and token.text == 'if..else'
    for token_ie in token.children:
        if token_ie.text == 'elseif':
            token_ie.text = 'elif'
        if token_ie.text in ['if', 'elif']:
            token_cond =  token_ie.children[0]
            translate(token_cond, **kwargs)
            if token_cond.value == True:
                for t in token_ie.children[1:]:
                    stat = translate(t, **kwargs)
                    if stat in ['break', 'continue', 'return']:
                        return stat
                break
        elif token_ie.text == 'else':
            for t in token_ie.children:
                stat = translate(t, **kwargs)
                if stat in ['break', 'continue', 'return']:
                    return stat

def translate_try_block(token: Token, **kwargs):
    assert token.ttype == TT_BLOCK
    if token.text == 'catch':
        token.text = 'except'

def translate_keyword(token: Token, **kwargs):
    assert token.ttype == TT_KW
    if token.text in ['break', 'continue', 'return']:
        return token.text
    else:
        return None


def translate_command(token: Token, **kwargs):
    root = token.root
    _globals = root._globals
    _locals = root._locals
    if token.text in ['clc', 'clf', 'close', 'clear']:
        logging.info(f'the command "{token.text}" is not required, will be commented out.')
        token.text = f'# {token.text} ' + ' '.join([generate(t)[0] for t in token.children])
        token.ttype = TT_EOL
        token.children = []
    elif token.text == 'global':
        exec(generate(token)[0], _globals, _locals)
    elif token.text == 'load':
        for t in token.children:
            t.ttype = TT_STR
        _token = root.build_func('load', *token.children)
        trans_func_load(_token, **kwargs)
        token.replaced_with(_token)


def translate_comment(token: Token, **kwargs):
    if token.text.startswith('%%'):
        token.text = '# ' + token.text
    else:
        token.text = token.text.replace('%', '#', 1)

def translate(token: Token, **kwargs):
    _globals = token.root._globals
    _locals = token.root._locals
    stat = None
    if token.ttype == TT_BLOCK:
        if token.text in 'if..else':
            stat = translate_if_else_block(token, **kwargs)
        elif token.text == 'while':
            stat = translate_while_block(token, **kwargs)
        elif token.text in ['for', 'for..in']:
            stat = translate_for_block(token, **kwargs)
        elif token.text in ['try', 'catch']:
            stat = translate_try_block(token, **kwargs)
        else:
            raise Exception(f'cannot translate block "{token.text}"')
    else:
        if token.lchild != None and token.lchild.translated == False:
            translate(token.lchild, **kwargs)
        if token.rchild != None and token.rchild.translated == False:
            translate(token.rchild, **kwargs)
        for subtoken in token.children:
            if subtoken.translated == False:
                translate(subtoken, **kwargs)

        if token.translated == False:
            if token.ttype == TT_KW:
                stat = translate_keyword(token, **kwargs)
            elif token.ttype == TT_CMD:
                stat = translate_command(token, **kwargs)
            elif token.ttype == TT_EOL:
                stat = translate_comment(token, **kwargs)
            else:
                stat = translate_expression(token, **kwargs)
        else:
            if token.ttype == TT_KW:
                pass
            elif token.ttype == TT_CMD:
                pass
            elif token.ttype == TT_EOL:
                pass
            else:
                if token.text == '=':
                    exec(generate(token)[0], _globals, _locals)
                else:
                    token.value = eval(generate(token)[0], _globals, _locals)
    return stat


def translate_staments(tokens: 'list[Token]', returns:'list[Token]'=[], **kwargs):
    lines:'list[str]' = []
    last_ttype = None
    for token in tokens:
        # try:
        translate(token, **kwargs)
        if token.translated == False:
            # TODO: ...
            pass
        sublines = generate(token)
        if token.ttype == TT_EOL:
            if len(lines) == 0 or last_ttype == TT_EOL:
                lines.append(f'{sublines[0]}\n')
            elif len(sublines[0]) > 0:
                lines[-1] = f'{lines[-1].rstrip()}  {sublines[0]}\n'
        else:
            for line in sublines:
                logging.info(f'translate line {token.row:3} to: "{line}"')
            lines.extend([line + '\n' for line in sublines])
        last_ttype = token.ttype
        # except:
        #     msg = f'translate line {token.row} error.'
        #     logging.fatal(msg)
        #     raise Exception(msg)
    return lines


def translate_function(token: Token, arg_values:'list[Token]', **kwargs):
    assert token.ttype == TT_BLOCK and token.text == 'def'
    arguments = token.rchild.children
    _locals = {'_locals': token.root._locals}
    for i, av in enumerate(arg_values):
        _locals[arguments[i].text] = av.value
    token.root._locals = _locals
    cur_func = kwargs['cur_func']
    kwargs['cur_func'] = token
    for t in token.children:
        stat = translate(t, **kwargs)
        if stat == 'return':
            break
    else:
        if token.lchild != None:
            token.children.append(Token(TT_KW, 'return'))
    kwargs['cur_func'] = cur_func
    if token.lchild == None:
        token.value = None
    elif token.lchild.ttype == TT_ID:
        token.value = token.lchild.value = _locals[token.lchild.text]
    else:
        returns = token.lchild.children
        for r in returns:
            r.value = _locals[r.text]
        token.value = (r.value for r in returns)

def read_file(filename: str):
    with open(filename, 'rb') as fp:
        encoding = chardet.detect(fp.read())['encoding']
    with open(filename, 'r', encoding=encoding) as fp:
        lines = fp.readlines()
    return lines

def load_file(file_basename:str, 
    input_path:str, 
    output_path:str,
    logging_level=logging.INFO):
    full_basename = os.path.join(input_path, file_basename).replace('\\', '/')
    logging.info(f'load "{full_basename}.m"')
    lines = read_file(f'{full_basename}.m')
    # scan
    logging.info('scan start.')
    root = scan(lines)
    root.text = file_basename
    full_basename = os.path.join(output_path, file_basename).replace('\\', '/')
    if __debug__ and logging_level == logging.DEBUG:
        fn = full_basename + '_scan.json'
        logging.debug(f'write "{fn}"')
        with open(fn, 'w') as fp:
            json.dump(root.todict(), fp, indent=4)
    logging.info('scan done.')
    # parse
    logging.info('parse start.')
    parse(root)

    if __debug__ and logging_level == logging.DEBUG:
        fn = full_basename + '_parse.json'
        logging.debug(f'write "{fn}"')
        with open(fn, 'w') as fp:
            json.dump(root.todict(), fp, indent=4)
    logging.info('parse done.')
    return root

def translate_file(file_basename:str, 
    input_path:str, 
    output_path:str, 
    replaced_functions={},
    logging_level=logging.INFO, 
    **kwargs):
    """
    parameters
    ----------
    """
    root = load_file(file_basename, input_path, output_path, logging_level)

    full_basename = os.path.join(output_path, file_basename).replace('\\', '/')

    # translate
    pyfilename = full_basename + '.py'
    # clear python file
    with open(pyfilename, 'w') as fp:
        pass
    root.exec_('import sys')
    root._locals['sys'].path[0] = output_path
    functions:'dict[str, Token]' = {}
    for token in root.children:
        translate(token, 
            input_path=input_path,
            output_path=output_path,
            replaced_functions=replaced_functions,
            functions=functions,
            cur_func=root,
            logging_level=logging_level, **kwargs)

    if __debug__ and logging_level == logging.DEBUG:
        fn = full_basename + '_trans.json'
        logging.debug(f'write "{fn}"')
        with open(fn, 'w') as fp:
            json.dump(root.todict(), fp, indent=4)
    logging.info(f'translate "{file_basename}" done.')

    logging.info(f'generate "{pyfilename}".')
    lines = generate(root)
    
    with open(pyfilename, 'w') as fp:
        fp.writelines([text + '\n' for text in lines])

    for n in functions:
        root = functions[n].root
        pyfilename = output_path + '/' + root.text + '.py'
        logging.info(f'generate "{pyfilename}".')
        lines = generate(root)
        with open(pyfilename, 'w') as fp:
            fp.writelines([text + '\n' for text in lines])


             