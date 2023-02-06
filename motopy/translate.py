"""
translate tokens.

This is a part of motopy.
"""
import chardet
import copy
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
from .port import __all__ as port_funcs
from .generate import generate
from .utils import ExecError, TransError


def show_code(token: Token, fname:str=''):
    if token.ttype == TT_EOL:
        return
    lines = generate(token)
    for line in lines:
        logging.info(f'({token.root.text}:{fname}:{token.row}) --> {line}')

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
            root.add_import('numpy', as_='np')
            fname = 'np.int32'
    elif not isinstance(token.value, int):
        fname = 'int'
    else:
        return
    _token = root.build_func(fname, token.move())
    _token.value = root.eval_expr(_token)
    token.replaced_with(_token)
    logging.info(f'translate the value of "{text}" in line {token.row:3} to int')

def translate_to_range(token: Token, **kwargs):
    assert token.ttype == TT_OP and token.text in [':', '::']
    text = generate(token)[0]
    root = token.root
    ltoken = token.lchild
    rtoken = token.rchild
    if token.text == '::':
        # a:step:b --> np.arange(a, b, step)
        token.set_children([ltoken, rtoken, token.children[0]])
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
        token.root.add_import('numpy', as_='np')
        token_fname = Token(TT_ID, 'np.arange', root=root)
    token.text = '?()'
    token.set_lchild(token_fname)
    token.set_rchild()
    token.value = root.eval_expr(token)
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
    token.value = root.eval_expr(token)

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
            token.root.add_import('time', from_='time')
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
            token.value = token.root.eval_expr(token)
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
    token.root.add_import('wavfile', from_='scipy.io')
    token.lchild.text = 'wavfile.read'
    token_assign = token.parent
    if token.is_rchild() and token_assign.text == '=' and \
        token_assign.lchild.text == ',':
            # data, fs --> fs, data
            token_assign.lchild.set_children(token_assign.lchild.children[::-1])
    token.value = token.root.eval_expr(token)
    token.translated = True

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
    root.add_import('scandir', from_='os')
    token_list = root.build_list(root.build_for_in('e', root.build_func('scandir', token.children[0]), ts=root.build_dict(
        name=Token(TT_ID, 'e.name'), 
        folder=Token(TT_ID, 'e.path'), 
        isdir=Token(TT_ID, 'e.is_dir()'))))
    token.text = '[]'
    token.set_lchild()
    token.set_children(token_list.children)
    token.value = root.eval_expr(token)
    token.translated = True
    
def trans_func_find(token: Token, **kwargs):
    root = token.root
    root.add_import('numpy', as_='np')
    cond = token.children[0].value
    if isinstance(cond, np.ndarray) and cond.ndim == 1:
        _token = Token(TT_OP, '?[]', lchild= root.build_func('np.nonzero', token.children[0]), 
            children=[root.build_token(0)], root=root)
        token.replaced_with(_token)
        token = _token
    else:
        token.lchild.text = 'np.nonzero'
    token.value = root.eval_expr(token)
    token.translated = True

def trans_func_isfield(token: Token, **kwargs):
    _token = token.move()
    s = _token.children[0]
    field = _token.children[1]
    token.text = 'in'
    token.set_lchild(field)
    token.set_rchild(s)
    token.value = token.root.eval_expr(token)
    token.translated = True

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
    token.root.add_import('numpy', as_='np')
    if len(token.children) == 1:
        # max(A)
        token.lchild.text = 'np.amax'
    elif len(token.children) == 2:
        # max(A, B)
        token.lchild.text = 'np.maximum'
    else:
        raise Exception('TODO:...')
    token.value = token.root.eval_expr(token)
    token.translated = True

def trans_func_min(token: Token, **kwargs):
    token.root.add_import('numpy', as_='np')
    if len(token.children) == 1:
        # min(A)
        token.lchild.text = 'np.amin'
    elif len(token.children) == 2:
        # min(A, B)
        token.lchild.text = 'np.minimum'
    else:
        raise Exception('TODO:...')
    token.value = token.root.eval_expr(token)
    token.translated = True

def trans_func_length(token: Token, **kwargs):
    root = token.root
    # length(a) --> max(np.shape(a))
    root.add_import('numpy', as_='np')
    token.lchild.text = 'max'
    token.set_children([root.build_func('np.shape', token.children[0])])
    token.value = token.root.eval_expr(token)
    token.translated = True

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
            root.add_import('loadmat', from_='scipy.io')
            token.lchild.text = 'loadmat'
            token.value = token.root.eval_expr(token)
            token.translated = True
        else:
            # If filename has an extension other than .mat, 
            # the load function treats the file as ASCII data.
            root.add_import('numpy', as_='np')
            token.lchild.text = 'np.loadtxt'
            token.children.append(Token(TT_OP, '=', 
                lchild=Token(TT_ID, 'ndmin'),
                rchild=Token(TT_NUM, '2')))
            token.value = root.eval_expr(token)
            token.translated = True
    else:
        # no returns
        if filename.endswith('.mat'):
            root.add_import('loadmat', from_='scipy.io')
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
            token.root.add_import('numpy', as_='np')
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
        token.value = token.root.eval_expr(token)
        token.translated = True

def trans_func_strcat(token: Token, **kwargs):
    # strcat(s1, s2) --> ''.join([s1, s2])
    root = token.root
    func = root.build_func("''.join", root.build_list(*token.children))
    token.lchild.text = "''.join"
    token.set_children(func.children)
    token.value = token.root.eval_expr(token)
    token.translated = True

def trans_func_zeros(token: Token, **kwargs):
    root = token.root
    token.lchild.text = 'np.' + token.lchild.text
    root.add_import('numpy', as_='np')
    for t in token.children:
        if not isinstance(t.value, int):
            translate_to_int(t, **kwargs)
    if len(token.children) == 1:
        token.set_children([Token(TT_OP, '()', children=[token.children[0], token.children[0]], root=root)])
    elif len(token.children) > 1:
        token.children = [Token(TT_OP, '()', children=token.children, root=root)]
    token.value = root.eval_expr(token)
    token.translated = True
    
func_dict = {
    'audioread': trans_func_audioread,
    'cell': trans_func_cell,
    'dir': trans_func_dir,
    'find': trans_func_find,
    'sprintf': trans_func_sprintf,
    'fprintf': trans_func_fprintf,
    'isfield': trans_func_isfield,
    'max': trans_func_max,
    'min': trans_func_min,
    # 'length': trans_func_length,
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
        value = _locals[ltoken.text]
        if eval(f'callable({ltoken.text})', _globals, _locals):
            # this is a function call
            token.value = token.root.eval_expr(token)
            token.translated = True
        elif isinstance(value, (list, np.ndarray)):
            # this is a array or cell slice
            token.text = '?[]'
            for subtoken in token.children:
                if subtoken.text not in [':', '::']:
                    if isinstance(subtoken.value, np.ndarray):
                        translate_subtract_num(subtoken, 1, **kwargs)
                        translate_to_int(subtoken, **kwargs)   
                    else:
                        _subtoken = copy.copy(subtoken)
                        translate_subtract_num(subtoken, 1, **kwargs)
                        translate_to_int(subtoken, **kwargs)
                        translate_to_int(_subtoken, **kwargs)
                        subtoken.set_lchild(subtoken.move())
                        subtoken.set_rchild(_subtoken)
                        subtoken.text = ':'
                        subtoken.ttype = TT_OP
            if len(token.children) == 1 and np.ndim(value) == 2:
                if value.shape[0] == 1:
                    token.set_children([Token(TT_OP, ':', root=root), token.children[0]])
                elif value.shape[1] == 1:
                    token.set_children([token.children[0], Token(TT_OP, ':', root=root)])
                else:
                    # TODO: ...
                    pass
            else:
                # TODO: ...
                pass

            token.value = token.root.eval_expr(token)
            token.translated = True
        else:
            raise TransError(token.text, root.text, token.row, token.col, 
                'object "{ltoken.text}" not allowed called or slice.')
    elif ltoken.translated == True:
        token.value = root.eval_expr(token)
        token.translated = True
    elif ltoken.text in kwargs['replaced_functions']:
        m, fn = kwargs['replaced_functions'][ltoken.text]
        exec(f'from {m} import {fn}', _globals, _locals)
        root.add_import(fn, from_=m)
        token.value = root.eval_expr(token)
        token.translated = True

    if token.translated == True:
        return

    if token.is_rchild() and token.parent.text == '=':
        if token.parent.lchild.text == ',':
            nargout = len(token.parent.lchild.children)
        else:
            nargout = 1
    elif token.parent.ttype in [TT_BLOCK, TT_ROOT]:
        nargout = 0
    else:
        nargout = 1

    func = cur_func
    while func != None:
        if  ltoken.text in func.functions:
            translate_function(func.functions[ltoken.text], token.children, nargout=nargout, **kwargs)
            token.value = func.functions[ltoken.text].value
            return
        else:
            func = func.parent

    if ltoken.text in functions and functions[ltoken.text].translated == False:
        # this is a local function.
        translate_function(functions[ltoken.text], token.children, nargout=nargout, **kwargs)
        token.value = functions[ltoken.text].value
    elif ltoken.text in func_dict:
        func_dict[ltoken.text](token, **kwargs)
    elif ltoken.text in func_name_dict:
        ltoken.text = func_name_dict[ltoken.text]
        if ltoken.text.startswith('np.'):
            root.add_import('numpy', as_='np')
        elif ltoken.text.startswith('linalg.'):
            root.add_import('linalg', from_='scipy')
        elif ltoken.text.startswith('signal.'):
            root.add_import('signal', from_='scipy')
        elif ltoken.text.startswith('integrate.'):
            root.add_import('integrate', from_='scipy')
        elif ltoken.text.startswith('random.'):
            root.add_import('random', from_='numpy')
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
        translate_function(functions[ltoken.text], token.children, nargout=nargout, **kwargs)
        token.value = functions[ltoken.text].value
        root.add_import(ltoken.text, from_=ltoken.text, noexe=True)
    else:
        raise TransError(token.text, root.text, token.row, token.col, 
                f'no translate rule for {ltoken.text}()')

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
                root.add_import('numpy', as_='np')
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
            try:
                text = f'ltoken.value {token.text} rtoken.value'
                token.value = eval(text)
            except:
                raise ExecError(text, root.text, token.row)
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
                if ltoken is not None:
                    translate_subtract_num(ltoken, 1, **kwargs)
                    translate_to_int(ltoken)
                if rtoken is not None:
                    if rtoken.text == 'end':
                        token.set_rchild(None)
                    else:
                        translate_to_int(rtoken)
                token.translated = True
            else:
                # range
                translate_to_range(token, **kwargs)
        elif token.text == '::':
            if token.parent.text == '?()':
                # slice
                if ltoken is not None:
                    translate_subtract_num(ltoken, 1, **kwargs)
                    translate_to_int(ltoken)
                if rtoken is not None:
                    if rtoken.text == 'end':
                        token.set_rchild(None)
                    else:
                        translate_to_int(rtoken)
                b = token.rchild
                s = token.children[0]
                translate_to_int(s)
                # a:s:b --> a:b:s
                token.set_rchild(s)
                token.set_children([b])
                token.translated = True
            else:
                # range
                translate_to_range(token, **kwargs)
        elif token.text == '[]':
            # [a, b] = fun() --> a, b = fun()
            if token.is_lchild() and token.parent.text == '=':
                token.text = ','
                token.translated = True
            elif len(token.children) == 0:
                # >> []
                token.value = []
                token.translated = True
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
                token.value = token.root.eval_expr(token)
                token.translated = all([t.translated for t in token.children])
            else:
                values = []
                for t in token.children:
                    if t.text == '[]':
                        values.extend([tt.value for tt in t.children])
                    else:
                        values.append(t.value)
                if any([isinstance(v, np.ndarray) for v in values]):
                    if len(values) == 1:
                        # [A] --> A
                        token.replaced_with(token.children[0])
                        return
                    else:
                        fname = 'np.block'
                else:
                    fname = 'np.array'
                # array and matrix
                _token = token.move()
                token.text = '?()'
                token.set_lchild(Token(TT_ID, fname, translated=True, root=root))
                token.set_children([_token])
                token.root.add_import('numpy', as_='np')
                token.value = root.eval_expr(token)
                # token.value = token.root.eval_expr(token)
                _token.translated = all([t.translated for t in _token.children])
                token.translated = _token.translated
        elif token.text == '{}':
            # matlab cell translate to python list
            token.text = '[]'
            text = generate(token)[0]
            try: 
                token.value = eval(text, _globals, _locals)
            except:
                raise ExecError(text, root.text, token.row)
        elif token.text == ',':
            if token.parent.text in ['[]', '{}'] and len(token.parent.children) > 1:
                token.text = '[]'
                token.value = root.eval_expr(token)
                token.translated = True
        elif token.text == '?{}':
            # cell slice
            token.text = '?[]'
            for subtoken in token.children:
                if subtoken.ttype == TT_NUM:
                    subtoken.value -= 1
                    subtoken.text = str(subtoken.value)
                elif subtoken.text != ':':
                    _token = subtoken.move()
                    subtoken.text = '-'
                    subtoken.ttype = TT_OP
                    subtoken.set_lchild(_token)
                    subtoken.set_rchild(Token(TT_NUM, '1', value=1))
            text = generate(token)[0]
            try: 
                token.value = eval(text, _globals, _locals)
            except:
                raise ExecError(text, root.text, token.row)
        elif token.text == '?()':
            translate_expr_fcall(token, **kwargs)
        elif token.text in binary_operators:
            if token.text in op_dict:
                token.text = op_dict[token.text]
            try:
                text = f'ltoken.value {token.text} rtoken.value'
                token.value = eval(text)
            except:
                raise ExecError(text, root.text, token.row)
            token.translated = ltoken.translated & rtoken.translated
        elif token.text in left_unary_operators:
            if token.text in op_dict:
                token.text = op_dict[token.text]
            op = token.text.rstrip('?')
            try:
                text = f'{op}rtoken.value'
                token.value = eval(text)
            except:
                raise ExecError(text, root.text, token.row)
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
                try:
                    text = f'ltoken.value.{token_t.text}'
                    token.value = eval(text)
                except:
                    raise ExecError(text, root.text, token.row)
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
        logging.warning(f'The catch block cannot reaches.')
    elif token.text == 'try':
        for t in token.children:
            stat = translate(t, **kwargs)
            if stat in ['break', 'continue', 'return']:
                return stat

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
            elif token.ttype == TT_CODE:
                exec(generate(token)[0], _globals, _locals)
            else:
                if token.text == '=':
                    exec(generate(token)[0], _globals, _locals)
                else:
                    token.value = token.root.eval_expr(token)
    return stat


def translate_function(token: Token, arg_values:'list[Token]', nargout:int=0, **kwargs):
    assert token.ttype == TT_BLOCK and token.text == 'def'
    arguments = token.rchild.children
    _locals = {'_locals': token.root._locals}
    for i, av in enumerate(arg_values):
        _locals[arguments[i].text] = av.value
    _locals['nargin'] = len(arg_values)
    _locals['nargout'] = nargout
    token.root._locals = _locals
    cur_func = kwargs['cur_func']
    kwargs['cur_func'] = token
    fname = token.rchild.lchild.text
    for t in token.children:
        stat = translate(t, **kwargs)
        show_code(t, fname)
        if stat == 'return':
            break
    else:
        if token.lchild != None:
            token.children.append(Token(TT_KW, 'return'))
    kwargs['cur_func'] = cur_func
    token.root._locals = _locals['_locals']
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
    for fn in port_funcs:
        replaced_functions[fn] = ('motopy', fn)
    functions:'dict[str, Token]' = {}
    skiped = False
    for token in root.children:
        if token.ttype == TT_CODE:
            root.exec_(token.text)
            skiped = True
        elif skiped == True:
            skiped = False
        else:
            translate(token, 
                input_path=input_path,
                output_path=output_path,
                replaced_functions=replaced_functions,
                functions=functions,
                cur_func=root,
                logging_level=logging_level, **kwargs)
            show_code(token)

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


             