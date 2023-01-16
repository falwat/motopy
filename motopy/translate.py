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
from .constants import *
from .token import Token
from .scan import scan
from .parse import parse
from .generate import generate, generate_imports, generate_import_as


def add_import_as(module_name: str, alias_name: str, **kwargs):
    """ Add module_name to kwargs['import_as'].
    if module_name is first added, will exec "import <module_name> as <alias_name>".
    """
    import_as:'dict[str, str]' = kwargs['import_as']
    if module_name not in import_as:
        exec(f'import {module_name} as {alias_name}', kwargs['_globals'], kwargs['_locals'])
        exec(f'import {module_name} as {alias_name}', kwargs['_globals'], kwargs['_globals'])
    import_as[module_name] = alias_name

def add_imports(module_name: str, object_name: str, **kwargs):
    """  Add module_name.object_name to kwargs['imports']
    if object_name is first added, will exec "from <module_name> import <object_name>".
    """
    imports: 'dict[str, set[str]]' = kwargs['imports']
    if module_name not in imports:
        exec(f'from {module_name} import {object_name}', kwargs['_globals'], kwargs['_locals'])
        exec(f'from {module_name} import {object_name}', kwargs['_globals'], kwargs['_globals'])
        imports[module_name] = set([object_name])
    elif object_name not in imports[module_name]:
        exec(f'from {module_name} import {object_name}', kwargs['_globals'], kwargs['_locals'])
        exec(f'from {module_name} import {object_name}', kwargs['_globals'], kwargs['_globals'])
        imports[module_name].add(object_name)

def translate_add_num(token: Token, num: int, **kwargs):
    text = generate(token)[0]
    if token.ttype == TT_NUM:
        token.value += num
        token.text = str(token.value)
    else:
        token.set_lchild(token.copy())
        token.set_rchild(Token(TT_NUM, str(num), value=num))
        token.text = '+'
        token.ttype = TT_OP
        token.value = token.lchild.value + num
    token.translated = True
    logging.info(f'translate "{text}" in line {token.row:3} to {generate(token)[0]}')

def translate_subtract_num(token: Token, num: int, **kwargs):
    text = generate(token)[0]
    if token.ttype == TT_NUM:
        token.value -= num
        token.text = str(token.value)
    else:
        t = token.copy()
        token.ttype = TT_OP
        token.text = '-'
        token.set_lchild(t)
        token.set_rchild(Token(TT_NUM, str(num), value=num))
        token.value = eval(generate(token)[0], kwargs['_globals'], kwargs['_locals'])
    token.translated = True
    logging.info(f'translate "{text}" in line {token.row:3} to {generate(token)[0]}')

def translate_to_int(token: Token, **kwargs):
    text = generate(token)[0]
    token_copy = token.copy()
    token_fname = token.copy()
    if isinstance(token.value, np.ndarray):
        if issubclass(token.value.dtype.type, np.integer):
            return
        else:
            add_import_as('numpy', 'np', **kwargs)
            token_fname.text = 'np.int32'
    elif not isinstance(token.value, int):
        token_fname.text = 'int'
    else:
        return
    token_fname.ttype = TT_ID
    token.set_lchild()
    token.set_rchild()
    token.set_children()
    token.text = '?()'
    token.ttype = TT_OP
    token.set_lchild(token_fname)
    token.set_rchild()
    token.set_children([token_copy])
    token.value = eval(generate(token)[0], kwargs['_globals'], kwargs['_locals'])
    token.translated = True
    logging.info(f'translate the value of "{text}" in line {token.row:3} to int')

def translate_to_range(token: Token, **kwargs):
    assert token.ttype == TT_OP and token.text == ':'
    text = generate(token)[0]
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
    # if token.parent != None and token.parent.text == '=' and \
    #     token.parent.parent != None and token.parent.parent.text == 'for' and \
    #     all([type(t)==int for t in token.values]):
    #     token_fname = Token(TT_ID, 'range')
    # else:
    add_import_as('numpy', 'np', **kwargs)
    token_fname = Token(TT_ID, 'np.arange')
    token.text = '?()'
    token.set_lchild(token_fname)
    token.set_rchild()
    token.value = eval(generate(token)[0], kwargs['_globals'], kwargs['_locals'])
    logging.warning(f'translate "{text}" to "{generate(token)[0]}". ' +
            'Please manually confirm the result.')

def translate_assign_copy(token: Token, **kwargs):
    assert token.ttype == TT_ID and isinstance(token.value, np.ndarray)
    # x --> x.copy()
    token_copy = token.copy()
    token_dot = token.copy()
    token_dot.ttype = TT_OP
    token_dot.text = '.'
    token_dot.set_lchild(token_copy)
    token_dot.set_rchild(Token(TT_ID, 'copy'))
    token.ttype = TT_OP
    token.text = '?()'
    token.set_lchild(token_dot)
    text = generate(token)[0]
    token.value = eval(text, kwargs['_globals'], kwargs['_locals'])
    logging.info(f'copy the variable "{token_copy.text}" in line {token.row}')

def translate_format_text(fmt_text, tokens: 'list[Token]'):
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

def translate_func_sprintf(token: Token, **kwargs):
    text = token.children[0].text
    token.text = translate_format_text(text, token.children[1:])
    token.ttype = TT_FMT_STR
    token.value = token.text
    token.children = []

def translate_func_fprintf(token: Token, **kwargs):
    fmt_token = token.children[0] 
    if isinstance(token.children[0].value, str):
        fmt_token.text = translate_format_text(fmt_token.text, token.children[1:])
        fmt_token.ttype = TT_FMT_STR
        fmt_token.value = fmt_token.text
        token.children = [fmt_token]
        token.lchild.text = 'print'

def translate_func_max(token: Token, **kwargs):
    token.lchild.text = 'np.amax'
    token.value = eval(generate(token)[0], kwargs['_globals'], kwargs['_locals'])

def translate_func_load(token: Token, **kwargs):
    _globals = kwargs['_globals']
    _locals = kwargs['_locals']
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
            add_imports('scipy.io', 'loadmat', **kwargs)
            token.lchild.text = 'loadmat'
            token.value = eval(generate(token)[0], _globals, _locals)
        else:
            # If filename has an extension other than .mat, 
            # the load function treats the file as ASCII data.
            add_import_as('numpy', 'np', **kwargs)
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
            add_imports('scipy.io', 'loadmat', **kwargs)
            token_copy = token.copy()
            token_copy.lchild.text = 'loadmat'
            token.text = '='
            token.set_lchild(Token(TT_ID, '_mat'))
            token.set_rchild(token_copy)
            token.set_children()
            exec(generate(token)[0], _globals, _locals)
            _mat:dict = _locals['_mat']    
            token_copy = token.copy()
            if len(arg_tokens) > 1:
                varnames = [t.text for t in token.children[1:]]
            else:
                varnames = []
                for n in _mat:
                    if n not in ['__header__', '__version__', '__globals__']:
                        varnames.append(n)
            children = [token_copy]
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
            token_copy = token.copy()
            vname = os.path.splitext(os.path.split(filename)[-1])[0]
            add_import_as('numpy', 'np', **kwargs)
            token_copy.lchild.text = 'np.loadtxt'
            token_copy.children.append(Token(TT_OP, '=', 
                lchild=Token(TT_ID, 'ndmin'),
                rchild=Token(TT_NUM, '2')))
            token.text = '='
            token.set_lchild(Token(TT_ID, vname))
            token.set_rchild(token_copy)
            token.set_children()
            exec(generate(token)[0], _globals, _locals)


def translate_func_size(token: Token, **kwargs):
    if len(token.children) == 2:
        token_func = token.lchild
        token_func.text = 'shape'
        token_var = token.children[0]
        token_dim = token.children[1]
        token_dot = token.copy()
        translate_subtract_num(token_dim, 1, **kwargs)
        token_dot.set_lchild(token_var)
        token_dot.set_rchild(token_func)
        token_dot.text = '.'
        token.set_lchild(token_dot)
        token.set_children([token_dim])
        token.text = '?[]'
        token.value = eval(generate(token)[0], kwargs['_globals'], kwargs['_locals'])

def translate_func_zeros(token: Token, **kwargs):
    token.lchild.text = 'np.' + token.lchild.text
    add_import_as('numpy', 'np', **kwargs)
    for t in token.children:
        if not isinstance(t.value, int):
            translate_to_int(t, **kwargs)
    if len(token.children) == 1:
        token.set_children([Token(TT_OP, '()', children=[token.children[0], token.children[0]])])
    elif len(token.children) > 1:
        token.children = [Token(TT_OP, '()', children=token.children)]

func_dict = {
    'sprintf': translate_func_sprintf,
    'fprintf': translate_func_fprintf,
    'max': translate_func_max,
    'load': translate_func_load,
    'ones': translate_func_zeros,
    'size': translate_func_size,
    'zeros': translate_func_zeros,
}

def translate_expression(token: Token, **kwargs):
    assert token.ttype in [TT_OP, TT_ID, TT_NUM, TT_STR] or token.text == 'end'
    _globals = kwargs['_globals']
    _locals = kwargs['_locals']
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
    elif token.ttype == TT_OP:
        # () [] {} . +? -? ~ ' .' ^ .^ * / \ .* ./ .\ : < <= > >= == ~=
        # & | && || = , ;
        if token.text == '=':
            if token.parent != None and token.parent.ttype == TT_BLOCK and \
                token.parent.text == 'for' and token == token.parent.children[0]:
                # >> for k = 1:10
                logging.info(f'Set "{ltoken.text}" to {rtoken.children[0].value}')
                text = generate(rtoken)[0]
                exec(f'{ltoken.text} = {text}[0]', _globals, _locals)
                ltoken.value = rtoken.value[0]
                token.translated = True
            elif rtoken.ttype == TT_ID and ltoken.ttype == TT_ID and \
                isinstance(rtoken.value, np.ndarray):
                # >> y=x translate to: y = x.copy()
                translate_assign_copy(rtoken, **kwargs)
                text = generate(token)[0]
                exec(text, _globals, _locals)
                token.value = rtoken.value
                token.translated = True
            else:
                text = generate(token)[0]
                exec(text, _globals, _locals)
                token.value = rtoken.value
                token.translated = True
        elif token.text == '*':
            # matrix multiplication
            if isinstance(ltoken.value, np.ndarray) and isinstance(rtoken.value, np.ndarray):
                token.text = '@'
            token.value = eval(f'ltoken.value {token.text} rtoken.value')
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
            else:
                # array and matrix
                t = token.copy()
                token.text = '?()'
                token.set_lchild(Token(TT_ID, 'np.array'))
                token.set_children([t])
                add_import_as('numpy', 'np', **kwargs)
                token.value = eval(generate(token)[0])
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
                    t = subtoken.copy()
                    subtoken.text = '-'
                    subtoken.ttype = TT_OP
                    subtoken.set_lchild(t)
                    subtoken.set_rchild(Token(TT_NUM, '1', value=1))
            text = generate(token)[0]
            token.value = eval(text, _globals, _locals)
        elif token.text == '?()':
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
                    msg = f'the type of variable "{ltoken}" not allowed called or slice.'
                    logging.fatal(msg)
                    raise Exception(msg)
            elif ltoken.text in kwargs['replaced_functions']:
                m, fn = kwargs['replaced_functions'][ltoken.text]
                exec(f'from {m} import {fn}', _globals, _locals)
                add_imports(m, fn, **kwargs)
                token.value = eval(f'{generate(token)[0]}', _globals, _locals)
            elif ltoken.text in functions and functions[ltoken.text].translated == False:
                # this is a local function.
                __locals = {}
                arguments_value = [t.value for t in token.children]
                func_name, arguments, _, _ = split_func_block(functions[ltoken.text])
                for i, av in enumerate(arguments_value):
                    __locals[arguments[i].text] = av
                lines = translate_function(functions[ltoken.text], 
                    _globals=_globals,
                    _locals=__locals,
                    file_basename=kwargs['file_basename'],
                    input_path=kwargs['input_path'],
                    output_path=kwargs['output_path'],
                    replaced_functions=kwargs['replaced_functions'],
                    imports=kwargs['imports'],
                    import_as=kwargs['import_as'],
                    functions=functions, 
                    indent=kwargs['indent'],
                    logging_level=kwargs['logging_level'])
                text = ''.join(lines)
                pyfilename = os.path.join(kwargs['output_path'], f"{kwargs['file_basename']}.py").replace('\\', '/')
                with open(pyfilename, 'a') as fp:
                    fp.write(text)
                exec(text, _globals, _locals)
                text = generate(token)[0]
                token.value = eval(text, _globals, _locals)
            elif f'{ltoken.text}.m' in  os.listdir(kwargs['input_path']):
                filename = f'{ltoken.text}.m'
                logging.info(f'found "{filename}" in "{kwargs["input_path"]}"')
                arguments_value = [t.value for t in token.children]
                translate_file(file_basename=ltoken.text,
                    input_path=kwargs['input_path'],
                    output_path=kwargs['output_path'],
                    arguments_value=arguments_value,
                    replaced_functions=kwargs['replaced_functions'],
                    indent=kwargs['indent'],
                    logging_level=kwargs['logging_level'])
                kwargs['replaced_functions'][ltoken.text] = (ltoken.text, ltoken.text)
                exec(f'from {ltoken.text} import {ltoken.text}', _globals, _locals)
                add_imports(ltoken.text, ltoken.text, **kwargs)
                token.value = eval(f'{generate(token)[0]}', _globals, _locals)
            elif ltoken.text == 'rand':
                add_imports('numpy.random', 'rand', **kwargs)
            elif ltoken.text in func_dict:
                func_dict[ltoken.text](token, **kwargs)
            elif ltoken.text in func_name_dict:
                ltoken.text = func_name_dict[ltoken.text]
                if ltoken.text.startswith('np.'):
                    add_import_as('numpy', 'np', **kwargs)
                elif ltoken.text.startswith('linalg.'):
                    add_imports('scipy', 'linalg', **kwargs)
                elif ltoken.text.startswith('signal.'):
                    add_imports('scipy', 'signal', **kwargs)
                elif ltoken.text.startswith('integrate.'):
                    add_imports('scipy', 'integrate', **kwargs)
                token.value = eval(generate(token)[0], _globals, _locals)
            else:
                logging.info(f'no translate rule for {ltoken.text}()')
        elif token.text in binary_operators:
            if token.text in op_dict:
                token.text = op_dict[token.text]
            token.value = eval(f'ltoken.value {token.text} rtoken.value')
        elif token.text in left_unary_operators:
            if token.text in op_dict:
                token.text = op_dict[token.text]
            token.value = eval(f'{token.text[0]}rtoken.value')
        elif token.text in right_unary_operators:
            if token.text == ".'":
                token_t = token.copy()
                token_t.text = 'T'
                token_t.ttype = TT_ID
                token_t.set_lchild()
                token.text = '.'
                token_t.value = ltoken.value.T
                token.set_rchild(token_t)
            elif token.text == "'":
                token_t = token.copy()
                token_t.text = 'conj().T'
                token_t.ttype = TT_ID
                token_t.set_lchild()
                token.text = '.'
                token_t.value = ltoken.value.conj().T
                token.set_rchild(token_t)

def translate_for_block(token: Token, **kwargs):
    assert token.ttype == TT_BLOCK and token.text == 'for'
    if token.children[0].text == '=':
        token.children[0].text = 'in'
    else:
        logging.error(f'syntax error at: {token.children[0].row}, {token.children[0].col}')

def translate_if_block(token: Token, **kwargs):
    assert token.ttype == TT_BLOCK
    if token.text == 'elseif':
        token.text = 'elif'

def translate_command(token: Token, **kwargs):
    _globals = kwargs['_globals']
    _locals = kwargs['_locals']
    if token.text in ['clc', 'clf', 'close', 'clear']:
        logging.info(f'the command "{token.text}" is not required, will be commented out.')
        token.text = f'# {token.text} ' + ' '.join([generate(t)[0] for t in token.children])
        token.ttype = TT_EOL
        token.children = []
    elif token.text == 'global':
        exec(generate(token)[0], _globals, _locals)
    elif token.text == 'load':
        token_fname = token.copy()
        token_fname.ttype = TT_ID
        token_fname.set_children()
        token.ttype = TT_OP
        token.text = '?()'
        token.set_lchild(token_fname)
        for t in token.children:
            t.ttype = TT_STR
        translate_func_load(token, **kwargs)


def translate_comment(token: Token, **kwargs):
    if token.text.startswith('%%'):
        token.text = '# ' + token.text
    else:
        token.text = token.text.replace('%', '#', 1)

def translate(token: Token, **kwargs):
    _globals = kwargs['_globals']
    _locals = kwargs['_locals']
    if token.rchild is not None:
        translate(token.rchild, **kwargs)
    for subtoken in token.children:
        translate(subtoken, **kwargs)
    if token.lchild is not None:
        translate(token.lchild, **kwargs)

    # try:
    if token.ttype == TT_NUM:
        token.value = eval(token.text, _globals, _locals)
    elif token.ttype == TT_STR:
        token.value = token.text
    elif token.ttype == TT_ID:
        if token.parent != None:
            if token == token.parent.lchild:
                if token.parent.text not in ['=', '?()']:
                    token.value = eval(f'{token.text}', _globals, _locals)
            elif token.text in value_dict:
                token.ttype = TT_NUM
                token.value = value_dict[token.text]
                token.text = str(token.value)
            elif token.parent.text != 'global':
                token.value = eval(f'{token.text}', _globals, _locals)
        else: 
            token.value = eval(f'{token.text}', _globals, _locals)
        # elif token.text in _locals:
        #     token.value = _locals[token.text]
        # elif token.parent.text == 'global' and token.text in _globals:
        #     token.value = _globals[token.text]
    elif token.ttype == TT_OP:
        translate_expression(token, **kwargs)
    elif token.ttype == TT_BLOCK and token.text in ['if', 'elseif', 'else']:
        translate_if_block(token, **kwargs)
    elif token.ttype == TT_BLOCK and token.text == 'for':
        translate_for_block(token, **kwargs)
    elif token.ttype == TT_CMD:
        translate_command(token, **kwargs)
    elif token.ttype == TT_EOL:
        translate_comment(token, **kwargs)
    # except Exception as e:
    #     msg = f'translate token "{token.text}" at (row: {token.row}, col: {token.col}) error.'
    #     logging.fatal(msg)
    #     raise Exception(msg)

def translate_staments(tokens: 'list[Token]', returns:'list[Token]'=[], **kwargs):
    lines:'list[str]' = []
    last_ttype = None
    for token in tokens:
        # try:
        translate(token, **kwargs)
        sublines = generate(token)
        if token.ttype == TT_EOL:
            if len(lines) == 0 or last_ttype == TT_EOL:
                lines.append(f'{sublines[0]}\n')
            elif len(sublines[0]) > 0:
                lines[-1] = f'{lines[-1].rstrip()}  {sublines[0]}\n'
        else:
            for line in sublines:
                logging.debug(f'translate line {token.row:3} to: "{line}"')
            lines.extend([line + '\n' for line in sublines])
        last_ttype = token.ttype
        # except:
        #     msg = f'translate line {token.row} error.'
        #     logging.fatal(msg)
        #     raise Exception(msg)
    return lines

def split_func_block(token: Token):
    """ Split `token` of function to `func_name`, `arguments`, `returns`, `statments`.
    """
    assert token.ttype == TT_BLOCK and token.text == 'function'
    token_header = token.children[0]
    rtoken = token_header.rchild
    ltoken = token_header.lchild
    if token_header.text == '=':
        # with return value
        assert rtoken.ttype == TT_OP and rtoken.text == '?()'
        func_name = rtoken.lchild.text
        arguments = rtoken.children
        if ltoken.ttype == TT_ID:
            # only one return value
            returns = [ltoken]
        elif ltoken.ttype == TT_OP and ltoken.text == '[]':
            # multi return values
            returns = ltoken.children
        else:
            raise Exception(f'syntax error at line {token.row}')
    elif token_header.text == '?()':
        # without return value
        func_name = ltoken.text
        arguments = token_header.children
        returns = []
    else:
        raise Exception(f'syntax error at line {token.row}')
    statments = token.children[1:]
    return func_name, arguments, returns, statments

def translate_function(token_func: Token, **kwargs):
    lines = []
    func_name, arguments, returns, statments = split_func_block(token_func)
    header_text = f'{func_name}({", ".join([t.text for t in arguments])})'
    logging.info(f'translate function: "{header_text}" start.')
    # try:
    sublines = translate_staments(statments, returns=returns, **kwargs)
    # except:
    #     logging.fatal(f'translate function "{header_text}" error.')
    lines.extend([' '*kwargs['indent'] + text for text in sublines])

    if len(returns) > 0:
        text = f'return {", ".join([t.text for t in returns])}\n'
        lines.append(' '*kwargs['indent'] + text)
    
    lines.insert(0, f'def {header_text}:\n')
    logging.info(f'translate function: "{header_text}" done.')
    return lines

def read_file(filename: str):
    with open(filename, 'rb') as fp:
        encoding = chardet.detect(fp.read())['encoding']
    with open(filename, 'r', encoding=encoding) as fp:
        lines = fp.readlines()
    return lines


def translate_file(file_basename:str, 
    input_path:str, 
    output_path:str, 
    arguments_value:list=None,
    replaced_functions={},
    logging_level=logging.INFO, 
    indent=4,
    **kwargs):
    """
    parameters
    ----------
    arguments_value: list
        the value of arguments.
    """
    full_basename = os.path.join(input_path, file_basename).replace('\\', '/')
    logging.info(f'translate "{full_basename}.m" start.')
    logging.info(f'read "{full_basename}.m"')
    lines = read_file(f'{full_basename}.m')
    full_basename = os.path.join(output_path, file_basename).replace('\\', '/')
    # scan
    logging.info('scan start.')
    tokens = scan(lines)
    if __debug__ and logging_level == logging.DEBUG:
        fn = full_basename + '_scan.json'
        logging.debug(f'write "{fn}"')
        with open(fn, 'w') as fp:
            lst = [token.todict() for token in tokens]
            json.dump(lst, fp, indent=4)
    logging.info('scan done.')
    # parse
    logging.info('parse start.')
    tokens = parse(tokens)

    if __debug__ and logging_level == logging.DEBUG:
        fn = full_basename + '_parse.json'
        logging.debug(f'write "{fn}"')
        with open(fn, 'w') as fp:
            lst = [token.todict() for token in tokens]
            json.dump(lst, fp, indent=4)
    logging.info('parse done.')

    # translate
    pyfilename = full_basename + '.py'
    # clear python file
    with open(pyfilename, 'w') as fp:
        pass
    _locals = {}
    # at the module level, locals() and globals() are the same dictionary.
    _globals = _locals
    imports = {}
    import_as = {}
    exec('import sys', _globals, _locals)
    _locals['sys'].path[0] = output_path
    if arguments_value is None:
        # this is a script file
        # remove functions from script file
        functions = {}
        for i, token in enumerate(tokens):
            if token.ttype == TT_BLOCK and token.text == 'function':
                tokens_funcs = tokens[i:]
                for token in tokens_funcs:
                    if token.ttype == TT_BLOCK and token.text == 'function':
                        func_name, _, _, _ = split_func_block(token)
                        functions[func_name] = token
                tokens = tokens[:i]
                break
    else:
        # this is a function file
        # find top function. the name of top function is same as the filename
        for i, token in enumerate(tokens):
            if token.ttype == TT_BLOCK and token.text == 'function':
                func_name, arguments, _, _ = split_func_block(token)
                if func_name == file_basename:
                    top_func = token
                    functions:'dict[str, Token]' = {func_name: token}    
                    break
                else:
                    raise Exception(f'The function name "{func_name}" is '+ 
                        f'different from the file name "{file_basename}".')

        for i, av in enumerate(arguments_value):
            _locals[arguments[i].text] = av
    
        for token in tokens[i+1:]:
            if token.ttype == TT_BLOCK and token.text == 'function':
                func_name, _, _, _ = split_func_block(token)
                functions[func_name] = token
        
    if arguments_value is None:
        lines = translate_staments(tokens, 
            _globals=_globals,
            _locals=_locals,
            file_basename=file_basename,
            input_path=input_path,
            output_path=output_path,
            replaced_functions=replaced_functions,
            imports=imports,
            import_as=import_as,
            functions=functions, 
            indent=indent,
            logging_level=logging_level)
    else:
        lines = translate_function(top_func,
            _globals=_globals,
            _locals=_locals,
            file_basename=file_basename,
            input_path=input_path,
            output_path=output_path,
            replaced_functions=replaced_functions,
            imports=imports,
            import_as=import_as,
            functions=functions, 
            indent=indent,
            logging_level=logging_level)

    prelines = generate_import_as(import_as)
    prelines.extend(generate_imports(imports))
    with open(pyfilename, 'r') as fp:
        func_lines = fp.readlines()
    with open(pyfilename, 'w') as fp:
        fp.writelines([text + '\n' for text in prelines])
        if len(prelines) > 0:
            fp.write('\n\n')
        fp.writelines(func_lines)
        if len(func_lines) > 0:
            fp.write('\n')
        fp.writelines(lines)

    if __debug__ and logging_level == logging.DEBUG:
        fn = full_basename + '_trans.json'
        logging.debug(f'write "{fn}"')
        with open(fn, 'w') as fp:
            lst = [token.todict() for token in tokens]
            json.dump(lst, fp, indent=4)
    logging.info(f'translate "{file_basename}" done.')


             