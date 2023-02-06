"""
Generate text from token.

This is a part of motopy.
"""
from .constants import *
from .token import Token, RootToken

def generate_expression(token: Token, **kwargs):
    """
    """
    assert token.ttype in [TT_ID, TT_NUM, TT_STR, TT_OP]
    ltoken: Token = token.lchild
    rtoken: Token = token.rchild
    lines:'list[str]' = []
    if token.ttype in [TT_ID, TT_NUM]:
        lines.append(token.text)
    elif token.ttype == TT_STR:
        lines.append(f"'{token.text}'")
    elif token.text == 'for..in':
        text = generate(token.children[0], **kwargs)[0]
        text += f' for {generate(token.lchild, **kwargs)[0]} in {generate(token.rchild, **kwargs)[0]}'
        lines.append(text)
    elif token.text in ['()', '[]', '{}']:
        values = []
        for subtoken in token.children:
            values.append(generate(subtoken, **kwargs)[0])
        lines.append(token.text[0] + ", ".join(values) + token.text[-1])
    elif token.text == ',':
        lines.append(', '.join([generate(t, **kwargs)[0] for t in token.children]))
    elif token.text == ';':
        lines.append('; '.join([generate(t, **kwargs)[0] for t in token.children]))
    elif token.text == ':':
        text = ''
        if ltoken is not None:
            text += generate(ltoken, **kwargs)[0]
        text += ':'
        if rtoken is not None:
            text += generate(rtoken, **kwargs)[0]
        lines.append(text)
    elif token.text == '::':
        text = ''
        if ltoken is not None:
            text += generate(ltoken, **kwargs)[0]
        text += ':'
        if len(token.children) > 0:
            text += generate(token.children[0])[0]
        text += ':'
        if rtoken is not None:
            text += generate(rtoken, **kwargs)[0]
        lines.append(text)
    elif token.text == '.':
        text = ''
        if ltoken is not None:
            text += generate(ltoken, **kwargs)[0]
        text += '.'
        if rtoken is not None:
            text += generate(rtoken, **kwargs)[0]
        lines.append(text)
    elif token.text in binary_operators:
        text = ''
        if ltoken is not None:
            text += generate(ltoken, **kwargs)[0]
        text +=f' {token.text} '
        if rtoken is not None:
            text += generate(rtoken, **kwargs)[0]
        lines.append(text)
    elif token.text in left_unary_operators:
        text = ' ' + token.text + generate(rtoken, **kwargs)[0]
        lines.append(text)
    elif token.text in right_unary_operators:
        text = generate(ltoken, **kwargs)[0] + f'{token.text} '
        lines.append(text)
    elif token.text == '?[]':
        text = generate(ltoken, **kwargs)[0] + '[' + \
            ', '.join([generate(t, **kwargs)[0] for t in token.children]) + ']'
        lines.append(text)
    elif token.text == '?()':
        # variable indexing
        text = generate(ltoken, **kwargs)[0] + '(' + \
            ', '.join([generate(t, **kwargs)[0] for t in token.children]) + ')'
        lines.append(text)
    else:
        text = ''
        if token.lchild is not None:
            text += generate(token.lchild, **kwargs)[0]
        text += f' {token.text} '
        if token.rchild is not None:
            text += generate(token.rchild, **kwargs)[0]
        lines.append(text)
    return lines


def generate_block(token: Token, indent:int=4, returns=None, **kwargs):
    assert token.ttype == TT_BLOCK
    _indent = indent
    last_ttype = token.ttype
    lines: 'list[str]' = []
    si = 0
    if token.text in ['if', 'elif', 'for', 'while']:
        text = token.text + ' '
        text += generate_expression(token.children[0], **kwargs)[0]
        text += ':'
        lines.append(text)
        last_ttype = token.children[0].ttype
        si = 1
    elif token.text == 'if..else':
        _indent = 0
    elif token.text == 'for..in':
        text = 'for ' + token.lchild.text + ' in ' + generate(token.rchild, **kwargs)[0] + ':'
        lines.append(text)
    elif token.text == 'def':
        text = 'def ' + generate(token.rchild, indent=indent, **kwargs)[0] + ':'
        lines.append(text)
        sublines = []
        for fn in token.functions:
            sublines.extend(generate_block(token.functions[fn], indent=indent, **kwargs))
        lines.extend([' ' * _indent + text for text in sublines])
    elif token.text in ['else', 'try', 'catch', 'except']:
        text = token.text + ':'
        lines.append(text)
    for i in range(si, len(token.children)):
        inner_lines = generate(token.children[i], indent=indent, returns=returns, **kwargs)
        if token.children[i].ttype == TT_EOL:
            if len(lines) == 0 or last_ttype == TT_EOL:
                lines.append(f'{inner_lines[0]}')
            elif len(inner_lines[0]) > 0:
                lines[-1] = f'{lines[-1].rstrip()}  {inner_lines[0]}'
        else:
            lines.extend([' ' * _indent +  text for text in inner_lines])
    return lines

def generate_root(root: RootToken, indent=4, **kwargs):
    assert root.ttype == TT_ROOT
    lines = generate_imports(root.imports)
    if len(lines) > 0:
        lines.extend(['', ''])
    for fn in root.functions:
        token = root.functions[fn]
        sublines = generate(token, returns=token.lchild, indent=indent, **kwargs)
        lines.extend(sublines)
        lines.append('')
    last_ttype = TT_EOL
    skiped = False
    for token in root.children:
        if token.ttype == TT_CODE:
            lines.append(token.text)
            skiped = True
        elif skiped == True:
            skiped = False
        else:
            sublines = generate(token, **kwargs)
            if token.ttype == TT_EOL:
                if len(lines) == 0 or last_ttype == TT_EOL:
                    lines.append(f'{sublines[0]}')
                elif len(sublines[0]) > 0:
                    lines[-1] = f'{lines[-1].rstrip()}  {sublines[0]}\n'
            else:
                lines.extend(sublines)
            last_ttype = token.ttype
    return lines

def generate_imports(imports: 'dict[str, dict[str, str]]'):
    lines: 'list[str]' = []
    for from_ in imports:
        items = []
        for import_, as_ in imports[from_].items():
            if as_ is not None:
                items.append(f'{import_} as {as_}')
            else:
                items.append(import_)        
        if from_ is None:
            for item in items:
                lines.append(f'import {item}')
        else:
            lines.append(f'from {from_} import ' + ', '.join(items))
    return lines

def generate(token: Token, indent=4, returns:Token = None, **kwargs):
    lines: 'list[str]' = []
    if token == None:
        lines.append('')
    elif token.ttype in [TT_ID, TT_NUM]:
        lines.append(token.text)
    elif token.ttype == TT_STR:
        lines.append(f"'{token.text}'")
    elif token.ttype == TT_FMT_STR:
        lines.append(f"f'{token.text}'")
    elif token.ttype == TT_BLOCK:
        lines.extend(generate_block(token, indent=indent, returns=returns, **kwargs))
    elif token.ttype == TT_OP:
        lines.extend(generate_expression(token, **kwargs))
    elif token.ttype == TT_EOL:
        lines.append(token.text)
    elif token.ttype == TT_ROOT:
        lines.extend(generate_root(token, indent=indent, **kwargs))
    elif token.ttype == TT_CMD:
        if token.text == 'global':
            lines.append('global ' + ', '.join([t.text for t in token.children]))
    elif token.ttype == TT_KW and token.text == 'return' and returns != None:
        lines.append(f'return {generate(returns, **kwargs)[0]}')
    else:
        lines.append(token.text)
    return lines

