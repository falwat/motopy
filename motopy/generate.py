"""
Generate text from token.

This is a part of motopy.
"""
from .constants import *
from .token import Token

def generate_expression(token: Token):
    """
    """
    assert token.ttype == TT_OP
    ltoken: Token = token.lchild
    rtoken: Token = token.rchild
    lines:'list[str]' = []
    if token.ttype in [TT_ID, TT_NUM]:
        lines.append(token.text)
    elif token.ttype == TT_STR:
        lines.append(f"'{token.text}'")
    elif token.text in ['()', '[]']:
        values = []
        for subtoken in token.children:
            values.append(generate(subtoken)[0])
        lines.append(token.text[0] + ", ".join(values) + token.text[-1])
    elif token.text == ',':
        lines.append(', '.join([generate(t)[0] for t in token.children]))
    elif token.text == ';':
        lines.append('; '.join([generate(t)[0] for t in token.children]))
    elif token.text == ':':
        text = ''
        if ltoken is not None:
            text += generate(ltoken)[0]
        text += ':'
        if rtoken is not None:
            text += generate(rtoken)[0]
        lines.append(text)
    elif token.text == '.':
        text = ''
        if ltoken is not None:
            text += generate(ltoken)[0]
        text += '.'
        if rtoken is not None:
            text += generate(rtoken)[0]
        lines.append(text)
    elif token.text in binary_operators:
        text = ''
        if ltoken is not None:
            text += generate(ltoken)[0]
        text +=f' {token.text} '
        if rtoken is not None:
            text += generate(rtoken)[0]
        lines.append(text)
    elif token.text in left_unary_operators:
        text = ' ' + token.text + generate(rtoken)[0]
        lines.append(text)
    elif token.text in right_unary_operators:
        text = generate(ltoken)[0] + f'{token.text} '
        lines.append(text)
    elif token.text == '?[]':
        text = generate(ltoken)[0] + '[' + \
            ', '.join([generate(t)[0] for t in token.children]) + ']'
        lines.append(text)
    elif token.text == '?()':
        # variable indexing
        text = generate(ltoken)[0] + '(' + \
            ', '.join([generate(t)[0] for t in token.children]) + ')'
        lines.append(text)
    else:
        text = ''
        if token.lchild is not None:
            text += generate(token.lchild)[0]
        text += f' {token.text} '
        if token.rchild is not None:
            text += generate(token.rchild)[0]
        lines.append(text)
    return lines


def generate_block(token: Token, indent:int=4):
    assert token.ttype == TT_BLOCK
    last_ttype = None
    lines: 'list[str]' = []
    if token.text in ['if', 'elif', 'for']:
        text = token.text + ' '
        text += generate_expression(token.children[0])[0]
        text += ':'
        lines.append(text)
        last_ttype = token.children[0].ttype
        si = 1
    elif token.text == 'else':
        text = token.text + ':'
        lines.append(text)
        last_ttype = token.ttype
        si = 0
    for i in range(si, len(token.children)):
        inner_lines = generate(token.children[i])
        if token.children[i].ttype == TT_EOL:
            if len(lines) == 0 or last_ttype == TT_EOL:
                lines.append(f'{inner_lines[0]}')
            elif len(inner_lines[0]) > 0:
                lines[-1] = f'{lines[-1].rstrip()}  {inner_lines[0]}'
        else:
            lines.extend([' ' * indent +  text for text in inner_lines])
    return lines

def generate_for_block(token: Token, indent:int=4):
    assert token.ttype == TT_BLOCK and token.text == 'for'
    lines: 'list[str]' = []
    if token.children[0].text != '=':
        raise Exception('generate for block error.')
    else:
        text = ' '*indent + 'for ' + token.children[0].lchild.text + ' in '
        text += generate_expression(token.children[0].rchild)
        text += ':'
        lines.append(text)
        for i in range(1, len(token.children)):
            text = ' '*(indent+indent) + generate_expression(token.children[i])
            lines.append(text)
    return lines

def generate_imports(imports: 'dict[str, set[str]]'):
    lines: 'list[str]' = []
    # lines.append('import numpy as np')
    for module_name in imports:
        text = f'from {module_name} import ' + ', '.join(imports[module_name])
        lines.append(text)
    return lines

def generate_import_as(import_as: 'dict[str, set[str]]'):
    lines: 'list[str]' = []
    for module_name in import_as:
        text = f'import {module_name} as {import_as[module_name]}'
        lines.append(text)
    return lines

def generate(token: Token, indent=4, **kwargs):
    lines: 'list[str]' = []
    if token.ttype in [TT_ID, TT_NUM]:
        lines.append(token.text)
    elif token.ttype == TT_STR:
        lines.append(f"'{token.text}'")
    elif token.ttype == TT_FMT_STR:
        lines.append(f"f'{token.text}'")
    elif token.ttype == TT_BLOCK:
        lines.extend(generate_block(token))
    elif token.ttype == TT_OP:
        lines.extend(generate_expression(token))
    elif token.ttype == TT_EOL:
        lines.append(token.text)
    elif token.ttype == TT_CMD:
        if token.text == 'global':
            lines.append('global ' + ', '.join([t.text for t in token.children]))
    else:
        lines.append(token.text)
    return lines

