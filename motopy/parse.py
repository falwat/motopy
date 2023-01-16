"""
Parse tokens stream to token tree.

This is a part of motopy.
"""
import json
from .constants import *
from .token import Token

def parse_bracket_inner(tokens: 'list[Token]'):
    """
    parse tokens in bracket
    """
    if len(tokens) == 0:
        return []
    rows:'list[Token]' = [Token(TT_OP, ',', row=tokens[0].row, col=tokens[0].col, children=[])]
    si = 0
    for i, token in enumerate(tokens):
        # element split
        if token.ttype == TT_SYM and token.text == ',':
            e = parse_expression(tokens[si:i])
            rows[-1].children.append(e)
            e.parent = rows[-1]
            si = i + 1
        # row split
        elif token.ttype == TT_SYM and token.text == ';':
            e = parse_expression(tokens[si:i])
            rows[-1].children.append(e)
            e.parent = rows[-1]
            row = Token(TT_OP, ',', row=tokens[si].row, col=tokens[si].col, children=[])
            rows.append(row)
            si = i + 1
        elif token.ttype == TT_EOL:
            si = i + 1
    if len(tokens) != 0:
        e = parse_expression(tokens[si:])
        rows[-1].children.append(e)
        e.parent = rows[-1]
    if len(rows) == 1:
        return rows[0].children
    else:
        return rows


def parse_expression_without_bracket(tokens: 'list[Token]'):
    if len(tokens) == 0:
        return None
    elif len(tokens) == 1:
        return tokens[0]
    pos = 0
    prior = -1
    for i, token in enumerate(tokens):
        if token.ttype in [TT_SYM, TT_OP] and op_priors[token.text] >= prior:
            pos = i
            prior = op_priors[token.text]
    token = tokens[pos]
    token.ttype = TT_OP
    if token.text in binary_operators:
        token.set_lchild(parse_expression_without_bracket(tokens[:pos]))
        token.set_rchild(parse_expression_without_bracket(tokens[pos+1:]))
    elif token.text in left_unary_operators:
        token.set_rchild(parse_expression_without_bracket(tokens[pos+1:]))
    elif token.text in right_unary_operators:
        token.set_lchild(parse_expression_without_bracket(tokens[:pos]))
    elif token.text in ['()', '[]', '{}']:
        if pos > 0 and tokens[pos-1].ttype == TT_ID:
            token.text = '?' + token.text
            token.set_lchild(tokens[pos-1])
    return token

def parse_bracket(tokens: 'list[Token]', bracket_opened: str = None):
    bracket_pairs = {'(':')', '[':']', '{':'}'}  
    bracket_closed = None
    if bracket_opened != None:
        bracket_closed = bracket_pairs[bracket_opened]
    i = 0
    while i < len(tokens):
        if tokens[i].text == bracket_closed:
            tokens[i].set_children(parse_bracket_inner(tokens[:i]))
            tokens[i].text = bracket_opened + bracket_closed
            tokens[i].ttype = TT_OP
            return tokens[i:]
        elif tokens[i].text in bracket_pairs:
            tokens = tokens[:i] + parse_bracket(tokens[i+1:], tokens[i].text)
        i += 1
    return tokens

def parse_expression(tokens: 'list[Token]'):
    i = 0
    while i < len(tokens):
        if tokens[i].text in bracket_pairs:
            tokens = tokens[:i] + parse_bracket(tokens[i+1:], tokens[i].text)
        i += 1
    return parse_expression_without_bracket(tokens)

def parse_cmd(tokens: 'list[Token]'):
    token = tokens[0]
    token.ttype = TT_CMD
    token.set_children(tokens[1:])
    return token

def parse_block(tokens: 'list[Token]', kw: str = None):
    statments:'list[Token]' = []
    stack = []
    # start index
    si = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.ttype == TT_SYM:
            if token.text in bracket_pairs:
                stack.append(token.text)
            elif token.text in [')', ']', '}'] and \
                token.text == bracket_pairs[stack[-1]]:
                stack.pop()
            elif token.text == ';' and len(stack) == 0:
                if tokens[si].ttype == TT_ID and tokens[si].text in matlab_commands:
                    t = parse_cmd(tokens[si:i])
                else:
                    t = parse_expression(tokens[si:i])
                statments.append(t)
                si = i + 1 
        elif token.ttype == TT_EOL and len(stack) == 0:
            if si == i:
                statments.append(token)
            else:
                if tokens[si].ttype == TT_ID and tokens[si].text in matlab_commands:
                    t = parse_cmd(tokens[si:i])
                else:
                    t = parse_expression(tokens[si:i])
                statments.append(t)
            si = i + 1
        elif token.ttype == TT_KW:
            if token.text == 'end' and len(stack) == 0:
                return statments, tokens[i+1:]
            elif kw in ['if', 'elseif'] and \
                token.text in ['elseif', 'else']:
                return statments, tokens[i:]
            elif token.text in matlab_blocks:
                ss, rem_tokens = parse_block(tokens[i+1:], kw=token.text)
                token.set_children(ss)
                token.ttype = TT_BLOCK
                statments.append(token)
                tokens = tokens[:i+1] + rem_tokens
                si = i + 1
        i += 1
    if si < len(tokens):
        statments.append(parse_expression(tokens[si:]))
    return statments, tokens  

def parse(tokens: 'list[Token]'):
    statments, _ = parse_block(tokens)
    return statments

if __name__ == '__main__':
    from scan import scan
    with open('tests/array_test.m', 'r') as fp:
        lines = fp.readlines()
    tokens = scan(lines)
    with open('tests/scan_array_test.json', 'w') as fp:
        lst = [token.todict() for token in tokens]
        json.dump(lst, fp, indent=4)
    # token_expr = parse_expression(tokens)
    statments = parse(tokens)
    with open('tests/parse_array_test.json', 'w') as fp:
        lst = [token.todict() for token in statments]
        json.dump(lst, fp, indent=4)
