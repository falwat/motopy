"""
scan .m file to tokens stream.

This is a part of motopy.
"""
import json
import re
from .constants import *
from .token import Token, RootToken
from .utils import ScanError

re_id = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*')
re_number = re.compile(r'^([0-9]*\.?[0-9]+|[0-9]+\.?[0-9]*)([eE][+-]?[0-9]+)?')
re_compare = re.compile(r'<=?|>=?|==|~=')
# check number is a int number
re_int = re.compile(r'^[0-9]*[0-9]$')

def scan_post(tokens: 'list[Token]'):
    """
    task 1: Convert unary operators '+' to '+?' and '-' to '-?'.
    task 2: check complex number and combine 'num' with 'j'
    """
    i = 1
    while i < len(tokens):
        if tokens[i].ttype == TT_SYM and tokens[i].text in '+-':
            if not ((tokens[i-1].ttype == TT_SYM and tokens[i-1].text in ')]}') or
                tokens[i-1].ttype in [TT_ID, TT_NUM]):
                if i < len(tokens)-1 and tokens[i+1].ttype == TT_NUM:
                    tokens[i+1].text = tokens[i].text + tokens[i+1].text
                    tokens.pop(i)
                else:
                    tokens[i].text += '?'
            
        if tokens[i-1].ttype == TT_NUM and tokens[i].ttype == TT_ID and \
            tokens[i].text in ['i', 'j']:
            tokens[i-1].text += 'j'
            tokens[i-1].value = complex(tokens[i-1].text)
            tokens.pop(i)
            continue
        i += 1
    return tokens

def scan(lines : 'list[str]'):
    root = RootToken(TT_ROOT, '')
    tokens = []
    for n, line in enumerate(lines, start=1):
        pos = 0
        while pos < len(line):
            while line[pos] in " \t":
                pos += 1
            if line[pos] == '"':
                pos +=1
                str_len = line[pos:].find('"')
                if str_len == -1:
                    raise ScanError('quotation marks(") mismatched.', n, pos+1)
                else:
                    tokens.append(Token(TT_STR, line[pos:pos+str_len], n, pos + 1, root=root))
                    pos += str_len + 1
                    continue
            if line[pos] == "'":
                if (tokens[-1].ttype == TT_ID and line[pos-1] not in ' \t') or \
                    (tokens[-1].ttype == TT_SYM and tokens[-1].text in ")]}"):
                    tokens.append(Token(TT_SYM, "'", n, pos + 1, root=root))
                    pos += 1
                    continue
                pos +=1
                str_len = line[pos:].find("'")
                if str_len == -1:
                    raise ScanError("quotation marks(') mismatched.", n, pos+1)
                else:
                    tokens.append(Token(TT_STR, line[pos:pos+str_len], n, pos + 1, root=root))
                    pos += str_len + 1
                    continue
            if line[pos] in "\r\n":
                tokens.append(Token(TT_EOL, "", n, pos + 1, root=root))
                break
            if line[pos] == '%':
                if line[pos:].startswith('%%>'):
                    text = line[pos:].replace('%%>', '', 1)
                    tokens.append(Token(TT_CODE, text.strip(), n, pos + 1, translated=True, root=root))
                else:
                    tokens.append(Token(TT_EOL, line[pos:].strip(), n, pos + 1, root=root))
                break
            if line[pos] in ",;[](){}":
                tokens.append(Token(TT_SYM, line[pos], n, pos + 1, root=root))
                pos += 1
                continue
            m = re_id.match(line[pos:])
            if m is not None:
                text = m.group(0)
                if text in matlab_keywords:
                    tokens.append(Token(TT_KW, m.group(0), n, pos + 1, root=root))
                else:
                    tokens.append(Token(TT_ID, m.group(0), n, pos + 1, root=root))
                pos += m.span(0)[1]
                continue
            m = re_number.match(line[pos:])
            if m is not None:
                text = m.group(0)
                try:
                    value = int(text)
                except:
                    value = float(text)
                tokens.append(Token(TT_NUM, m.group(0), n, pos + 1, root=root))
                pos += m.span(0)[1]
                continue
            m = re_compare.match(line[pos:])
            if m is not None:
                tokens.append(Token(TT_SYM, m.group(0), n, pos + 1, root=root))
                pos += m.span(0)[1]
                continue
            if line[pos] in "+-":
                tokens.append(Token(TT_SYM, line[pos], n, pos + 1, root=root))
                pos += 1
                continue
            if line[pos] in "*/^":
                tokens.append(Token(TT_SYM, line[pos], n, pos + 1, root=root))
                pos += 1
                continue
            if line[pos] in "=":
                tokens.append(Token(TT_SYM, '=', n, pos + 1, root=root))
                pos += 1
                continue
            if line[pos] == '.':
                pos += 1
                if line[pos] in "*/^'":
                    tokens.append(Token(TT_SYM, line[pos-1:pos+1], n, pos + 1, root=root))
                    pos += 1
                else:
                    tokens.append(Token(TT_SYM, '.', n, pos + 1, root=root))
                continue
            # :
            if line[pos] == ':':
                tokens.append(Token(TT_SYM, ':', n, pos + 1, root=root))
                pos += 1
                continue
            # not(~), placeholder
            if line[pos] == '~':
                tokens.append(Token(TT_SYM, '~', n, pos + 1, root=root))
                pos += 1
                continue
            # & &&
            if line[pos] == '&':
                pos += 1
                if line[pos] == '&':
                    tokens.append(Token(TT_SYM, '&&', n, pos + 1, root=root))
                    pos += 1
                else:
                    tokens.append(Token(TT_SYM, '&', n, pos + 1, root=root))
                continue
            # | ||
            if line[pos] == '|':
                pos += 1
                if line[pos] == '|':
                    tokens.append(Token(TT_SYM, '||', n, pos + 1, root=root))
                    pos += 1
                else:
                    tokens.append(Token(TT_SYM, '|', n, pos + 1, root=root))
                continue
            raise ScanError(line[pos:], n, pos+1)
    tokens = scan_post(tokens)
    root.set_children(tokens)
    return root