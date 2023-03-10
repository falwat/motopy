"""
Token class implement.

This is a part of motopy.
"""
import copy
from typing import Union
from .utils import ExecError
from .constants import *

class Token:
    """ Token class
    """
    def __init__(self, ttype: str, text: str, row: int=None, col: int=None, 
                value = None, lchild=None, rchild=None, children=[], 
                translated=False, parent=None, root=None) -> None:
        """
        parameters
        ----------
        ttype: token type.
        """
        assert isinstance(text, str)
        self.ttype = ttype
        self.text = text
        self.value = value
        self.row = row
        self.col = col
        self.root:'RootToken' = root
        self.set_lchild(lchild)
        self.set_rchild(rchild)
        self.set_children(children)
        self.value = value
        self.parent: Token = parent
        self.functions:'dict[str, Token]' = {}
        self.translated = translated

    def set_lchild(self, token:'Union[Token, None]'=None):
        assert isinstance(token, Token) or token is None
        self.lchild = token
        if token is not None:
            token.parent = self

    def remove_lchild(self):
        token = self.lchild
        self.lchild = None
        return token

    def set_rchild(self, token:'Union[Token, None]'=None):
        assert isinstance(token, Token) or token is None
        self.rchild:Token = token
        if token is not None:
            token.parent = self

    def remove_rchild(self):
        token = self.rchild
        self.rchild = None
        return token

    def set_children(self, children:'list[Token]'=[]):
        assert isinstance(children, list)
        self.children = children[:]
        for token in children:
            if token != None:
                token.parent = self

    def append_children(self, token:'Token'):
        assert isinstance(token, Token)
        token.parent = self
        self.children.append(token)


    def extend_children(self, children:'list[Token]'=[]):
        assert isinstance(children, list)
        for t in children:
            self.children.append(t)
            t.parent = self

    def pop_child(self, _index: int):
        return self.children.pop(_index)

    def is_lchild(self):
        return self.parent != None and self == self.parent.lchild

    def is_rchild(self):
        return self.parent != None and self == self.parent.rchild

    def trace_back(self, ttype:int, text:str):
        last_token = self
        parent_token = self.parent
        nlayer = 0
        while parent_token != None:
            if parent_token.ttype == ttype and parent_token.text == text:
                if last_token == parent_token.lchild:
                    return parent_token, 'left', nlayer
                elif last_token == parent_token.rchild:
                    return parent_token, 'right', nlayer
                else:
                    for i, token in enumerate(parent_token.children):
                        if token == last_token:
                            return parent_token, i, nlayer
            last_token = parent_token
            parent_token = parent_token.parent
            nlayer += 1
        return None, None, -1


    def move(self):
        token = copy.copy(self)
        # token = Token(ttype=self.ttype, text=self.text, row=self.row, col=self.col, 
        #     value = self.value, translated=self.translated, 
        #     lchild=self.lchild, rchild=self.rchild, children=self.children, 
        #     parent=self.parent, root=self.root)
        self.lchild = None
        self.rchild = None
        self.children = []
        return token


    def replaced_with(self, token:'Token'):
        if self.parent != None:
            if self == self.parent.lchild:
                self.parent.set_lchild(token)
            elif self == self.parent.rchild:
                self.parent.set_rchild(token)
            else:
                for i, t in enumerate(self.parent.children):
                    if self == t:
                        self.parent.children[i] = token
                        token.parent = self.parent
                        break
                else:
                    raise Exception(f"self can't be replaced with token")
        else:
            raise Exception(f"self can't be replaced with token")
        
    def __repr__(self) -> str:
        return str(self.todict())

    def todict(self):
        d = {'text' : self.text, 'ttype' : self.ttype}
        if self.lchild is not None:
            d['lchild'] = self.lchild.todict()
        if self.rchild is not None:
            d['rchild'] = self.rchild.todict()

        if len(self.children) > 0:
            d['children'] = [t.todict() if t != None else None for t in self.children]

        if len(self.functions) > 0:
            d['functions'] = {fn:self.functions[fn].todict() for fn in self.functions}
        d['row'] = self.row
        d['col'] = self.col

        return d



class RootToken(Token):
    def __init__(self, ttype: str, text: str, children=[]) -> None:
        super().__init__(ttype, text, children=children)
        # {_from:{_import:_as}}
        self.imports:'dict[str,dict[str,str]]' = {}
        self._globals = {}
        self._locals = self._globals

    def add_import(self, import_: str, *, from_:str=None, as_:str=None, noexe=False):
        """
        root.add_import('json') --> import json
        root.add_import('numpy', as_='np') --> import numpy as np
        root.add_import('random', from_='numpy') --> from numpy import random
        root.add_import('pyplot', from_='matplotlib', as_='plt') --> from matplotlib import pyplot as plt
        """
        text = ''
        if from_ is not None:
            text += f'from {from_} '
        text += f'import {import_} '
        if as_ is not None:
            text += f'as {as_}'

        if from_ in self.imports:
            if import_ in self.imports[from_]:
                return
            else:
                self.imports[from_][import_] = as_
        else:
            self.imports[from_] = {import_:as_}
        
        if noexe == False:
            exec(text, self._globals, self._globals)
            if id(self._locals) != id(self._globals):
                exec(text, self._globals, self._locals)

    def exec_(self, text: str):
        exec(text, self._globals, self._locals)

    def eval_expr(self, token: Token):
        assert token.ttype in [TT_OP, TT_NUM, TT_STR, TT_ID]
        from .generate import generate
        text = generate(token)[0]
        try:
            value = eval(text, self._globals, self._locals)
        except:
            raise ExecError(text, self.text, token.row)
        return value

    def build_token(self, value: Union[int, float, complex, str, Token]):
        if isinstance(value, str):
            token = Token(TT_STR, value, root=self)
        elif isinstance(value, (int, float, complex)):
            token = Token(TT_NUM, str(value), root=self)
        elif isinstance(value, Token):
            token = value
        else:
            raise Exception(f"can't build token for type: {type(value)}")
        return token

    def build_id(self, name: str):
        return Token(TT_ID, name, translated=True, root=self)

    def build_dot(self, v:Union[str, Token], m:Union[str, Token]):
        tv = self.build_id(v) if type(v)==str else v
        tm = self.build_id(m) if type(m)==str else m
        token = Token(TT_OP, '.', lchild=tv, rchild=tm, root=self)

    def build_func(self, name: Union[Token, str], *args: Union[int, float, complex, str, Token]):
        if isinstance(name, str):
            name = Token(TT_ID, name, root=self)
        for i, arg in enumerate(args):
            if not isinstance(arg, Token):
                args[i] = self.build_token(arg)
        token = Token(TT_OP, '?()', lchild=name, children=[*args], root=self)
        return token

    def build_for_in(self, vname: str, tc: Token, ts: Token):
        """`s` for `i` in `c`"""
        token = Token(TT_OP, 'for..in', lchild=Token(TT_ID, vname), rchild=tc, children=[ts], root=self)
        token.translated = tc.translated & ts.translated
        return token

    def build_dict(self, **kwargs):
        pairs = []
        for kw in kwargs:
            pairs.append(Token(TT_OP, ':', lchild=self.build_token(kw), rchild=kwargs[kw], root=self))
        token = Token(TT_OP, '{}', children=pairs, root=self)
        return token

    def build_list(self, *args):
        token = Token(TT_OP, '[]', children=[self.build_token(arg) for arg in args])
        token.translated = all([t.translated for t in token.children])
        return token
        
