"""
Token class implement.

This is a part of motopy.
"""
import copy
from .constants import *
class Token:
    """ Token class
    """
    def __init__(self, ttype: str, text: str, row: int=None, col: int=None, 
                value = None, lchild=None, rchild=None, children=[], 
                parent=None) -> None:
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
        self.set_lchild(lchild)
        self.set_rchild(rchild)
        self.set_children(children)
        self.value = value
        self.parent: Token = parent
        self.translated = False

    def set_lchild(self, token:'Token'=None):
        assert isinstance(token, Token) or token is None
        self.lchild:Token = token
        if token is not None:
            token.parent = self

    def set_rchild(self, token:'Token'=None):
        assert isinstance(token, Token) or token is None
        self.rchild:Token = token
        if token is not None:
            token.parent = self

    def set_children(self, children:'list[Token]'=[]):
        assert isinstance(children, list)
        self.children = children[:]
        for token in children:
            token.parent = self

    def extend_children(self, children:'list[Token]'=[]):
        assert isinstance(children, list)
        for t in children:
            self.children.append(t)
            t.parent = self

    def copy(self):
        return copy.deepcopy(self)
        
    def __repr__(self) -> str:
        return str(self.todict())

    def todict(self):
        d = {'text' : self.text, 'ttype' : self.ttype}
        if self.lchild is not None:
            d['lchild'] = self.lchild.todict()
        if self.rchild is not None:
            d['rchild'] = self.rchild.todict()

        if len(self.children) > 0:
            d['children'] = [t.todict() for t in self.children]
        d['row'] = self.row
        d['col'] = self.col

        return d