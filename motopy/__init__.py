"""
motopy
======

The powerfull tool of converting Matlab/Octave code TO PYthon.

Source Code and tutorial can be found:
    https://github.com/falwat/motopy

Contact:
--------
    Jackie Wang <falwat@163.com>

How to Use
----------

Import `motopy` and use `motopy.make()` translate your mfile:

```py
import motopy

motopy.make(
    entryfile='<the script filename without extension(*.m)>',
    input_path='<the input path of *.m files>', 
    output_path='<the output path of *.py files>' 
)
```

Please see [readme](https://github.com/falwat/motopy) for more information.

License:
--------
MIT License

Copyright (c) 2023 falwat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '2.0.0'
__all__ = ['make', 'WARN', 'INFO', 'DEBUG']
__author__ = 'Jackie Wang <falwat@163.com>'

import os
import logging
from logging import WARN, INFO, DEBUG
from .constants import *
from .translate import translate_file

console_handler = None

def make(entry_basename: str, input_path:str='.', output_path:str='.', * ,
        replaced_functions:'dict[str, tuple[str, str]]'= {},
        indent:int=4,
        logging_file:str='motopy.log', logging_level = INFO, **kwargs):
    """
    Read the `entryfile(*.m)` from `input_path`, translate to python file(*.py), and write to `output_path`.
    the `entryfile` is a matlab/octave script file with extension of ".m".
    if other function files(*.m) in `input_path` called by this script file, will also be translated to.

    Parameters
    ----------
    `entryfile`: str
        A matlab/octave script file(`.m`) that may be call other functions.
    `input_path`: str
        The path that input ".m" files loaded.
    `output_path`: str
        The path that output ".py" files stored.
    `replaced_functions`: 'dict[str, tuple[str, str]]' 
        A dictionary indicated the functions that do not require translated.
        {'func_name_in_mfile': ('pyfile_name', 'func_name_in_pyfile'), ...}
    `indent`: int
        indent using space. Default: 4. 
    `logging_file`: str
        The logging file name. Default: 'motopy.log'.
        If `logging_file` does not contain path, the log file will be saved to `output_path`.
    `logging_level`: INFO|DEBUG
        Set the root logger level to the specified level. 
        @see: `logging.basicConfig()`

    """
    if os.path.isabs(input_path):
        input_path = input_path.replace('\\', '/')
    else:
        input_path = os.path.abspath(input_path).replace('\\', '/')
    if os.path.isabs(output_path):
        output_path = output_path.replace('\\', '/')
    else:
        output_path = os.path.abspath(output_path).replace('\\', '/')
    if not os.path.isabs(logging_file):
        logging_file = os.path.join(output_path, 'motopy.log')

    fmt = '%(asctime)s [%(levelname)s] : %(message)s'
    logging.basicConfig(filename=logging_file, filemode='w', 
        level=logging_level, format=fmt)
    global console_handler
    if console_handler is None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        formatter = logging.Formatter(fmt)
        console_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(console_handler)

    _locals = {}
    _globals = _locals

    old_path = os.path.abspath(os.curdir)
    os.chdir(output_path)
    # exec(f'os.chdir("{output_path}")', _globals, _locals)

    translate_file(entry_basename, input_path, output_path, 
        replaced_functions=replaced_functions,
        indent=indent, logging_level=logging_level, 
        _globals=_globals, _locals=_locals, **kwargs)

    os.chdir(old_path)
