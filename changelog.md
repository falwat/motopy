# 2.1.0
- Add code annotation("code annotation" is python statment in mfile start with "%%>") the code annotation will insert to pyfile and mask next matlab/octave stament.
- Add `port.py` and `utils.py`
- Fix known bugs.


# 2.0.0

- Translate support for `while` block.
- Translate support for "struct" Creation: `s.m = val`.
- Translate support for functions: `abs`, `audioread`, `cell`, `dir`, `find`, 
  `length`, `mod`, `num2str`, `rand`, `strcat`, `tic`, `toc`.
- `for k = int_val0:int_val1` will translate to: `for k in range(int_val0, int_val1+1)`
- `[str1, str2, ..]` will translate to `''.join([str1, str2, ..])`

# 1.0.0

- This is original version.
- Translate support for `function` definition.
- Translate support for `for`, `if..elseif..else` block.
- Translate support for `start_val:end_val`, `start_val:step_val:end_val`
- Translate support for array creation: `[..]`, `ones()`, `zeros()`
- Translate support for array copy: `B = A` --> `B = A.copy()`
- Translate support for format text.
- Translate support for functions: `acos`, `asin`, `atan`, `ceil`, `cos`, 
  `diag`, `disp`, `eye`, `exp`, `fft`, `fix`, `floor`, `fprintf`, `ifft`, 
  `inv`, `linspace`, `load`, `log`, `log10`, `log2`, `max`, `ndims`, `numel`, 
  `ones`,  `pinv`, `rank`, `round`, `sin`, `size`, `sort`, `sprintf`, 
  `sqrt`, `unique`, `zeros`.