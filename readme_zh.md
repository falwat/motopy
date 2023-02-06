# motopy

## 介绍

`motopy` 是一款功能强大的 `Matlab`/`Octave` 转 `PYthon`工具. 在转换的过程中, 自动执行转换后的`python`语句, 保证转换过程的正确性. 例如下述`Matlab`/`Octave`代码:

```m
a = ones(1, 3);
b = a';
c = a * b;
```
将转换为:
```py
import numpy as np
a = np.ones((1, 3))
b = a.T
c = a @ b
```
变量 `a` 和 `b` 的值类型均为数组类型. 所以在转换第三条语句 `c = a * b` 时, 将会转换为: `c = a @ b`.


## 安装

使用 `pip` 安装 `motopy`.

```bash
pip install motopy
```

## 快速开始

`motopy`使用起来及其简单, 首先请准备好你的`Matlab/Octave`文件, 将脚本文件(*.m)和其调用的函数文件放置在同一个文件夹, 保证你的`Matlab/Octave`脚本能够正常运行, 并且满足[m代码预处理](#m代码预处理). 下面我将给出一个简单的例子:

- 创建名为"demo"的文件夹.
- 在"demo"文件夹下, 创建两个.m文件, 文件名分别为"func.m" 和 "func_test.m", 文件内容如下:
	```m
	% file: func.m
	function s = func(a, b)
	    s = sqrt(a.^2 + b.^2);
	end
	```

	```m
	% file: func_test.m
	a = 3;
	b = 4;
	s = func(a, b);
	disp(s)
	```
- 在"demo"文件夹中, 创建一个`python`脚本文件, 导入`motopy`, 并调用`motopy.make()`完成代码的转换. `python`脚本内容如下:
  ```py
  import motopy
  motopy.make(entry_basename='func_test')
  ```
  `entry_basename`参数用于指定要转换的m文件的脚本入口(***!!注意, 此处不加扩展名!!***).
  > 当然, 你也可以直接在`python`命令行中执行上述代码. 请保证当前目录是"demo"文件夹.

### 指定输入输出文件夹
`python`脚本文件可以不放置到m文件所在文件夹中, 输入的m文件和输出的py文件也可以位于不同的文件夹中. 此时可以使用`input_path`参数指定输入m文件所在位置, 使用`output_path`参数指定生成的`python`文件的输出路径.

```py
import motopy
motopy.make(
    entry_basename='func_test', # no extension
    input_path='输入m文件所在路径', 
    output_path='输出py文件所在路径')
```

### 指定替代函数

如果你已经完成了某个函数的转换, 可以通过`motopy.make()`函数中的 `replaced_functions`参数指定此函数的替代函数.
```py
import motopy
motopy.make(
    entry_basename='func_test', # no extension
    input_path='输入m文件所在路径', 
    output_path='输出py文件所在路径',
    replaced_functions={
        'func': ('func', 'func') # 
    }
)
```
`replaced_functions`参数为一个字典, 键为m文件中出现的函数名, 值为二元元组(`模块名`, `函数名`). 上述示例中, `func`函数文件将不会再次转换. 

什么情况下使用`replaced_functions`?

- `motopy`生成的`py`文件, 我们对其进行了修改, 并且不希望`motopy`重新生成它.
- `motopy` 暂不支持转换的函数, 你可以自己实现它.

### 在m文件中插入Python代码

m文件中以 "`%%>`" 开头的行会被插入到生成的`python`文件中.并且下一行代码中的第一条语句会被屏蔽掉.比如下述m代码:
```m
%%> print('this is a code annotation.')
disp('this statment will be skiped.')
```
将会生成如下`python`代码:
```py
print('this is a code annotation.')
```

### 输出日志
默认在`output_path`文件夹下生成名为"motopy.log"日志文件. 可以通过`logging_file`参数, 指定日志文件的输出位置和名称. 使用`logging_level`设置日志等级: `WARN|INFO|DEBUG`

```py
import motopy
motopy.make(.., logging_level=motopy.DEBUG, ..)
```

### 缩进

默认生成的py文件使用4个空格进行缩进, 可以通过`indent`参数指定缩进所需的空格数.

## m代码预处理

因为`motopy`采用边转译边执行的方式进行转换. 所以转换可能失败. 为了提高转换的成功率. 请对你的".m"代码进行适当修改.

- 数组和元胞中的元素使用","分隔, 而不是空格. 下述代码是一个错误示例:

    ```m
    a = [1 2 3; 4 5 6]; % 不要使用空格分隔元素
    c = {1 2 'abc'};
    ```
- 函数文件的名称和函数名必须相同.
- 数组和元胞应该事先定义, 且分配足够大的空间. 下述代码是一个错误示例:

    ```m
    for k=1:5
        A(k) = 2*k; % A 在赋值之前没有定义.
    end
    ```

    ```m
    A = []; % 虽然定义了A, 但是没有给足够大的空间
    for k=1:5
        A(k) = 2*k; % A的大小在迭代的过程中改变了
    end
    ```
- 不要使用"`[]`"定义空数组. 下述代码将转换失败:
    ```m
    A = [];
    for k=1:5
        B= rand(2,2);
        A = [A;B];
    end
    disp(A)
    ```
    转换上述代码时, 转换到表达式`[A;B]`时, 转换失败. 因为`numpy`中, `0x0`的空数组`A` 无法与`2x2`的数组B进行拼接.

    一种简单的处理方法是将数组`A` 定义为 `0x2` 的空数组:

    ```m
    A = zeros(0,2);
    for k=1:5
        B= rand(2,2);
        A = [A;B];
    end
    disp(A)
    ```


## 已实现的转换

### 矩阵, 数组和元胞的创建

Matlab/Octave|Python|Note
:-|:-|:-
`a = [1,2,3,4]` | `a = np.array([1, 2, 3, 4])` | matlab中的数组会被转换为`np.array`
`a = [1,2;3,4]` | `a = np.array([[1, 2], [3, 4]])`
`a = [1;2;3;4]` | `a = np.array([[1], [2], [3], [4]])`
`C = {1,2,3;'4',[5,6],{7,8,9}}` | `C = [[1, 2, 3], ['4', np.array([5, 6]), [7, 8, 9]]]` | `matlab` 的 `cell` 会被转换成 `python`中的`list`
`r1 = 1:10;` | `r1 = arange(1, 11)` | 上限自动+1
`N = 10;`<br>`r2 = 1:N;` | `N = 10`<br>`r2 = arange(1, N + 1)`
`zeros(3)` | `np.zeros((3, 3))`
`zeros(2,3)` | `np.zeros((2, 3))`
`ones(3)` | `np.ones((3, 3))`
`ones((2, 3))` | `np.ones((2, 3))`
`C = cell(2,3)` | `C = [[None for _c in range(3)] for _r in range(2)]`


### 矩阵, 数组和元胞的切片
Matlab/Octave|Python|Note
:-|:-|:-
`a(1,1)` | `a[0:1, 0:1]`
`a(1,:)` | `a[0:1, :]`
`a(:,1)` | `a[:, 0:1]`
`a(1, 1:2)` | `a[0:1, 0:2]`
`a(1:2, 1)` | `a[0:2, 0:1]`
`a(1,2:end)` | `a[0:1, 1:]`
`m = 1;`<br>`n = 1;`<br>`a(m, n*2)` | `m = 1`<br>`n = 1`<br>`a[m - 1:m, n * 2 - 1:n * 2]`

### 函数
Matlab/Octave|Python|Note
:-|:-|:-
`abs` | `np.abs`
`acos` | `np.arccos`
`asin` | `np.arcsin`
`atan` | `np.arctan`
`[y,Fs] = audioread(filename)` | `Fs, y = wavfile.read(filename)`
`ceil` | `np.ceil`
`cos` | `np.cos`
`diag` | `np.diag`
`d = dir(name)` | `d = [{'name':e.name, 'folder':e.path, 'isdir':e.is_dir()} for e in scandir(name)]`
`disp` | `print`
`eye` | `np.eye`
`exp` | `np.exp`
`fft` | `np.fft`
`find` | `np.nonzero`
`fix` | `np.fix`
`floor` | `np.floor`
`fprintf` | 
`ifft` | `np.ifft`
`inv` | `linalg.inv`
`length(a)` | `max(np.shape(a))`
`linspace` | `np.linspace`
`S = load('data.mat')`| `S = loadmat('data.mat')` | the Variable `S` is a dict 
`A = load('data.txt')` | `A = np.loadtxt('data.txt')` | the file "data.txt" is a ASCII data.
`load('data.mat')` | `_mat = loadmat('data.mat');`<br>`a = _mat['a'];`<br>`b = _mat['b']` | assume there are two variable `a` and `b` in "data.mat"
`load('data.txt')` | `data = np.loadtxt('data.txt')` | the file "data.txt" is a ASCII data.
`log` | `np.log`
`log10` | `np.log10`
`log2` | `np.log2`
`mod` | `np.mod`
`ndims` | `np.ndim`
`num2str` | `str`
`numel` | `np.size`
`pinv` | `linalg.pinv`
`rand` | `random.rand`
`rank` | `linalg.matrix_rank`
`round` | `np.round`
`sin` | `np.sin`
`sort` | `np.sort`
`sprintf('%d%s',a, b)` | `f'{a}{b}'`
`sqrt` | `np.sqrt`
`s = strcat(s1,...,sN)` | `s = ''.join([s1,...,sN])`
`unique` | `np.unique`

## 修改日志

查看 [changelog.md](https://github.com/falwat/motopy/blob/main/changelog.md) 获取更多信息.

## 说明
`Motopy` 目前处于开发阶段, 如果你在使用的过程中发现任何关于`motopy`的问题, 烦请提交至[Issues](https://github.com/falwat/motopy/issues), 或邮件告知(falwat@163.com), 留言亦可. 我将在后续版本中更新修复. 感谢使用`motopy`. 

 