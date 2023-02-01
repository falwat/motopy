def func():
    global a
    a = a + 1
    print(a)

global a
a = 10
func()

