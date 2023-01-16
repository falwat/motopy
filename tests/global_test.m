global a
a = 10;
func();

function func()
    global a
    a = a + 1
    disp(a)
end