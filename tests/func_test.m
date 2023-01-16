a = 3;
b = 4;
s = func(a, b);
disp(s)
func2(a)

function s = func(a, b)
    s = sqrt(a.^2 + b.^2);
end

function func2(a)
    if a > 0
        disp('a > 0')
    else
        disp('a <= 0')
    end
end