limit = 0.8;
s = 0;

while 1
    tmp = rand(1);
    disp(tmp)
    if tmp > limit
        break
    end
    s = s + tmp;
end