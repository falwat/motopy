function nested_func()

f1()
f2()
    function f1()
        disp('f1')
    end

    function f2()
        disp('f2')
        f21()
        f3()
        function f21()
            disp('f2.1')
        end
    end
end

function f3()
disp('f3');
end