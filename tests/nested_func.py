def nested_func():
    def f1():
        print('f1')
    def f2():
        def f21():
            print('f2.1')
        print('f2')
        f21()
        f3()
    f1()
    f2()

def f3():
    print('f3')

