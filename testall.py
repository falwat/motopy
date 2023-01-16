import os
import motopy

if __name__ == '__main__':
    input_path = './tests'
    output_path = './tests'
    entryfiles = [
        'array_test',
        # 'cell_test',
        'cmd_test',
        # 'for_range_test',
        # 'for_test',
        # 'func_test',
        # 'func_test2',
        # 'global_test',
        # 'if_else_test',
        # 'load_test',
        # 'op_test',
        # 'range_test',
        # 'slice_test',
        # 'test',
        # 'test2',
    ]
    for filename in entryfiles:
        motopy.make(filename, input_path, output_path, indent = 4)
        basename = os.path.splitext(filename)[0]
        expect_filename = output_path + '/' + basename+'_expect.py'
        output_filename = output_path + '/' + basename+'.py'
        with open(output_filename, 'r') as fp:
            output_text = fp.read()
        with open(expect_filename, 'r') as fp:
            expect_text = fp.read()
        assert output_text == expect_text, f'{output_filename} check failed.'

    

