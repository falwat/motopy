import os
import motopy

if __name__ == '__main__':
    input_path = './tests'
    output_path = './tests/out'
    entryfiles = [
        'array_test',
        'block_test',
        'break_test',
        'cell_test',
        # 'cmd_test',
        'continue_test',
        # 'dir_test',
        'find_test',
        'for_range_test',                                                               
        'for_test',
        'func_test',
        'func_test2',
        'func_without_end_test',
        'global_test',
        # 'if_else_test',
        'length_test',
        # 'load_test',
        'nested_func_test',
        # 'nargout_test',
        'op_test',
        'printf_test',
        'range_test',
        'return_test',
        'slice_test',
        'str_test',
        'struct_test',
        'tic_toc_test',
    ]
    for filename in entryfiles:
        motopy.make(filename, input_path, output_path, indent = 4, logging_level=motopy.DEBUG)
        basename = os.path.splitext(filename)[0]
        expect_filename = input_path + '/' + basename+'.py'
        output_filename = output_path + '/' + basename+'.py'
        with open(output_filename, 'r') as fp:
            output_text = fp.read()
        with open(expect_filename, 'r') as fp:
            expect_text = fp.read()
        assert output_text == expect_text, f'{output_filename} check failed.'

    

