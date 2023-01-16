import motopy

motopy.make(
    entry_basename='func_test2', # no extension
    input_path='./tests', 
    output_path='./tests', 
    logging_level=motopy.DEBUG,
    replaced_functions={
        # 'func': ('func', 'func')
    }
)