import os
import errno


def create_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    return filename


def list_files(inp_path, out_path, file_names = None):
    for inp_file in os.listdir(inp_path):
        if file_names is None:
            if os.path.isfile(os.path.join(inp_path, inp_file)) :
                yield (os.path.join(inp_path, inp_file),
                        os.path.join(out_path, inp_file))
        elif inp_file in file_names:
            if os.path.isfile(os.path.join(inp_path, inp_file)) :
                yield (os.path.join(inp_path, inp_file),
                       os.path.join(out_path, file_names[inp_file]))                      