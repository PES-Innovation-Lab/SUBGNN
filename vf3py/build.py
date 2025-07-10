import subprocess
import os
import shutil
import glob
import platform
import sys
import copy
import re

NPROC = 6
SHAREDOBJ_MASKS = {
    'Windows': ["vf3py_base.cp{pyver}*.pyd", "vf3py_vf3l.cp{pyver}*.pyd", "vf3py_vf3p.cp{pyver}*.pyd"],
    'Linux': ["vf3py_base.cpython-{pyver}*.so", "vf3py_vf3l.cpython-{pyver}*.so", "vf3py_vf3p.cpython-{pyver}*.so"],
}
assert platform.system() in ['Windows', 'Linux'], f"Running on an unsupported platform ({platform.system()})"
SHAREDOBJ_MASK = SHAREDOBJ_MASKS[platform.system()]

def is_int(x):
    try:
        int(x)
        return True
    except:
        return False
    

class FlagParser:
    def __init__(self, argv_flags):
        self.parsed_flags = []
        self.unparsed_flags = [flag for flag in argv_flags if flag.startswith('-')]
        self.all_flags = copy.deepcopy(argv_flags)

        self.ints_assignment = {}
        for i, flag in enumerate(argv_flags):
            if not flag.startswith('-'):
                assert argv_flags[i-1].startswith('-') and is_int(flag), f"Problem with flag {flag}"
                self.ints_assignment[argv_flags[i - 1]] = int(flag)

    def __contains__(self, flag_name):
        assert flag_name not in self.parsed_flags
        res = flag_name in self.unparsed_flags
        if res:
            self.parsed_flags.append(flag_name)
            del self.unparsed_flags[self.unparsed_flags.index(flag_name)]
        return res

    def __getitem__(self, flag_name):
        return self.ints_assignment[flag_name]

    def parsed_all_flags(self):
        return len(self.unparsed_flags) == 0

def runcmd(cmd):
    # Use regular expression to split by spaces while preserving quotes
    parts = [x for x in re.findall(r'(?:[^\s"]*(?:"[^"]*")?)+', cmd) if len(x)> 0]
    subprocess.call(parts)


def build_vf3py(argv_flags):
    flags = []
    # flags.append(f"-DBUILDFLAGS=\"{' '.join(argv_flags)}\"")
    argv_flags = FlagParser(argv_flags)

    # if '-l' in argv_flags:
    #     flags.append("-DUNITTEST=1")
    #     flags.append("-DDEBUG=1")
        
    assert argv_flags.parsed_all_flags()

    runcmd("rm -Rf build/")
    runcmd("mkdir build")

    # runcmd("conan install . --output-folder=build --build=missing --profile=debug")
    runcmd("conan install . --output-folder=build --build=missing")
    
    mainwd = os.getcwd()
    os.chdir(os.path.join(mainwd, "build"))
    # runcmd("cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug")
    runcmd("cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release")
    runcmd("cmake --build . {flags}".format(flags=" ".join(flags)))

    os.chdir(mainwd)
    for somask in SHAREDOBJ_MASK:
        sofile = glob.glob(os.path.join('./build', somask.format(pyver='')))
        assert len(sofile) == 1
        sofile = sofile[0]

        shutil.copy2(sofile, '.')

if __name__ == "__main__":
    build_vf3py(sys.argv[1:])

