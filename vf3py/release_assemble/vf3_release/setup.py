import platform, glob, shutil, os, json
from setuptools import setup

# OS specifics
CUR_OS = platform.system()
SHAREDOBJ_TEMPLATE = {
    'Windows': ["vf3py_base.cp{py_ver}-win_amd64.pyd", "vf3py_vf3l.cp{py_ver}-win_amd64.pyd", "vf3py_vf3p.cp{py_ver}-win_amd64.pyd"],
    'Linux': ["vf3py_base.cpython-{py_ver}*-x86_64-linux-gnu.so", "vf3py_vf3l.cpython-{py_ver}*-x86_64-linux-gnu.so", "vf3py_vf3p.cpython-{py_ver}*-x86_64-linux-gnu.so"],
}

SO_FINAL_NAMES = {
    'Linux': ['vf3py_base.so', 'vf3py_vf3l.so', 'vf3py_vf3p.so'],
    'Windows': ['vf3py_base.pyd', 'vf3py_vf3l.pyd', 'vf3py_vf3p.pyd'],
}
SO_FINAL_NAME = SO_FINAL_NAMES[CUR_OS]

assert CUR_OS in ['Linux', 'Windows'], "Only Linux and Windows platforms are supported"
if CUR_OS == 'Windows':
    DLLDEPS_JSON = 'win_dlldeps.json'
    DLL_STORAGE_DIR = 'win_dlls'

# Python version specifics
python_version_tuple = platform.python_version_tuple()
py_ver = int(f"{python_version_tuple[0]}{python_version_tuple[1]}")

VF3PYRELEASE_DIR = 'vf3py'
INSTALL_ADDITIONAL_FILES = []

# ===============================
# Actual installation starts here
# ===============================
# Find the appripriate build of vf3py shared library
for idx, (somask, so_finalfname) in enumerate(zip(SHAREDOBJ_TEMPLATE[CUR_OS], SO_FINAL_NAME)):
    vf3py_so_list = glob.glob(somask.format(py_ver=py_ver))
    assert len(vf3py_so_list) < 2, f"Several pre-built libraries were found for {so_finalfname} for your python version. This shouldn't have happend"
    assert len(vf3py_so_list) == 1, "There is no pre-built library for your version of Python"\
                                    f"(your={python_version_tuple[0]}.{python_version_tuple[1]}), "\
                                    "supported={3.7, ..., 3.11} (Linux) / {3.8, ..., 3.11} (Windows)"
    vf3py_object_name = vf3py_so_list[0]

    # Remove the library copied earlier if exists
    so_final_path = os.path.join(VF3PYRELEASE_DIR, so_finalfname)
    if os.path.exists(so_final_path):
        os.remove(so_final_path)
    # Put the build of ringo shared library into the package
    shutil.copy2(vf3py_object_name, so_final_path)

    # Copy all DLLs required for running in Windows
    if CUR_OS == 'Windows':
        assert os.path.isfile(DLLDEPS_JSON), f'Required file "{DLLDEPS_JSON}" not found'
        with open(DLLDEPS_JSON, 'r') as f:
            dlldeps_data = json.load(f)
        assert vf3py_object_name in dlldeps_data, f"'{vf3py_object_name}' is not accounted in {DLLDEPS_JSON}"
        for file in dlldeps_data[vf3py_object_name]:
            shutil.copy2(os.path.join(DLL_STORAGE_DIR, file), VF3PYRELEASE_DIR)
            INSTALL_ADDITIONAL_FILES.append(file)


setup(
    name='vf3py',
    version='0.0.1',
    author='Nikolai Krivoshchapov',
    python_requires=f'=={python_version_tuple[0]}.{python_version_tuple[1]}.*',
    install_requires=[
        'numpy',
        'networkx',
    ],
    packages=['vf3py'],
    package_data={'vf3py': ['__init__.py', *SO_FINAL_NAME, 'cpppart/*.py*', 'test/*.py', 'test/mols/*.sdf'] + INSTALL_ADDITIONAL_FILES}
)

