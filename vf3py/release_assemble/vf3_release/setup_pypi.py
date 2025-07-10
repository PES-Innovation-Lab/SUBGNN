import platform
import glob
import ntpath
import os
import json
import shutil
from setuptools import setup

# OS specifics
CUR_OS = platform.system()
SHAREDOBJ_TEMPLATE = {
    'Windows': ["vf3py_base.cp{py_ver}-win_amd64.pyd", "vf3py_vf3l.cp{py_ver}-win_amd64.pyd", "vf3py_vf3p.cp{py_ver}-win_amd64.pyd"],
    'Linux': ["vf3py_base.cpython-{py_ver}*-x86_64-linux-gnu.so", "vf3py_vf3l.cpython-{py_ver}*-x86_64-linux-gnu.so", "vf3py_vf3p.cpython-{py_ver}*-x86_64-linux-gnu.so"],
}

assert CUR_OS in ['Linux', 'Windows'], "Only Linux and Windows platforms are supported"

if CUR_OS == 'Windows':
    DLLDEPS_JSON = 'win_dlldeps.json'
    DLL_STORAGE_DIR = 'win_dlls'
    assert os.path.isfile(DLLDEPS_JSON), f'Required file "{DLLDEPS_JSON}" not found'

# Python version specifics
python_version_tuple = platform.python_version_tuple()
py_ver = int(f"{python_version_tuple[0]}{python_version_tuple[1]}")

object_names = []
for somask in SHAREDOBJ_TEMPLATE[CUR_OS]:
    so_list = glob.glob(os.path.join('./vf3py', somask.format(py_ver=py_ver)))
    assert len(so_list) == 1
    object_names.append(ntpath.basename(so_list[0]))

for file in glob.glob('./vf3py/*.pyd') + glob.glob('./vf3py/*.so'):
    if ntpath.basename(file) not in object_names:
        os.remove(file)

if CUR_OS == 'Windows':
    assert os.path.isfile(DLLDEPS_JSON), f'Required file "{DLLDEPS_JSON}" not found'

    with open(DLLDEPS_JSON, 'r') as f:
        dlldeps_data = json.load(f)
    
    for object_name in object_names:
        assert object_name in dlldeps_data, f"'{object_name}' is not accounted in {DLLDEPS_JSON}"
        for file in dlldeps_data[object_name]:
            shutil.copy2(os.path.join(DLL_STORAGE_DIR, file), './vf3py')
    
    ADDITIONAL_FILES = ['*.dll']

elif CUR_OS == 'Linux':
    ADDITIONAL_FILES = []

setup(
    name=PACKAGE_DATA['package'],
    version=PACKAGE_DATA['version'],
    author=PACKAGE_DATA['author'],
    python_requires=f'=={python_version_tuple[0]}.{python_version_tuple[1]}.*',
    install_requires=PACKAGE_DATA['install_requires'],
    platforms=PACKAGE_DATA['platforms'],
    packages=[PACKAGE_DATA['package']],
    package_data={'vf3py': ['__init__.py', *object_names, 'cpppart/*.py*', 'test/*.py', 'test/mols/*.sdf', *ADDITIONAL_FILES]}
)
