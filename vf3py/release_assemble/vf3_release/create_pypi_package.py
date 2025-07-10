import glob
import os
import string
import shutil
import inspect
import tarfile
from typing import List, Tuple, Callable, Union, Any


TARGET_SDIST_DIR = './dist' # WARNING This directory will be cleaned up, if exists
TEMPLATE_SDIST_DIR = './sdist_template'
README_PATH = '../../README.md'
PROJECT_URLS = {
    'Gitlab home page': 'https://gitlab.com/knvvv/vf3py',
    'Docs': 'https://knvvv.gitlab.io/vf3py',
}

PACKAGE_DATA = {
    'package': 'vf3py', # Will appear as "pip install {package}"
    'module': 'vf3py', # Will appear as "import {module}"
    'version': '1.0.1',
    'author': 'Nikolai Krivoshchapov',
    'platforms': ['Linux'], # , 'Windows'
    'python_versions': '>=3.8.0',
    'install_requires': ['numpy', 'networkx'],
    'project_summary': 'Interfacing Python and NetworkX with VF3 – the fastest algorithm for graph/subgraph isomorphism calculation',
    'project_urls': '\n'.join([
        f"Project-URL: {description}, {url}"
        for description, url in PROJECT_URLS.items()
    ]),
    'readme_content': "Description-Content-Type: text/markdown\n\n" + open(README_PATH, 'r').read(),
    'files': ''
}

PACKAGE_DATA['install_requires_lines'] = "\n".join(PACKAGE_DATA['install_requires'])
PACKAGE_DATA['platforms_lines'] = "\n".join([
    f"Platform: {platform}"
    for platform in PACKAGE_DATA['platforms']
])


PULL_FILES = {
    # relpath of dirs and files in cwd => relpath in sdist
    # wildcards (*) are allowed on the left side
    # {version} etc. are allowed on both sides 
    './setup_pypi.py': './{package}-{version}/setup.py',
    './vf3py': './{package}-{version}/vf3py',
    './*.so': './{package}-{version}/vf3py',
    # './*.pyd': './{package}-{version}/vf3py',
    # './win_dlldeps.json': './{package}-{version}/win_dlldeps.json',
    # './win_dlls': './{package}-{version}/win_dlls',
}


def insertion_at_beginning(item: Union[str, Any], use_subs=True, include_linebreak=True) -> Callable:
    if not isinstance(item, str):
        item = inspect.getsource(item)
    def insertion_at_beginning_task(process_path: str, template_substitution: Callable=None, **unused_kwargs):
        if template_substitution is not None and use_subs:
            add_code = template_substitution(item)
        else:
            add_code = item
        
        if include_linebreak:
            add_code += '\n'

        with open(process_path, 'r') as f:
            all_lines = f.readlines()
        all_lines.insert(0, add_code)
        with open(process_path, 'w') as f:
            f.write(''.join(all_lines))
    return insertion_at_beginning_task


def archive_directory(archive_name: str) -> Callable:
    def archive_directory_task(process_path: str, target_abs: str, template_substitution: Callable=None, **unused_kwargs):
        all_target_files, _ = recurse_directory_contents(process_path)
        archive_path = os.path.join(target_abs, template_substitution(archive_name))
        with tarfile.open(archive_path, 'w:gz') as archive:
            for file_name in all_target_files:
                archive.add(file_name, os.path.relpath(file_name, target_abs))
    return archive_directory_task


def remove_path(strict=True) -> Callable:
    def remove_path_task(process_path: str, target_abs: str, **unused_kwargs):
        if strict:
            assert os.path.exists(process_path), \
                f"Path '{process_path}' does not exist - can not delete"
    
        if os.path.isdir(process_path):
            shutil.rmtree(process_path)
        elif os.path.isfile(process_path):
            os.remove(process_path)
    return remove_path_task


POSTPROCESSING_TASKS = [
    ('./{package}-{version}/setup.py', insertion_at_beginning(f"PACKAGE_DATA = {repr(PACKAGE_DATA)}", use_subs=False)),
    ('./{package}-{version}/vf3py/__init__.py', insertion_at_beginning(f"__version__ = '{PACKAGE_DATA['version']}'")),
    # ('./{package}-{version}/vf3py/*.dll', remove_path(strict=False)),
    ('./{package}-{version}/vf3py/%s_base.so' % PACKAGE_DATA['module'], remove_path(strict=False)),
    ('./{package}-{version}/vf3py/%s_vf3l.so' % PACKAGE_DATA['module'], remove_path(strict=False)),
    ('./{package}-{version}/vf3py/%s_vf3p.so' % PACKAGE_DATA['module'], remove_path(strict=False)),
    ('./{package}-{version}', archive_directory('./{package}-{version}.tar.gz')),
    ('./{package}-{version}', remove_path(strict=True)),
]


def keys_as_list(mask: str) -> list:
    return [
        t[1]
        for t in string.Formatter().parse(mask)
        if t[1] is not None
    ]


def create_substitution_function(subs: dict) -> Callable[[str], str]:
    def subs_function(text: str) -> str:
        return text.format(**{
            key: subs[key]
            for key in keys_as_list(text)
        })
    return subs_function


def assert_no_mask(text: str) -> None:
    assert len(keys_as_list(text)) == 0, f"This should NOT contain any keys: '{text}'"


def prepare_directory(dirname: str) -> None:
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def recurse_directory_contents(directory: str) -> Tuple[List[str], List[str]]:
    all_files = []
    all_dirs = []
    
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                all_files.append(entry.path)
            elif entry.is_dir():
                cur_files, cur_dirs = recurse_directory_contents(entry.path)
                all_dirs.append(entry.path)
                all_dirs.extend(cur_dirs)
                all_files.extend(cur_files)

    return all_files, all_dirs


if __name__ == "__main__":
    prepare_directory(TARGET_SDIST_DIR)

    target_abs = os.path.abspath(TARGET_SDIST_DIR)
    template_abs = os.path.abspath(TEMPLATE_SDIST_DIR)
    assert os.path.isdir(template_abs), \
        f"Directory '{template_abs}' does not exist"
    assert_no_mask(target_abs)
    assert_no_mask(template_abs)
    assert os.path.dirname(target_abs) == os.path.dirname(template_abs), \
        f"These dirs are expected to be in a common directory: '{target_abs}' and '{template_abs}'"
    
    # Map 'template' strings => 'target' strings
    template_substitution = create_substitution_function(PACKAGE_DATA)

    template_files, template_dirs = recurse_directory_contents(template_abs)
    
    for template_directory in template_dirs:
        # Substitution + translation from template to target directory
        target_directory = os.path.relpath(template_substitution(template_directory), template_abs)
        target_directory = os.path.join(target_abs, target_directory)
        
        os.mkdir(target_directory)
        print(f"Created directory '{target_directory}'")
    
    for template_filename in template_files:
        # Substitution + translation from template to target directory
        target_filename = os.path.relpath(template_substitution(template_filename), template_abs)
        target_filename = os.path.join(target_abs, target_filename)

        with open(template_filename, 'r') as f:
            template_contents = f.read()
        target_contents = template_substitution(template_contents)
        with open(target_filename, 'w') as f:
            f.write(target_contents)
        print(f"Created '{target_filename}'")
    
    for current_mask, target_mask in PULL_FILES.items():
        current_wildcard = template_substitution(current_mask)
        # 'current_wildcard' might contain some *
        current_wildcard = os.path.join(os.getcwd(), current_wildcard)
        current_paths = glob.glob(current_wildcard)
        assert len(current_paths) > 0, \
            f"No files or dirs found for '{current_mask}'"
        
        target_path = template_substitution(target_mask)
        # No wildcards are allowed for target
        assert '*' not in target_path, \
            f"Target mask '{target_path}' is not allowed to be a wildcard"
        target_path = os.path.join(target_abs, target_path)
        assert os.path.isdir(os.path.dirname(target_path)), \
            f"Containing directory of '{target_path}' does not exist"
        
        if len(current_paths) > 1:
            assert os.path.isdir(target_path), \
                f"Multiple files are being copied '{current_paths}' but containing dir '{target_path}' does not exist"
        for current_path in current_paths:
            print(f"Copying '{os.path.relpath(current_path, os.getcwd())}' => '{os.path.relpath(target_path, os.getcwd())}'")
            if os.path.isfile(current_path):
                shutil.copy2(current_path, target_path)
            elif os.path.isdir(current_path):
                shutil.copytree(current_path, target_path)
            else:
                raise Exception(f"Unable to copy '{current_path}'")

    for target_mask, processing_call in POSTPROCESSING_TASKS:
        target_expression = template_substitution(target_mask)
        target_expression = os.path.join(target_abs, target_expression)
        target_paths = glob.glob(target_expression)
        for target_path in target_paths:
            processing_call(process_path=target_path, template_substitution=template_substitution, target_abs=target_abs)
        print(f"Completed postprocessing of '{os.path.relpath(target_expression, os.getcwd())}'")
    
    print("""
Now you can push the new release to PyPi with
twine upload dist/*
""")
