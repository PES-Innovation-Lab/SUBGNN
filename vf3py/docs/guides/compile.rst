Build from source
=================

This Python library uses C++ backend to perform VF3 calculations. Thus, it has to be compiled before it can be used. For most users it will suffice to use precompiled version (see :ref:`installation-label`) that is available on PyPi. But, for VF3Py developers, it is desired to modify the C++ part, compile it for Linux and Windows, and push releases.


.. _singleuser-build-label:

Build for single user
---------------------

.. code-block:: bash

    git clone --recursive https://gitlab.com/knvvv/vf3py.git
    cd vf3py
    pip install conan
    conan profile detect
    cd release_assemble
    python release_assemble.py


Build release for PyPi
----------------------

Building VF3Py release includes 3 steps:

#. Build C++ part for Linux. This will create several so-files in the ``./release_assemble/vf3_release`` directory.
#. Build C++ part for Windows. This will create several pyd-files, ``win_dlldeps.json`` in the ``./release_assemble/vf3_release`` directory and another directory ``./release_assemble/vf3_release/win_dlls``
#. Construct overall release as \*.tar.gz-package and then upload it.


+++++
Linux
+++++

First, prepare `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_/`conda <https://docs.conda.io/projects/conda/en/stable/>`_ environments:

.. code-block:: bash

    mamba create -n py38 python=3.8.19 -c defaults cmake -y
    mamba activate py38
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py39 python=3.9.19 cmake -y
    mamba activate py39
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py310 python=3.10.14 cmake -y
    mamba activate py310
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py311 python=3.11.9 cmake -y
    mamba activate py311
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py312 python=3.12.8 cmake -y
    mamba activate py312
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py313 python=3.13.1 cmake -y
    mamba activate py313
    pip install numpy twine conan networkx
    mamba deactivate

    # Verify that we're good
    mamba activate py37; python --version; mamba deactivate
    mamba activate py38; python --version; mamba deactivate
    mamba activate py39; python --version; mamba deactivate
    mamba activate py310; python --version; mamba deactivate
    mamba activate py311; python --version; mamba deactivate
    mamba activate py312; python --version; mamba deactivate
    mamba activate py313; python --version; mamba deactivate

Then, C++ part can be built for all these Python versions using the following script:

.. code-block:: bash

    cd vf3py/release_assemble
    python release_assemble.py -full

Note that ``release_assemble.py`` creates inner Bash-sessions and calls ``conda activate ...`` inside them, so your bash must start up into base conda environment.


+++++++
Windows
+++++++

NOTE: Windows is not currently supported

All building is done in `MSYS2 environment <https://www.msys2.org/wiki/MSYS2-installation/>`_. After installation, open up MSYS2 MinGW x64 and download prerequisites:

.. code-block:: bash

    pacman -Syu
    pacman -S --needed base-devel mingw-w64-x86_64-toolchain
    pacman -S mingw-w64-x86_64-gsl mingw-w64-x86_64-boost mingw-w64-x86_64-cmake mingw-w64-x86_64-pybind11 git

For MSYS2 to have access to Anaconda installed in your Windows, add the following line to ``C:\msys64\home\*myusername*\.bash_profile`` file:

.. code-block:: bash

    eval "$('/c/tools/Anaconda3/Scripts/conda.exe' 'shell.bash' 'hook')"
    # Check the path to your conda.exe (same for mamba)

Then, build VF3Py for a single user (see :ref:`singleuser-build-label`).

Create conda envs for Python \>= 3.8:

.. code-block:: bash

    mamba create -n py38 python=3.8.19 -c defaults cmake -y
    mamba activate py38
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py39 python=3.9.19 cmake -y
    mamba activate py39
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py310 python=3.10.14 cmake -y
    mamba activate py310
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py311 python=3.11.9 cmake -y
    mamba activate py311
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py312 python=3.12.8 cmake -y
    mamba activate py312
    pip install numpy twine conan networkx
    mamba deactivate

    mamba create -n py313 python=3.13.1 cmake -y
    mamba activate py313
    pip install numpy twine conan networkx
    mamba deactivate

Finally, build C++ part for all these Python versions:

.. code-block:: bash

    cd vf3py/release_assemble
    python release_assemble.py -full -dll-copy
    python release_assemble.py -full


++++++++++++++++++++++++++++++++++++
Completing and uploading the release
++++++++++++++++++++++++++++++++++++

As a result of previous two steps, these files were produced:

#. ``./release_assemble/vf3_release/*.so`` (in Linux)
#. ``./release_assemble/vf3_release/*.pyd`` (in Windows)
#. ``./release_assemble/vf3_release/win_dlls/*.dll`` (in Windows)
#. ``./release_assemble/vf3_release/win_dlldeps.json`` (in Windows)

Combine them from Linux and Windows machines in a single ``vf3_release`` directory.

To prepare and upload the package release, set the new version in the ``create_pypi_package.py`` file and do these steps:

.. code-block:: bash

    python create_pypi_package.py
    twine upload dist/*
    # For test upload:
    # twine upload --repository testpypi dist/*
    username: __token__
    password: *PyPi - API key*

The latest release of VF3Py can be found `here <https://pypi.org/project/vf3py/>`_.
