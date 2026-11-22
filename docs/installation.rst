Installation
==============

The ``pygwtf`` package can be installed using pip. To install the latest version from PyPI:

.. code-block:: console

   $ python -m pip install pygwtf

``pygwtf`` contains compiled extensions, so installing from source needs a C++17 compiler (CMake, nanobind and
scikit-build-core are fetched automatically by pip). The GPU backend is built when a CUDA (``nvcc``) or HIP
(``hipcc``) compiler is found; select it explicitly with the ``GWTF_GPU`` environment variable
(``CUDA``, ``HIP``, ``OFF``). See :doc:`v2_backend` for details.

For GPU usage you must also install the ``cupy`` build for your platform (e.g. ``cupy-cuda12x``, ``cupy-cuda13x``
or ``cupy-rocm-*``), which is not included in the standard installation requirements.

From source
-------------

To install from source for development, clone the repository and perform an editable installation:

.. code-block:: console

   $ git clone https://github.com/cchapmanbird/gwtf.git
   $ cd gwtf
   $ python -m pip install -e ".[dev]"

Editable installs compile the extensions once; after changing C++ sources, re-run the install command (the build
directory is reused, so this is incremental). Run the tests with ``pytest tests``.

Building documentation has further requirements, sourcing from the ``docs/requirements.txt`` file:

.. code-block:: console

   $ python -m pip install -r docs/requirements.txt

Note that this will not install ``pandoc``. This is most easily installed via ``conda`` (for some reason, ``pip`` does not work).
