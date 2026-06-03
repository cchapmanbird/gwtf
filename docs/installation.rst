Installation
==============

The ``pygwtf`` package can be installed using pip. To install the latest version from PyPI:

.. code-block:: console

   $ python -m pip install pygwtf

Note that for GPU usage, you must also install the ``numba-cuda`` and ``cupy`` packages, which are not included
in the standard installation requirements definition.

From source
-------------

To install from source for development, clone the repository and perform an editable installation:

.. code-block:: console

   $ git clone https://github.com/cchapmanbird/gwtf.git
   $ cd gwtf
   $ python -m pip install -e ".[dev]"

Building documentation has further requirements, sourcing from the ``docs/requirements.txt`` file:

.. code-block:: console

   $ python -m pip install -r docs/requirements.txt

Note that this will not install ``pandoc``. This is most easily installed via ``conda`` (for some reason, ``pip`` does not work).
