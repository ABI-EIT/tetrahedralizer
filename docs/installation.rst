.. highlight:: shell

============
Installation
============


Stable release
--------------

To install the Tetrahedralizer python package, run this command in your terminal:

.. code-block:: console

    $ pip install git+https://github.com/ABI-EIT/tetrahedralizer

This is the preferred method to install Tetrahedralizer, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

GUI App
-------
The GUI app can be installed using a Pyinstaller script build into the Tetrahedralizer package.
Run the following command in your terminal:

.. code-block:: console

    $ python -m tetrahedralizer --install tetrahedralizer_qt

This will create a folder containing the installation of the Tetrahedralizer GUI
app in your current directory. Run the app using tetrahedralizer_qt.exe contained within this folder.

From sources
------------

The sources for Tetrahedralizer can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/ABI-EIT/tetrahedralizer

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/ABI-EIT/tetrahedralizer/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/ABI-EIT/tetrahedralizer/issues
.. _tarball: https://github.com/ABI-EIT/tetrahedralizer/tarball/master
