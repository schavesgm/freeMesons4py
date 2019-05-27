# freeMesons4py

**freeMesons4py** is a code that allows you to calculate
mesonic correlations functions on the lattice and
restricted to the free theory. It allows you to choose
different boundary conditons in the time direction,
such as periodic, open/Dirichlet or infinite time extent.
Furthermore, it allows you to run simulations with different
regularizations of the fermionic action, such as Wilson-Dirac
or Wilson Twisted Mass (WTM).

The code is written in Python and it is parallelized using
_mpi4py_. In order to be able to run it you need:

* Python 3.6 or 2.7 (It should work in both )
* **Numpy** package - pip install numpy
* A **MPI** implementation, such as **openMPI**
* The module **mpi4py* - pip install mpi4py

Inside the file _input.in_ you can manipulate the different
properties of you simulation. It should be autoconsistent
so you can understand what you are simulating by reading it.

The code is free to use. Any question just contact the author:

sergiozteskate@gmail.com

The code was written as part of a masters dissertation in
Lattice QCD by Sergio Chaves Garcia-Mascaraque.
