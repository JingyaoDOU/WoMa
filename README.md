WoMa
======

A python solver of the differential equations of a planet in hydrostatic
equilibrium for rotating and non-rotating cases.

It includes SEAGen (Kegerreis et al. 2019) and a new algorithm
to suit initial conditions for spining planets in SPH simulations.

Sergio Ruiz-Bonilla: sergio.ruiz-bonilla@durham.ac.uk  

This program has been tested for a wide range of cases but not exhaustively. If
you find any bugs, potential improvements, or features worth adding, then please
let us know!


Contents
--------
+ `woma.py` The main program classes and functions.
+ `seagen.py` The main program classes and functions of the SEAGen project.
+ `examples.py` Examples to demonstrate how to use the WoMa module.
+ `eos.py` Functions related equations of states.
+ `LICENSE.txt` GNU general public license v3+.


Basic Usage
-----------
+ See the doc strings in `woma.py` and `eos.py` for all the details.
+ See `examples.py` for full working examples.


Requirements
------------
+ Python 3 (tested with 3.6.0).


Notation etc.
-------------
+ PEP8 is followed in most cases apart from some indentation alignment.
+ Arrays are explicitly labelled with a prefix `A1_`, or `An_` for an
    `n`-dimensional array.
+ Particle is abbreviated to `picle`.


To Do etc. (Jacob thoughts in progress)
---------------------------------------
+ Separate into more files e.g. profiles and spinning, utilities (with file doc strings!)
+ Change the iterations to be for some tolerance not a number of iterations
+ Move the add-L3 function out of the class, like the others
+ Add documentation to examples and basic instructions to readme
+ Make into a PyPI package
+ Tidy line lengths for where I've messed things up
+ Add all output variables to the hdf5 file
+ Automatically choose how to gen profiles depending on which inputs are given
+ See `###` and `...` notes throughout the code
+ Check that MoI is the reduced MoI everywhere (or change name)
+ Add doc strings and prints to the examples, explain what they're trying to do
    and how, so that someone can copy them and understand.
+ Make the examples faster to run. Maybe start close to the right answer and 
    do fewer iterations.
