This is a set of analysis tools to be used in CMB research. It currently consists of code to calculate the pure Cl power spectra from a HEALPix map along with the associated mode-mixing matrices. In addition it contains a port of polspice to Python. Finally, we have added code to calculate the COBE Quadcube pixel number given an input position on the sky.

## Requirements ##

There are several Python modules needed in order to run this code. They are

* Numpy
* Scipy
* Astropy
* Healpy
* Mpi4py

In addition, the location of the cmb_analysis directory needs to be added to your PYTHONPATH environment variable.

## Contributing ##

We welcome any outside contributions to this code, However, in order for us to accept any pull requests, the contributor must fill out and return the contributor license agreements found [here](https://lambda.gsfc.nasa.gov/data/cla/Ind_CLA_final_GSC-17661-1_CMB_Analysis_Software.pdf "CLA for individuals") for individuals and [here](https://lambda.gsfc.nasa.gov/data/cla/Corp_CLA_final_GSC-17661-1_CMB_Analysis_Software.pdf "CLA for organizations") for organizations.

## Note ##

The visualization code for the COBE Quadcube pixelization inherits from the projection axes used in Healpy. A better implementation of the visualization would just copy the Healpy mollview code and replace the HpxZZZ axes in the Healpy code with the QcZZZ axes implemented in this code, but given the licensing and copyright requirements that are on this repository due to going through the NASA official release process, we didn't think it would be right to copy the code so a basic/bare version of mollview is implemented here.

## Contact ##

Nathan Miller - nathan.j.miller@nasa.gov

Copyright © 2016 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.
