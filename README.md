# FDFDRadiowaveSimulator
Finite-difference frequency domain radiowave simulator - based on Helmholtz equation

GitHub repository accompanying the paper A. Oustry, M. Le Tilly, T. Clausen, C. D'Ambrosio, L. Liberti, "Optimal deployment of indoor wireless local area networks", Submitted.



## Dependencies
The simulator requires Python3, with standard modules (numpy, scipy, PIL, pandas, matplotlib).

## Data and numerical experiments

The make-believe building maps used in the aforementioned article are .png located in the folder ''sources''. To run the simulations on these instances, execute the command
```
python3 examples.py
```
The simulation outputs are cloned in the folder github.com/aoustry/Odewine/MAPLib, where they are used for WLAN deployment optimization.

------------------------------------------------------------------------------------------

Researchers affiliated with

(o) LIX CNRS, École polytechnique, Institut Polytechnique de Paris, 91128, Palaiseau, France

(o) École des Ponts, 77455 Marne-La-Vallée

Developed as part of the ODEWINE project sponsored by the Cisco Research Foundation
