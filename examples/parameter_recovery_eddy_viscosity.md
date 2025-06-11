# Notes for equation discovery

The file `parameter_recovery_eddy_viscosity.ipynb` demonstrates an attempt to discover an SGS model coefficient using the "on-the-fly" algorithm.
Most of the functions relevant specifically to updating the parameter are found in `parameter_recovery_eddy_viscosity.py` and are used in this demo notebook.

The file `plot.py` may also be useful for visualizing the high and low resolution states and their difference which is used in nudging and in parameter updates.

## Future

- We've discussed using other metrics for error.
  See, for example, `examples/spectrum.ipynb`.
