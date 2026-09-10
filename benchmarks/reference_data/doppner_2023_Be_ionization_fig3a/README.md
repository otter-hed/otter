# Döppner et al. (2023), Be ionization, Fig. 3(a)

Source: T. Döppner et al., *Observing the onset of pressure-driven K-shell
delocalization*, Nature **618**, 270–275 (2023).
[Article and DOI](https://doi.org/10.1038/s41586-023-05996-8).

The CSV contains digitized model curves from Fig. 3(a).
Its first row names nine series; its second row gives alternating `rho,Zbar`
columns. Density is in g/cm³; ionization is dimensionless (electrons per Be).
Empty trailing pairs are missing observations, not zero. Each series is loaded
independently, without extrapolation or smoothing.

- DFT-MD: 150 eV (7 points), 100 eV (9), 50 eV (9).
- Stewart–Pyatt (`SP`): **160**, 100, 50 eV (40 points each).
- OPAL: **160**, 100, 50 eV (40 points each).

The 160 eV reference curves must not be relabelled 150 eV. Ionization
definitions depend on the electron partition used by each model.

License status: `NOASSERTION`. The source article is copyrighted; Otter's code
license does not apply to digitized publication data. Attribution is retained.
