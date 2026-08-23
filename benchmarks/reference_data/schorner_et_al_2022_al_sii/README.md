# Schörner et al. (2022) aluminium structure factors

This package contains two curves digitized by the Otter maintainer from
Figure 2 of Maximilian Schörner, Hannes R. Rüter, Martin French, and Ronald
Redmer, *Extending ab initio simulations for the ion-ion structure factor of
warm dense aluminum to the hydrodynamic limit using neural network
potentials*, Physical Review B **105**, 174310 (2022),
<https://doi.org/10.1103/PhysRevB.105.174310>.

Both states are equilibrium aluminium states with `Te = Ti`: `1 eV` at
`rho = 4.712 g/cc` and `5 eV` at `rho = 8.1 g/cc`. Wavenumber is in inverse
ångström and `Sii(k)` is dimensionless.
The current CSV contains 60 digitized wavenumber points for each curve.

The raw CSV is retained exactly as supplied. Only the 5 eV curve was used to
calibrate the digitization ordinate. Consequently, the stored 1 eV ordinate
is displaced downward by 1.5. The benchmark loader applies

```text
Sii_corrected(k) = Sii_stored(k) + 1.5
```

to `Al_T_1p0_rho_4p712gcc` and applies no correction to the 5 eV curve. The
manifest records this transformation so that the raw data and the displayed
physical values remain distinguishable and auditable.

No open-data license is recorded for the digitized values. They carry license
status `NOASSERTION` and are not covered by Otter's BSD-3-Clause software
license. Consult the publication and `manifest.json` before reuse.
