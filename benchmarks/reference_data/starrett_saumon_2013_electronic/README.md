# Starrett and Saumon (2013), electronic structure and ionization

These CSV files transcribe Tables I--III of:

C. E. Starrett and D. Saumon, “Electronic and ionic structures of warm and
hot dense matter,” *Physical Review E* **87**, 013104 (2013),
[doi:10.1103/PhysRevE.87.013104](https://doi.org/10.1103/PhysRevE.87.013104).

The values are published table entries, not values digitized from curves:

- `table_i_al_levels.csv` contains the Al bound-state energies, pressure-
  ionization weights, and common broadening width at 2 and 15 eV. Blank
  published entries for the unbound 3s state at 2 eV are represented by
  `nan`, with `is_bound=0`. Energies and widths are in Hartree.
- `table_ii_al_ionization.csv` contains the Al values of
  \(Z^*=n_e^0/n_I^0\), \(\bar Z\), and the two coupling parameters.
- `table_iii_fe_hugoniot.csv` contains the corresponding Fe Hugoniot values.
  The article caption identifies the charge used by the OCP calculation as
  the TFTCP charge.

The article couples its average atom and TCP structure self-consistently; its
Appendix B ion-sphere calculation is the initial guess, not the reported final
model.  The Otter benchmark therefore presents both production IS and
experimental SC-feedback results.  Otter's feedback fixes the IS chemical
potential and is not claimed to reproduce every detail of the article's
simultaneous QTCP/TFTCP iteration.  The numerical table entries retain source
attribution and license status `NOASSERTION`; they are not covered by Otter's
BSD-3-Clause software license.
