# Data

The simulation data are not stored in this repository. 

Expected layout (paths are set in `configs/*.yaml`; the root is `data/` or `$RHEOPT_DATA`):


## File format

One `.npz` per simulation:

| key | shape | content |
|---|---|---|
| `pos` | (n, 2) | node coordinates |
| `vel` | (T, n, 2) | velocity: m/s for the non-Newtonian files, cm/s for the Newtonian contraction files; the loader converts to cm/s |
| `Sigma_xy`, `Sigma_xx`, `Sigma_yy` | (T, n) | total stress components |
| `Re` | scalar | Reynolds number |
| `Wi` or `Bi` | scalar | Weissenberg number (Giesekus, Oldroyd-B) or Bingham number; absent for Newtonian files |
| `u_in` (or `u_0`, `u0`) | scalar | inlet velocity |

The loader keeps every second stored frame. Model channels: vx, vy, σxy, σxx, σyy.
