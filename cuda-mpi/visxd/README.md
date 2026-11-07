# visxd

Quick-look visualization of ZDF simulation output.

```python
import visxd as vis          # Cartesian (1D / 2D) simulations
import visxd.cyl as vis      # quasi-3D (cylindrical modes) simulations
```

`visxd.cyl` re-exports the full `visxd` API. Particle plots default to (z, r),
and the derived quantities `x = r cos θ` and `y = r sin θ` can be used wherever a
particle quantity is accepted.
