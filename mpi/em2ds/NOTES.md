# MPI / em2ds

## Data structures

### `grid.h`

This is a tiled grid object, supporting multiple datatypes and distributed memory nodes. Each tile has an optional halo. Main methods are:

+ `gather()` Gather tile data into a contiguous grid. Data is gathered on local node only.
+ `scatter()` Scatter contiguous grid data into a tiled grid. Data is scattered on local node only.
+ `copy_to_gc()` Copy values into neighboring guard cells (local node and MPI)
+ `add_from_gc()` Add values from neighboring guard cells (local node and MPI)
+ `kernel3_x()` and `kernel3_y()` - Apply 3 point kernel convolution (local node only)
+ `save()` Save grid data onto a `.zdf` file. Gathers data from all MPI nodes.

### `basic_grid.h`

This is a grid object, mapped continuously in memory, supporting multiple datatypes and distributed memory nodes.

#### Initialization

The initialization uses a parallel partition object to decide the grid size on each MPI process:

```c++
basic_grid( uint2 const global_dims, bnd<unsigned int> const gc, Partition & part, 
        uint2 const granularity)
```

The `granularity` parameter allows control over the possible local grid size values; these values will need to be a multiple of the `granularity` value. The routine will check if `global_dims` divides evenly by `granularity`.

This allows us to easily create a `basic_grid` that that is "compatible" with a tiled grid, as long as the tile size is equal to the granularity.

