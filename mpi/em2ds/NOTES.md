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

## Grid transpose (for FFT)

### 1D parallel partition

We consider (for now) only parallel partitions along y, so that the initial x direction FFTs can be done locally, using `P` processes.

1. Create a target grid object with size ( global_dim.y, global_dim.x ) (notice that the indices are reversed) using the same `P` partition.
2. Separate source into (logical) tiles along the x direction. Each longitudinal tile size must have the same size as the target process y size; if global_dim.x is divisible by `P`.
2. Transpose locally each of these tiles into message buffers
   + The tiles sitting on the diagonal can be transposed directly into the target grid object
   + Alternatively, we can use the message buffer as a staging ground and copy into the target object from here
4. Send messages to target nodes; wait for them to arrive
5. Copy message data into target grid object buffer


## 2D FFT r2c

