#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/list.h>

#include <sstream>

#include "em2d/simulation.h"
#include "util.h"

// Additional modules
#include "em2d/laser.h"

namespace nb = nanobind;
using namespace nb::literals;

static inline uint2  to_uint2 (std::pair<unsigned, unsigned> p) { return uint2{ p.first, p.second }; }
static inline float2 to_float2(std::pair<float, float> p)       { return float2{ p.first, p.second }; }
static inline float3 to_float3(std::tuple<float, float, float> t) {
    return float3{ std::get<0>(t), std::get<1>(t), std::get<2>(t) };
}

/**
 * @brief Helper function for gathering vec3grid<float3> component data
 * @note  Data is copied to host and returned as a numpy.ndarray()
 * 
 * @param grid 
 * @param fc 
 * @return auto 
 */
static auto gather_grid_component(vec3grid<float3>* grid, fcomp::cart fc) {
    if (grid == nullptr) {
        throw std::runtime_error("grid pointer is null");
    }

    const uint2 ntiles = grid->ntiles;
    const uint2 nx     = grid->nx;
    const size_t gnx   = static_cast<size_t>(ntiles.x) * nx.x;
    const size_t gny   = static_cast<size_t>(ntiles.y) * nx.y;
    const size_t bsize = gnx * gny;

    float* h_data = new float[bsize];
    float* d_data = device::malloc<float>( bsize );;

    grid->gather(fc, d_data);

    device::memcpy_tohost( h_data, d_data, bsize );
    device::free( d_data );

    nb::capsule owner(h_data, [](void* p) noexcept {
        delete[] static_cast<float*>(p);
    });

    const size_t shape[2] = { gny, gnx };
    return nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>(
        h_data, 2, shape, owner
    );
}


NB_MODULE(_core, m) {
    m.doc() = "ZPIC em2d bindings";

    m.def("sys_info", &util_sys_info);
    m.def("build_info", &util_build_info);

    nb::enum_<coord::cart>(m, "cart", "Cartesian axis selector.")
        .value("x", coord::cart::x)
        .value("y", coord::cart::y)
        .export_values();

    // fcomp.cart enum
    nb::module_ fcomp_mod = m.def_submodule(
        "fcomp",
        "Phase-space quantities for diagnostics."
    );

    nb::enum_<fcomp::cart>(fcomp_mod, "cart",
        "Field component selector.")
        .value("x", fcomp::cart::x)
        .value("y", fcomp::cart::y)
        .value("z", fcomp::cart::z)
        .export_values();

    // part.quant enum
    nb::module_ part_mod = m.def_submodule(
        "part",
        "Particle quantities for diagnostics."
    );

    nb::enum_<part::quant>(part_mod, "quant",
        "Particle quantity selector.")
        .value("x",  part::quant::x)
        .value("y",  part::quant::y)
        .value("ux", part::quant::ux)
        .value("uy", part::quant::uy)
        .value("uz", part::quant::uz)
        .export_values();

    nb::module_ emf_mod = m.def_submodule(
        "emf",
        "EM field."
    );

    nb::enum_<emf::field>(emf_mod, "field",
        "EMF selector.")
        .value("e",  emf::field::e)
        .value("b",  emf::field::b)
        .export_values();

    // Simulation class
    nb::class_<Simulation>(m, "Simulation")
        .def("__init__",
            [](Simulation* self,
               std::pair<unsigned, unsigned> ntiles,
               std::pair<unsigned, unsigned> nx,
               std::pair<float, float>       box,
               double                        dt,
               std::optional<std::vector<Species*>> species )
            {
                new (self) Simulation(
                    to_uint2(ntiles),
                    to_uint2(nx),
                    to_float2(box),
                    dt
                );

                if (species.has_value()) {
                    for (Species* sp : *species) {
                        if (sp == nullptr) {
                            throw nb::value_error(
                                "species list contains a null entry");
                        }
                        self->add_species(*sp);
                    }
                }
            },
            "ntiles"_a, "nx"_a, "box"_a, "dt"_a,
            nb::kw_only(),
            "species"_a = nb::none(),
            "Create a tiled 2D plasma simulation.\n\n"
            "Parameters\n"
            "----------\n"
            "ntiles : (int, int)\n"
            "    Number of tiles along x and y.\n"
            "nx : (int, int)\n"
            "    Cells per tile along x and y.\n"
            "box : (float, float)\n"
            "    Physical size of the simulation box.\n"
            "dt : float\n"
            "    Timestep.\n"
            "species : list[Species], optional\n"
            "    Species to add to the simulation at construction time.\n"
            "    Each species is kept alive for the lifetime of the\n"
            "    simulation."
        )
        .def("__repr__",
            [](const Simulation &s) { 
                std::stringstream repr;
                repr << "<em2d.Simulation"
                     << ", ntiles: " << s.ntiles
                     << ", nx: " << s.nx
                     << ", box: " << s.box
                     << ", dt: " << s.dt
                     << " >";
                return repr.str();
            }
        )
        .def("add_species", &Simulation::add_species,
            "species"_a,
            nb::keep_alive<1, 2>(),   // Simulation (self=1) keeps species (arg=2) alive
            "Add a particle species to the simulation.\n\n"
            "The simulation stores a non-owning reference to the species,\n"
            "so it must outlive the simulation.")
        
        .def_prop_ro("emf",
            [](Simulation& s) -> EMF& { return s.emf; },
            nb::rv_policy::reference_internal,
            "Electromagnetic field object (read-only reference).")
        .def_prop_ro("current",
            [](Simulation& s) -> Current& { return s.current; },
            nb::rv_policy::reference_internal,
            "Electric current object (read-only reference).")

        .def("advance", &Simulation::advance, "Advance simulation 1 iteration")
        .def("advance_mov_window", &Simulation::advance_mov_window, "Advance simulation 1 iteration using a moving window")
        .def("energy_info",  &Simulation::energy_info, "Print global energy diagnostic")
        .def_prop_ro("iter", &Simulation::get_iter, "Current iteration value")
        .def_prop_ro("t", &Simulation::get_t, "Current simulation time")
        .def_prop_ro("nmove", &Simulation::get_nmove, "Total number of particles moved")
    ;


    // EMF class
    nb::class_<EMF>(m,"EMF",
        "Electromagnetic field container.")
        .def_prop_ro("iter", &EMF::get_iter, "Completed iterations")
        .def_prop_ro("dt", &EMF::get_dt, "Time step value")
        .def_prop_ro("box", [](const EMF& self) {
            return std::make_pair(self.box.x, self.box.y);
        })
        .def("__repr__",
            [](const EMF &s) { 
                return "<em2d.EMF , iter: " + std::to_string(s.get_iter()) + ">";
            }
        )
        .def("save",
            [](EMF& self, emf::field field, fcomp::cart fc) {
                self.save(field, fc);
            },
            "field"_a, "fc"_a,
            "Save a single field component to disk.\n\n"
            "Parameters\n"
            "----------\n"
            "field : emf.field\n"
            "    Which field to save (E or B).\n"
            "fc : fcomp.cart\n"
            "    Which Cartesian component (x, y, or z)."
        )

        .def("get_energy",
            [](EMF& self) {
                double3 ene_E{}, ene_B{};
                self.get_energy(ene_E, ene_B);
                return std::make_pair(
                    std::make_tuple(ene_E.x, ene_E.y, ene_E.z),
                    std::make_tuple(ene_B.x, ene_B.y, ene_B.z)
                );
            },
            "Compute the electromagnetic field energy.\n\n"
            "Returns\n"
            "-------\n"
            "((float, float, float), (float, float, float))\n"
            "    Tuple ``(ene_E, ene_B)`` where each is the per-component\n"
            "    energy ``(x, y, z)`` for the electric and magnetic fields\n"
            "    respectively."
        )
        // EMF
        .def("gather",
            [](EMF& self, emf::field field, fcomp::cart fc) {
                vec3grid<float3>* grid = (field == emf::field::e) ? self.E
                                    : (field == emf::field::b) ? self.B
                                    : nullptr;
                if (grid == nullptr) {
                    throw nb::value_error("field must be emf.field.E or emf.field.B");
                }
                return gather_grid_component(grid, fc);
            },
            "field"_a, "fc"_a,
            "Gather a field component from the tiled grid into a contiguous\n"
            "2D NumPy array.\n\n"
            "Parameters\n"
            "----------\n"
            "field : emf.field\n"
            "    Which field to gather (E or B).\n"
            "fc : fcomp.cart\n"
            "    Which Cartesian component (x, y, or z).\n\n"
            "Returns\n"
            "-------\n"
            "numpy.ndarray, shape (ny, nx), dtype float32\n"
            "    The full 2D grid for the selected component. The array owns\n"
            "    its data; modifying it does not affect the simulation."
        )
    ;

    // Current class
    nb::class_<Current>(m,"Current",
        "Electric current container.")
        .def_prop_ro("iter", &Current::get_iter, "Completed iterations")
        .def_prop_ro("dt", &Current::get_dt, "Time step value")
        .def_prop_ro("box", [](const Current& self) {
            return std::make_pair(self.box.x, self.box.y);
        })
        .def("__repr__",
            [](const Current &s) { 
                return "<em2d.Current , iter: " + std::to_string(s.get_iter()) + ">";
            }
        )
        .def("save",
            [](Current& self, fcomp::cart jc) {
                self.save(jc);
            },
            "jc"_a,
            "Save a electric current component to disk.\n\n"
            "Parameters\n"
            "----------\n"
            "fc : fcomp.cart\n"
            "    Which Cartesian component (x, y, or z)."
        )
        .def("gather",
            [](Current& self, fcomp::cart fc) {
                return gather_grid_component(self.J, fc);
            },
            "fc"_a,
            "Gather a field component from the tiled grid into a contiguous\n"
            "2D NumPy array.\n\n"
            "Parameters\n"
            "----------\n"
            "fc : fcomp.cart\n"
            "    Which Cartesian component (x, y, or z).\n\n"
            "Returns\n"
            "-------\n"
            "numpy.ndarray, shape (ny, nx), dtype float32\n"
            "    The full 2D grid for the selected component. The array owns\n"
            "    its data; modifying it does not affect the simulation."
        )
    ;

    // Species class
    nb::class_<Species>(m, "Species")
        .def("__init__",
            [](Species* self,
               const std::string& name,
               float m_q,
               std::pair<unsigned, unsigned> ppc,
               std::optional<UDistribution::Type*> udist,
               std::optional<Density::Profile*> density )
            {
                new (self) Species( name, m_q, to_uint2(ppc) );

                if (udist.has_value() && udist.value() != nullptr) {
                    self->set_udist(*udist.value());
                }
                if (density.has_value() && density.value() != nullptr) {
                    self->set_density(*density.value());
                }
            },
            "name"_a, "m_q"_a, "ppc"_a,
            nb::kw_only(),                     // everything after is keyword-only
            "udist"_a   = nb::none(),
            "density"_a = nb::none(),
            "Create a particle species.\n\n"
            "Parameters\n"
            "----------\n"
            "name : (string)\n"
            "    Name for the species.\n"
            "m_q : (float)"
            "    Mass over charge ratio.\n"
            "ppc : (int, int)\n"
            "    Number of particles per cell."
        )
        
        .def_ro("name", &Species::name, "Name")
        .def_ro("m_q", &Species::m_q, "Mass over charge ratio")

        .def("__repr__",
            [](const Species &s) { 
                return "<em2d.Species , name: " + s.name 
                       + ", iter: " + std::to_string(s.get_iter())
                       + " >";
            }
        )

        // Setters exposed as regular methods too, so users can change
        // udist/density after construction.
        .def("set_udist",   &Species::set_udist,   "udist"_a)
        .def("set_density", &Species::set_density, "density"_a)

        .def_prop_ro("energy", &Species::get_energy, "Time centered kinetic energy")
        .def_prop_ro("nmove", &Species::get_nmove, "Total number of particles moved")
        .def_prop_ro("iter", &Species::get_iter, "Completed iterations")
        .def_prop_ro("dt", &Species::get_dt, "Time step value")
        .def_prop_ro("ntiles", &Species::get_ntiles, "Number of tiles")
        .def_prop_ro("nx", &Species::get_nx, "Tile size")

        .def_prop_ro("box", [](const Species& self) {
            auto box = self.get_box();
            return std::make_pair(box.x, box.y);
        })
        .def("save", &Species::save, "Save particle data to file")
        .def("save_charge", &Species::save_charge, "Save charge density for species to file")

        .def("save_phasespace",
            [](const Species& s,
            phasespace::quant quant,
            std::pair<float, float> range,
            int size)
            {
                if (size <= 0) {
                    throw nb::value_error("size must be > 0");
                }
                s.save_phasespace(quant, to_float2(range), size);
            },
            "quant"_a, "range"_a, "size"_a,
            "Save a 1D phase-space diagnostic.\n\n"
            "Parameters\n"
            "----------\n"
            "quant : phasespace.quant\n"
            "    Quantity to bin along the axis.\n"
            "range : (float, float)\n"
            "    Lower and upper bounds of the histogram.\n"
            "size : int\n"
            "    Number of bins (must be > 0).")

        .def("save_phasespace",
            [](const Species& s,
            phasespace::quant quant0,
            std::pair<float, float> range0,
            int size0,
            phasespace::quant quant1,
            std::pair<float, float> range1,
            int size1)
            {
                if (size0 <= 0 || size1 <= 0) {
                    throw nb::value_error("size0 and size1 must be > 0");
                }
                s.save_phasespace(
                    quant0, to_float2(range0), size0,
                    quant1, to_float2(range1), size1
                );
            },
            "quant0"_a, "range0"_a, "size0"_a,
            "quant1"_a, "range1"_a, "size1"_a,
            "Save a 2D phase-space diagnostic.\n\n"
            "Parameters\n"
            "----------\n"
            "quant0, quant1 : phasespace.quant\n"
            "    Quantities to bin along each axis.\n"
            "range0, range1 : (float, float)\n"
            "    Lower and upper bounds of the histogram along each axis.\n"
            "size0, size1 : int\n"
            "    Number of bins along each axis (must be > 0).")

        .def("gather",
            [](Species &self, part::quant quant) {
                // Total number of particles owned by this species
                const size_t n = self.np_total();

                // Allocate an owning buffer on the device
                float * h_data = new float[n];
                float * d_data = device::malloc<float>( n );

                self.gather(quant, d_data);
                device::memcpy_tohost( h_data, d_data, n );
                device::free( d_data );

                nb::capsule owner(h_data, [](void *p) noexcept {
                    delete[] static_cast<float *>(p);
                });

                // Shape must live long enough for the constructor call; a local is fine.
                size_t shape[1] = { n };

                return nb::ndarray<nb::numpy, float, nb::ndim<1>>(
                    /* data  */ h_data,
                    /* ndim  */ 1,
                    /* shape */ shape,
                    /* owner */ owner
                );
            },
            "quant"_a,
            "Gather the given quantity for every particle in the species.\n"
            "Returns a new 1-D NumPy array of length np_total(), independent of "
            "the simulation state."
        )

        .def("get_charge",
            [](Species &self) {
                // Linear interpolation: 1 guard cell on the upper boundary
                bnd<unsigned int> gc;
                gc.x = {0, 1};
                gc.y = {0, 1};

                // Deposit charge on a temporary tiled grid
                grid<float> charge(self.get_ntiles(), self.get_nx(), gc);
                charge.zero();
                self.deposit_charge(charge);
                charge.add_from_gc();

                // Global (untiled) dimensions of the gathered array.
                // ntiles.{x,y} tiles, each nx.{x,y} cells — guard cells are NOT
                // included in the gathered output.
                const size_t ny = static_cast<size_t>(self.get_ntiles().y) *
                                  static_cast<size_t>(self.get_nx().y);
                const size_t nx = static_cast<size_t>(self.get_ntiles().x) *
                                  static_cast<size_t>(self.get_nx().x);
                const size_t n  = nx * ny;

                // Owning contiguous buffer that will outlive `charge`
                float* h_data = new float[ n ];
                float* d_data = device::malloc<float>( n ); 

                charge.gather(d_data);

                device::memcpy_tohost( h_data, d_data, n );
                device::free( d_data );

                // Capsule frees the buffer when the ndarray is GC'd
                nb::capsule owner( h_data, [](void *p) noexcept {
                    delete[] static_cast<float *>(p);
                });

                // Shape: (ny, nx) — row-major, C order
                size_t shape[2] = { ny, nx };

                return nb::ndarray<nb::numpy, float, nb::ndim<2>>(
                    h_data,
                    2,
                    shape,
                    owner
                );
            },
            "Deposit and return the charge density as a new 2-D array on the GPU.\n"
            "The returned array is independent of the simulation."
        )

        .def("get_phasespace",
            [](Species& self,
               phasespace::quant quant,
               std::pair<float, float>       range,
               unsigned size )
            { 
                if (size <= 2) {
                    throw nb::value_error("size must be > 1");
                }

                float* h_data = new float[size];
                float* d_data = device::malloc<float>( size );
                
                self.dep_phasespace( d_data, quant, to_float2(range), size);
                device::memcpy_tohost( h_data, d_data, size );
                device::free( d_data );
                
                nb::capsule owner( h_data, [](void *p) noexcept {
                    delete[] static_cast<float *>(p);
                });

                size_t shape[1] = { size };
                return nb::ndarray<nb::numpy, float, nb::ndim<1>>(
                    h_data,
                    1,
                    shape,
                    owner
                );
            },
            "quant"_a, "range"_a, "size"_a,
            "Deposit and return the requested phasespace density as a new 1-D array on the GPU.\n\n"
            "Parameters\n"
            "----------\n"
            "quant : phasespace.quant\n"
            "    Quantity to bin along the axis.\n"
            "range : (float, float)\n"
            "    Lower and upper bounds of the histogram.\n"
            "size : int\n"
            "    Number of bins (must be > 0).\n\n"
            "The returned array is independent of the simulation."
        )

        .def("get_phasespace",
            [](Species& self,
               phasespace::quant quant0, std::pair<float, float> range0, unsigned size0,
               phasespace::quant quant1, std::pair<float, float> range1, unsigned size1
            )
            { 
                if (size0 <= 2 || size1 <= 2) {
                    throw nb::value_error("Both size0 and size1 must be > 1");
                }
                const size_t total = size0 * size1;

                float* h_data = new float[ total ];
                float* d_data = device::malloc<float>( total );

                self.dep_phasespace( d_data, 
                    quant0, to_float2(range0), size0,
                    quant1, to_float2(range1), size1
                );
                
                device::memcpy_tohost( h_data, d_data, total );
                device::free( d_data );

                nb::capsule owner( h_data, [](void *p) noexcept {
                    delete[] static_cast<float *>(p);
                });

                size_t shape[2] = { size0, size1 };
                return nb::ndarray<nb::numpy, float, nb::ndim<2>>(
                    h_data,
                    2,
                    shape,
                    owner
                );
            },
            "quant0"_a, "range0"_a, "size0"_a,
            "quant1"_a, "range1"_a, "size1"_a,
            "Deposit and return the requested phasespace density as a new 2-D array on the GPU.\n\n"
            "Parameters\n"
            "----------\n"
            "quant0, quant1 : phasespace.quant\n"
            "    Quantities to bin along each axis.\n"
            "range0, range1 : (float, float)\n"
            "    Lower and upper bounds of the histogram along each axis.\n"
            "size0, size1 : int\n"
            "    Number of bins along each axis (must be > 0).\n\n"
            "The returned array is independent of the simulation."
        )
        ;

        nb::module_ phasespace_mod = m.def_submodule(
            "phasespace",
            "Phase-space quantities for diagnostics."
        );

        nb::enum_<phasespace::quant>(phasespace_mod, "quant",
            "Phase-space quantity selector.")
            .value("x",  phasespace::x,  "Position along x.")
            .value("y",  phasespace::y,  "Position along y.")
            .value("ux", phasespace::ux, "Generalized velocity along x.")
            .value("uy", phasespace::uy, "Generalized velocity along y.")
            .value("uz", phasespace::uz, "Generalized velocity along z.")
            .export_values();

    // =====================================================================
    // udist submodule
    // =====================================================================
    nb::module_ udist = m.def_submodule(
        "udist",
        "Velocity distribution types for particle initialization."
    );

    // ---- Abstract base ----
    // Bound here (in the submodule) so it acts as the parent for all
    // concrete distributions. Species::set_udist(UDistribution::Type const &)
    // will accept any of the derived types below.
    nb::class_<UDistribution::Type>(udist, "Type", 
        "Abstract base class for velocity distributions.");

    // ---- None ----
    nb::class_<UDistribution::None, UDistribution::Type>(udist, "Zero")
        .def(nb::init<>(),
             "Zero-velocity distribution (all particles start at rest).")
        .def("__repr__",
            [](const UDistribution::None &u) { 
                return "<em2d.UDistribution::None>";
            }
        )
    ;

    // ---- Cold ----
    nb::class_<UDistribution::Cold, UDistribution::Type>(udist, "Cold")
        .def("__init__",
            [](UDistribution::Cold* self, std::tuple<float, float, float> ufl) {
                new (self) UDistribution::Cold(to_float3(ufl));
            },
            "ufl"_a,
            "Cold beam with fluid velocity `ufl` = (ux, uy, uz)."
        )
        .def("__repr__",
            [](const UDistribution::Cold &u) { 
                std::stringstream repr;
                repr << "<UDistribution::Cold"
                     << ", ufl: " << u.ufl
                     << " >";
                return repr.str();
            }
        )
        .def_prop_ro("ufl", [](const UDistribution::Cold& u) {
            return std::make_tuple(u.ufl.x, u.ufl.y, u.ufl.z);
        }        )
    ;

    // ---- Thermal ----
    nb::class_<UDistribution::Thermal, UDistribution::Type>(udist, "Thermal")
        .def("__init__",
            [](UDistribution::Thermal* self,
               std::tuple<float, float, float> uth,
               std::tuple<float, float, float> ufl)
            {
                new (self) UDistribution::Thermal(to_float3(uth), to_float3(ufl));
            },
            "uth"_a, "ufl"_a,
            "Maxwellian distribution.\n\n"
            "Parameters\n"
            "----------\n"
            "uth : (float, float, float)\n"
            "    Thermal velocity along each axis.\n"
            "ufl : (float, float, float)\n"
            "    Fluid (drift) velocity along each axis.")
        .def("__repr__",
            [](const UDistribution::Thermal &u) { 
                std::stringstream repr;
                repr << "<UDistribution::Thermal"
                     << ", uth: " << u.uth
                     << ", ufl: " << u.ufl
                     << " >";
                return repr.str();
            }
        )
        .def_prop_ro("ufl", [](const UDistribution::Thermal& u) {
            return std::make_tuple(u.ufl.x, u.ufl.y, u.ufl.z);
        })
        .def_prop_ro("uth", [](const UDistribution::Thermal& u) {
            return std::make_tuple(u.uth.x, u.uth.y, u.uth.z);
        })
    ;

    // ---- ThermalCorr ----
    nb::class_<UDistribution::ThermalCorr, UDistribution::Type>(udist, "ThermalCorr")
        .def("__init__",
            [](UDistribution::ThermalCorr* self,
               std::tuple<float, float, float> uth,
               std::tuple<float, float, float> ufl,
               int npmin)
            {
                new (self) UDistribution::ThermalCorr(
                    to_float3(uth), to_float3(ufl), npmin
                );
            },
            "uth"_a, "ufl"_a, "npmin"_a = 2,
            "Maxwellian distribution with fluid-velocity correction.\n\n"
            "Parameters\n"
            "----------\n"
            "uth : (float, float, float)\n"
            "    Thermal velocity along each axis.\n"
            "ufl : (float, float, float)\n"
            "    Fluid (drift) velocity along each axis.\n"
            "npmin : int, optional\n"
            "    Minimum number of particles per cell for the correction\n"
            "    (must be > 1). Default: 2.")
        .def("__repr__",
            [](const UDistribution::ThermalCorr &u) { 
                std::stringstream repr;
                repr << "<UDistribution::Thermal"
                     << ", uth: " << u.uth
                     << ", ufl: " << u.ufl
                     << ", npmin: " << u.npmin
                     << " >";
                return repr.str();
            }
        )
        .def_prop_ro("ufl", [](const UDistribution::ThermalCorr& u) {
            return std::make_tuple(u.ufl.x, u.ufl.y, u.ufl.z);
        })
        .def_prop_ro("uth", [](const UDistribution::ThermalCorr& u) {
            return std::make_tuple(u.uth.x, u.uth.y, u.uth.z);
        })
        .def_ro("npmin", &UDistribution::ThermalCorr::npmin)
    ;

    // =====================================================================
    // density submodule
    // =====================================================================
    nb::module_ density = m.def_submodule(
        "density",
        "Plasma density profiles for particle injection."
    );

    // ---- Abstract base ----
    nb::class_<Density::Profile>(density, "Profile")
        .def_ro("n0", &Density::Profile::n0)
        .doc() = "Abstract base class for density profiles.";

    // ---- None ----
    nb::class_<Density::None, Density::Profile>(density, "None_")
        .def(nb::init<>(),
             "Zero density (disables particle injection).")
        .def("__repr__",
            [](const Density::None &u) { return "<Density::None>"; }
        )
    ;

    // ---- Uniform ----
    nb::class_<Density::Uniform, Density::Profile>(density, "Uniform")
        .def(nb::init<float>(), "n0"_a,
             "Spatially uniform density.\n\n"
             "Parameters\n"
             "----------\n"
             "n0 : float\n"
             "    Reference density (absolute value is used).")
        .def("__repr__",
            [](const Density::Uniform &d) { 
                return "<Density::Uniform, n0: " + std::to_string(d.n0)
                    + " >"; 
            }
        )
    ;

    // ---- Step ----
    nb::class_<Density::Step, Density::Profile>(density, "Step")
        .def("__init__",
            [](Density::Step* self,
                std::string dir,
                float n0,
                float pos)
            {
                coord::cart _dir;
                if ( dir == "x" ) {
                    _dir = coord::x;
                } else if ( dir == "y" ) {
                    _dir = coord::y;
                } else {
                    throw nb::value_error(
                                "Invalid dir parameter, expected 'x' or 'y'");
                }

                new (self) Density::Step( _dir, n0, pos );
            },
            "dir"_a, "n0"_a, "pos"_a,
            "Heaviside step density: uniform n0 beyond `pos` along `dir`.\n\n"
            "Parameters\n"
            "----------\n"
            "dir : cart\n"
            "    Axis along which the step is applied (cart.x or cart.y).\n"
            "n0 : float\n"
            "    Density on the populated side of the step.\n"
            "pos : float\n"
            "    Position of the step along the chosen axis."
        )
        .def("__repr__",
            [](const Density::Step &d) { 
                std::stringstream repr;
                repr << "<Density::Step"
                     << ", dir: " << (( d.dir == coord::x ) ? 'x' : 'y' )
                     << ", n0: " << d.n0
                     << ", pos: " << d.pos
                     << " >";
                return repr.str();
            }
        )
        .def_ro("pos", &Density::Step::pos)
        .def_ro("dir", &Density::Step::dir)
    ;

    // ---- Slab ----
    nb::class_<Density::Slab, Density::Profile>(density, "Slab")
        .def("__init__",
            [](Density::Slab* self,
                std::string dir,
                float n0,
                float begin,
                float end)
            {
                coord::cart _dir;
                if ( dir == "x" ) {
                    _dir = coord::x;
                } else if ( dir == "y" ) {
                    _dir = coord::y;
                } else {
                    throw nb::value_error(
                                "Invalid dir parameter, expected 'x' or 'y'");
                }

                new (self) Density::Slab( _dir, n0, begin, end );
            },
             "dir"_a, "n0"_a, "begin"_a, "end"_a,
             "Uniform density inside a 1D slab, zero outside.\n\n"
             "Parameters\n"
             "----------\n"
             "dir : cart\n"
             "    Axis along which the slab extends.\n"
             "n0 : float\n"
             "    Density inside the slab.\n"
             "begin : float\n"
             "    Lower edge of the slab along `dir`.\n"
             "end : float\n"
             "    Upper edge of the slab along `dir`."
        )
        .def("__repr__",
            [](const Density::Slab &d) { 
                std::stringstream repr;
                repr << "<Density::Slab"
                     << ", dir: " << (( d.dir == coord::x ) ? 'x' : 'y' )
                     << ", n0: " << d.n0
                     << ", begin: " << d.begin
                     << ", end: " << d.end
                     << " >";
                return repr.str();
            }
        )
        .def_ro("begin", &Density::Slab::begin)
        .def_ro("end",   &Density::Slab::end)
        .def_ro("dir",   &Density::Slab::dir);

    // ---- Sphere ----
    nb::class_<Density::Sphere, Density::Profile>(density, "Sphere")
        .def("__init__",
            [](Density::Sphere* self,
               float n0,
               std::pair<float, float> center,
               float radius)
            {
                new (self) Density::Sphere(n0, to_float2(center), radius);
            },
            "n0"_a, "center"_a, "radius"_a,
            "Uniform-density sphere (2D disc) centered at `center`.\n\n"
            "Parameters\n"
            "----------\n"
            "n0 : float\n"
            "    Density inside the sphere.\n"
            "center : (float, float)\n"
            "    Center position (x, y).\n"
            "radius : float\n"
            "    Radius.")
        .def("__repr__",
            [](const Density::Sphere &d) { 
                std::stringstream repr;
                repr << "<Density::Sphere"
                     << ", n0: " << d.n0
                     << ", center: " << d.center
                     << ", radius: " << d.radius
                     << " >";
                return repr.str();
            }
        )
        .def_prop_ro("center", [](const Density::Sphere& s) {
            return std::make_pair(s.center.x, s.center.y);
        })
        .def_ro("radius", &Density::Sphere::radius)
    ;

    // =====================================================================
    // laser submodule
    // =====================================================================
    nb::module_ laser = m.def_submodule(
        "laser",
        "Laser pulses."
    );

    // ---- Abstract base ----
    nb::class_<Laser::Pulse>(laser, "Pulse")
        .def_rw("start", &Laser::Pulse::start, "Front edge of the laser pulse")
        .def_rw("fwhm", &Laser::Pulse::fwhm, "FWHM of the laser pulse duration")
        .def_rw("rise", &Laser::Pulse::rise, "Rise time of the laser pulse")
        .def_rw("flat", &Laser::Pulse::flat, "Flat time of the laser pulse")
        .def_rw("fall", &Laser::Pulse::fall, "Fall time of the laser pulse")
        .def_rw("a0", &Laser::Pulse::a0, "Normalized peak vector potential")
        .def_rw("omega0", &Laser::Pulse::omega0, "Laser frequency")
        .def_rw("polarization", &Laser::Pulse::polarization, "Polarization angle")
        .def_rw("cos_pol", &Laser::Pulse::cos_pol, "Cosine of the polarization angle")
        .def_rw("sin_pol", &Laser::Pulse::sin_pol, "Sine of the polarization angle")
        .def_rw("filter", &Laser::Pulse::filter, "Filter level")
        .def("add", &Laser::Pulse::add, "Add laser pulse onto EMF grid")
        .doc() = "Abstract base class for laser pulses."
    ;

    // ---- Plane wave ----
    nb::class_<Laser::PlaneWave, Laser::Pulse>(laser, "PlaneWave")
        .def("__init__",
            []( Laser::PlaneWave* self,
                std::optional<float> start,
                std::optional<float> fwhm,
                std::optional<float> rise,
                std::optional<float> flat,
                std::optional<float> fall,
                std::optional<float> a0,
                std::optional<float> omega0,
                std::optional<float> polarization,
                std::optional<float> cos_pol,
                std::optional<float> sin_pol,
                std::optional<int> filter
            )
            {
                new (self) Laser::PlaneWave();

                if ( start.has_value() ) self->start = start.value();
                if ( fwhm.has_value() ) self->fwhm = fwhm.value();
                if ( rise.has_value() ) self->rise = rise.value();
                if ( flat.has_value() ) self->flat = flat.value();
                if ( fall.has_value() ) self->fall = fall.value();

                if ( a0.has_value() ) self->a0 = a0.value();
                if ( omega0.has_value() ) self->omega0 = omega0.value();

                if ( polarization.has_value() ) self->polarization = polarization.value();
                if ( cos_pol.has_value() ) self->cos_pol = cos_pol.value();
                if ( sin_pol.has_value() ) self->sin_pol = sin_pol.value();

                if ( filter.has_value() ) self->filter = filter.value();
            },
            nb::kw_only(),
            "start"_a = nb::none(),
            "fwhm"_a = nb::none(),
            "rise"_a = nb::none(),
            "flat"_a = nb::none(),
            "fall"_a = nb::none(),
            "a0"_a = nb::none(),
            "omega0"_a = nb::none(),
            "polarization"_a = nb::none(),
            "cos_pol"_a = nb::none(),
            "sin_pol"_a = nb::none(),
            "filter"_a = nb::none(),
            "Plane wave laser pulse.\n\n"
            "Parameters\n"
            "----------\n"
            "start : float, optional\n"
            "    Position of front edge of laser pulse\n"
            "fwhm : float, optional\n"
            "    FWHM of the laser pulse duration\n"
            "rise : float, optional\n"
            "    Rise time of the laser pulse\n"
            "flat : float, optional\n"
            "    Flat time of the laser pulse\n"
            "fall : float, optional\n"
            "    Fall time of the laser pulse\n"
            "a0 : float, optional\n"
            "    Normalized peak vector potential\n"
            "omega0 : float, optional\n"
            "    Laser frequency\n"
            "polarization : float, optional\n"
            "    Polarization angle (radians)\n"
            "cos_pol : float, optional\n"
            "    Cosine of the polarization angle\n"
            "sin_pol : float, optional\n"
            "    Sine of the polarization angle\n"
            "filter : int, optional\n"
            "    Filter level"
        )
        .def("add", &Laser::PlaneWave::add, "Add laser pulse onto EMF grid")
    ;
}

