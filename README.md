# Motivation

Create triangle meshes from 2D arrays with height information. This allows for nice terrain 3D object, e.g. for 3D printing.

# Installation
Dependencies:
- numpy
- numpy-stl


# Alternatives
- pyvista has very similar functionality, see [Creating a structured surface](https://docs.pyvista.org/version/stable/examples/00-load/create-structured-surface.html#sphx-glr-examples-00-load-create-structured-surface-py).
However, I was not able to save the resulting grid as .stl, so I decided to write my own lightweight package.

- [Terrain2STL](https://jthatch.com/Terrain2STL/) allows you to select an area directly in your browser and download it as an STL - awesome!