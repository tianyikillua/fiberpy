# Computational methods for fiber-reinforced composites

This package provides several computational models for fiber-reinforced composites (thermoplastics reinforced by glass fibers, for instance).

- Compute the 4th-order fiber orientation tensor from the 2nd-order one (linear, quadratic, hybrid, orthotropic closure models...)
- Compute the effective thermomechanical properties from the microstructure definition (Mori-Tanaka, orientation averaging...)
- Compute fiber orientation tensor evolution using the Folgar-Tucker model or its variants (RSC model...)

<p align="center">
  <img src="https://user-images.githubusercontent.com/4027283/60251521-f4d52000-98c8-11e9-804e-e3a1d031286d.png" width="800">
</p>

Documentation is available [here](https://fiberpy.readthedocs.io).

## Installation

To install `fiberpy`, run

```sh
pip install -U fiberpy
```

## Testing

To run the `fiberpy` unit tests, run

```sh
pytest
```

## License

`fiberpy` is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
