## Introduction
gal1d is a python package for analysing functional forms of known surface or volume density (luminosity) profiles and fitting them.


## Installation

```
git clone https://github.com/wx-ys/gal1d.git
```
Install this in editable mode.
```
cd gal1d
pip install -e .
```
See [example](example.ipynb) for usage,


## Profiles
analysis a single theoretical profile or a combination of the addition of some of these profiles.
* NFWProfile: a Navarro-Frenk-White (NFW) profile
* GNFWProfile: a generalized NFW profile profile (Kravtsov 1998)
* DoublePowerLawProfile: a 'broken' or double power-law profile (Hernquist 1990; Zhao 1996)
* EinastoProfile: an Einasto profile (Einasto 1965, 1969)
* SersicProfile: a Sersic profile (Sérsic 1963, 1968)
* ExponentialProfile: an exponential profile
* CoreSersicProfile: a core-Sersic profile (Erwin et al. 2008)

...

## Maintainers

[@wx-ys](https://github.com/wx-ys).


## License

[MIT](LICENSE) © Shuai Lu

## Acknowledgments
* [pynbody](https://github.com/pynbody/pynbody)
* [pyimfit](https://github.com/perwin/pyimfit)
* [lenstronomy](https://lenstronomy.readthedocs.io/en/latest/)