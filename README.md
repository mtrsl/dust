# dust <img src='man/figures/logo.png' align="right" height="139" />

<!-- badges: start -->
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![R build status](https://github.com/mrc-ide/dust/workflows/R-CMD-check/badge.svg)](https://github.com/mrc-ide/dust/actions)
[![Build status](https://badge.buildkite.com/bf7030c393da3ab92f65c63de87bb301b9657a8a9ac6dfb981.svg)](https://buildkite.com/mrc-ide/dust)
[![CodeFactor](https://www.codefactor.io/repository/github/mrc-ide/dust/badge)](https://www.codefactor.io/repository/github/mrc-ide/dust)
[![codecov.io](https://codecov.io/github/mrc-ide/dust/coverage.svg?branch=master)](https://codecov.io/github/mrc-ide/dust?branch=master)
<!-- badges: end -->

Stochastic models appear in many domains as they are easy to write out, but hard to analyse without running many realisations of the process. `dust` provides an engine for running stochastic models, taking care of many bookkeeping details such as:

* Running stochastic models in parallel on multi-core machines
* Running models massively in parallel on GPU without writing any low-level code
* Reducing output by filtering over steps or intermediate outputs
* Providing useful verbs for working with stochastic models (for initialisation, state setting, parameter updating, and inspection)
* Combining multiple realisations due to stochastic variability and due to different parameter sets

## Get started

Two vignettes provide an overview of the package, depending on your tastes:

* `vignette("design")` describes the problems `dust` tries to solve ([read on package website](https://mrc-ide.github.io/dust/articles/design.html))
* `vignette("dust")` describes `dust` by example, showing two simple models and the methods that can drive them ([read on package website](https://mrc-ide.github.io/dust/articles/dust.html))

There are further vignettes describing details:

* `vignette("multi")` on simulating multiple parameter sets at once
* `vignette("rng")` on the parallel random number generator used
* `vignette("data")` on running a bootstrap particle filter
* `vignette("cuda")` on creating and running models on GPUs

## Higher-level

The `dust` package, while designed to be user-friendly, is lower-level than many users need. The [`odin.dust`](https://mrc-ide.github.io/odin.dust/) package provides a way of compiling stochastic [`odin`](https://mrc-ide.github.io/odin/) models to work with `dust`.  For example, to create a parallel epidemiological model, one might write simply:

```r
sir <- odin.dust::odin_dust({
  update(S) <- S - n_SI
  update(I) <- I + n_SI - n_IR
  update(R) <- R + n_IR

  n_IR <- rbinom(I, 1 - exp(-beta * I / (S + I + R)))
  n_SI <- rbinom(S, 1 - exp(-gamma))

  initial(S) <- S_ini
  initial(I) <- I_ini
  initial(R) <- 0

  S_ini <- user(1000)
  I_ini <- user(10)
  beta <- user(0.2)
  gamma <- user(0.1)
})
```

## Use-cases

We use `dust` to power several epidemiological models. Public examples include:

* [`sircovid`](https://mrc-ide.github.io/sircovid/) - a large model (over 15k states) of COVID-19 in the UK
* [`gonovaxdust`](https://mrc-ide.github.io/gonovaxdust/) - a model of gonorrhoea infection with vaccination

## Installation

To install `dust`:

```r
# install.packages("drat") # -- if you don't have drat installed
drat:::add("ncov-ic")
install.packages("dust")
```

You will need a compiler to install dependencies for the package, and to build any models with dust.  `dust` uses `pkgbuild` to build its shared libraries so use `pkgbuild::check_build_tools()` to see if your system is ok to use.

The development version of the package can be installed directly from GitHub if you prefer with:

```r
remotes::install_github("mrc-ide/dust", upgrade = FALSE)
```

## License

MIT © Imperial College of Science, Technology and Medicine
