# pensiv
A Rust-based AI & machine learning library

## Overview
`pensiv` is a Rust crate that aims to provide implementations of various artificial intelligence & machine learning techniques.

## Contents
The `src` directory contains the source code for the crate; all other root-level files are either Git- or Cargo-related metadata files. `lib.rs` defines the crate's exports, which it takes from the other files in the directory:
* `stats.rs` implements basic probability & statistics functionality, mainly properties of certain basic discrete & continuous distributions, e.g. PMF/PDF, CDF, mean, & variance

## Installation & Use
In order to use `pensiv` as a Cargo dependency, include it in your `Cargo.toml` manifest file in the `[dependencies]` section:
```toml
# ...

[dependencies]
pensiv = { git = "https://github.com/arthurlafrance/pensiv" )
```

Note that `pensiv` is currently not published to `crates.io`, so the above statement will essentially become obsolete when that happens. Also worth noting is that  documentation of the crate will be available once it's published and ready for use.

## Contributing
As with the previous section, `pensiv` is by no means mature, and is currently mostly a place for my personal experimentation. That said, those so inclined are welcome to use this code for their own personal experimentation. Note that any pull requests that are submitted will be accepeted at my discretion. Also note that I encourage anyone who finds issues with the crate to open an issue to let me know about it; I'm always looking to improve the quality of my code and my coding skills, and that's an important step toward this goal.

## Still to Come
Evidently, `pensiv` hardly lives up to its description as an AI & machine learning library. In the immediate future, it's my goal to add support for:
* Basic classification techniques, i.e. naive Bayes classification
* Adversarial search & game trees
* Basic reinforcement learning techniques

This list will undoubtedly expand in the future as well.
