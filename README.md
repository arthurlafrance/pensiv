# pensiv
Rust-based AI & machine learning

## Overview
`pensiv` is a Rust crate that aims to provide implementations of various artificial intelligence & machine learning techniques. To date, `pensiv` contains implementations of basic probability & statistics as well as adversarial search; see the roadmap below for what will be added in the near future. My ultimate goal is to develop the crate into a comprehensive library of foundational techniques in artificial intelligence & machine learning providing a complete set of optimized implementations of such techniques.

## Installation & Use
In order to use `pensiv` as a Cargo dependency, include it in your `Cargo.toml` manifest file in the `[dependencies]` section:
```toml
# ...

[dependencies]
pensiv = { git = "https://github.com/arthurlafrance/pensiv" )
```

Note that `pensiv` is currently not published to `crates.io`, so the above statement will essentially become obsolete when that happens (unless you want to pull the crate from `git`, perhaps to use an experimental version). Also, when the crate is released for use, two things will be available: full documentation of the crate (complete with runnable example programs) and basic usage instructions.

## Contributing
Evidently, `pensiv` is constantly growing and evolving, so if you have suggestions for features to add or find issues with the code, feel free to create an issue for it and I'll see what I can do.

Do note, though, that any feature requests for types of machine learning techniques will most likely be redundant because there's a high chance I'm already planning on implementing it; see the [roadmap](ROADMAP.md) for more information.

## Still to Come
As mentioned above, `pensiv` is and will continue to be constantly growing. See the [roadmap](ROADMAP.md) for details about what's coming to Pensiv in the future; the short version is that I'm planning to add basic classification, regression, and clustering techniques in the immediate future, with more to come after that.
