
This repo contains a Rust implementation of the GCM paper, and Python bindings.

# Installation 

You need to have `rust` (from my personal experience, installing rust with homebrew messes things up) and `maturin` installed (`pip install maturin`). 
Note that you may need to use the `nightly` toolchain : 
```
rustup install nightly
rustup default nightly
```

If you want to generate the Python package without installing it : 
```
maturin build (--release) (-Z build-std --target aarch64-apple-darwin)
```
You'll find a file named `libgcmpyo3.dylib` in `target/{debug,release}`. Just rename the file into `gcmpyo3.so` to use it. Otherwise, you can pip install the wheel file located in the `target/wheels` folder.
If you use a M1, compiling without the option `-Z build-std --target aarch64-apple-darwin` (which compiles the standard library) might not work.

Use `maturin develop` to compile + install the library.

**If you are not on Mac OS**

You should probably comment the following lines in the Cargo.toml file : 
```
[target.x86_64-apple-darwin]
rustflags = [
  "-C", "link-arg=-undefined",
  "-C", "link-arg=dynamic_lookup",
]

[target.aarch64-apple-darwin]
rustflags = [
  "-C", "link-arg=-undefined",
  "-C", "link-arg=dynamic_lookup",
]
```
but it might work with these lines anyway, what do I know.

# Testing 

To make sure things are working, run  
```
>>> import gcmpyo3
>>> gcmpyo3.test()
The module is loaded correctly
```

