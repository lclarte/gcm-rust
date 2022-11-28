
# Installation 

If you want to generate the Python package without installing it : 
```
maturin build (--release)
```

You'll find a file named `libgcmpyo3.dylib` in `target/{debug,release}`. Just rename the file into `gcmpyo3.so` to use it. Otherwise, you can pip install the wheel file located in the `target/wheels` folder.

Use `maturin develop` to compile + install the library.

# Testing 

To make sure things are working, run  
```
>>> import gcmpyo3
>>> gcmpyo3.test()
The module is loaded correctly
```

# Summary of variable names 

**Parameters**

$\lambda$ : $\ell_2$ penalization of logistic regression

$\beta$   : inverse temperature in pseudo-bayes

$\alpha$  : sampling ratio

$\delta$  : variance of the Gaussian noise 

$\gamma$  : ratio between student and teacher dimensions

$\kappa_1, \kappa_{\star}$ : parameters of the student covariance in the random feature case
