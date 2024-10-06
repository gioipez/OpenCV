
# Setup

To install intelligent-scissors was necesary to:

1. download the repo
2. Execute the following command with the virtual enviroment enabled

```shell
pip install wheel --upgrade
```

3. Run the following command in the repo directory

```shell
pip install cython numpy pillow scipy scikit-image opencv-python
```

4. Run the Search installer in the root directory:

```shell
python setup.py build_ext --inplace
```

# Reparing

## Re write `flatten_first_dims` function

Source file: `IntelligentScissors.scissors.utils`

Before:

```python
def flatten_first_dims(x, n_dims=2):
    shape = x.shape
    return np.reshape(x, ((np.product(shape[:n_dims]),) + shape[n_dims:]))
```

It was showing an error in np.product, so it was re-writed to:

```python
def flatten_first_dims(x, n_dims=2):
    shape = x.shape
    return np.reshape(x, ((np.prod(shape[:n_dims]),) + shape[n_dims:]))
```

## `np.int` was depracted and replaced by `np.int32`

