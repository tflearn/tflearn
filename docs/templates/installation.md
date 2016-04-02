# Installation

## Tensorflow Installation

TFLearn requires Tensorflow (version >= 0.7.0) to be installed.

```python
# Ubuntu/Linux 64-bit, CPU only:
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled:
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
easy_install --upgrade six
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
```

- For more details: [Tensorflow installation instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md).

## TFLearn Installation

To install TFLearn, the easiest way is to run:
```python
pip install git+https://github.com/tflearn/tflearn.git
```
Otherwise, you can also install from source by running (from source folder):
```python
python setup.py install
```

## Upgrade Tensorflow

If you version for Tensorflow is too old (under 0.7.0), you may upgrade Tensorflow to avoid some incompatibilities with TFLearn.
To upgrade Tensorflow, you first need to uninstall Tensorflow and Protobuf:

```python
pip uninstall protobuf
pip uninstall tensorflow
```

Then you can re-install Tensorflow:

```python
# Ubuntu/Linux 64-bit, CPU only:
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled:
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
easy_install --upgrade six
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
```

## Using Latest Tensorflow

TFLearn is compatible with [master version](https://github.com/tensorflow/tensorflow) of Tensorflow, but some warnings may appear.
