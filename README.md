# TimeGAN.jl
Julia implementation of "Time-series Generative Adversarial Networks (TimeGAN)"

This repository contains a Julia implementation of the paper "Time-series Generative Adversarial Networks (TimeGAN)" by [Yoon et al.](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf).

The original Python code can be found on the [author's GitHub](https://github.com/jsyoon0823/TimeGAN).

**Disclaimer**: While the current code is function, it is still unfinished and not fully tested.

## TODO List
- [ ] Add GPU compatibility
- [ ] Make `z_dim` match `data` dimension (or not?)
- [x] Add data loading utilities
- [x] Add data scaling
- [ ] Complete / polish docstrings
- [ ] Add example usage / notebook
- [ ] Add moments losses for generator