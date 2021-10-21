## Non-monotonic Attentive Graph Neural Networks

### Environment

Pytorch >= 1.8

Pytorch Geometric >= 1.7

### Introduction

Modify the GAT, GLCN, IDGL and add an non-monotonic mapping layer in the original models.

As for the non-monotonic functions, we achieve sin mapping, fourier-basis mapping, polynomial-basis mapping, and radial-basis mapping.

### Simple Test

Run GAT with sin mapping on Cora:

```shell
sh run_cora.sh
```

