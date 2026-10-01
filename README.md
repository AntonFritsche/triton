# Custom NN-Library with Fused Triton-Operations
- it calculates grouping for fused operations based on memory usage of every layer
- chooses grouping size (with max group size) based on memory usage to utilize the gpu to the maximum
