import torch
import json
import subprocess
from pathlib import Path



# noinspection unsupported-operator
class nn:
    def __init__(self, max_group_size: int):
        self.layers = []
        self.depth_nn = len(self.layers)
        self.max_group_size = max_group_size

        self.dev = torch.cuda.current_device()
        self.props = torch.cuda.get_device_properties(self.dev)

        # shared memory (L1 Cache) per SM
        self.shared_mem_per_block_kb = getattr(self.props, 'shared_memory_per_block', 0)
        if self.shared_mem_per_block_kb != 0:
            self.shared_mem_per_block_kb /= 1024

        # max shared memory per LM with dynamic allocation
        self.shared_mem_optin_kb = getattr(self.props, 'shared_memory_per_multiprocessor', 0)
        if self.shared_mem_optin_kb != 0:
            self.shared_mem_optin_kb /= 1024

        # sram (L2 Cache)
        self.l2_cache_size_mb = getattr(self.props, 'l2_cache_size', 0)
        if self.l2_cache_size_mb != 0:
            self.l2_cache_size_mb /= (1024 ** 2)

        # SM and register count
        self.num_sms = getattr(self.props, 'multi_processor_count', 0)
        self.regs_per_block = getattr(self.props, 'regs_per_block', 0)

        # forward + backward propagation
        self.forward_propagation = []
        self.backward_propagation = []

    def __call__(self, *args, **kwargs):
        return

    def add(self, layer):
        self.layers.append(layer)

    def calculate_grouping(self) -> dict:
        # contains the final grouped fusion kernels
        grouping_dict = {}

        for layer_idx in range(self.depth_nn):
            layer = self.layers[layer_idx]

            group_pointwise = []
            current_memory_sum_pointwise_group = 0

            group_reduction = []
            current_memory_sum_reduction_group = 0

            param_information = layer.get_params()

            if param_information.type_operation == 'pointwise':
                if current_memory_sum_pointwise_group < self.shared_mem_per_block_kb and len(group_pointwise) < self.max_group_size:
                    # new group element in pointwise fused operation group
                    group_pointwise.append(layer)

                    # updating memory usage of the whole group
                    current_memory_sum_pointwise_group += param_information.memory_usage
                elif current_memory_sum_pointwise_group + param_information.memory_usage > self.shared_mem_per_block_kb:
                    gen_operation_string = ""
                    spec_operation_string = ""

                    for operation in group_pointwise:
                        gen_operation_string += operation.operation_type + "_"
                        spec_operation_string += operation.operation_name + "_"

                    grouping_dict[f"{gen_operation_string}-{spec_operation_string}-{layer_idx}"] = group_pointwise
            elif param_information.type_operation == 'reduction':
                if current_memory_sum_reduction_group < self.shared_mem_per_block_kb and len(group_reduction) < self.max_group_size:
                    # new group element in reduction fused operation group
                    group_reduction.append(layer)

                    # updating memory usage of the whole group
                    current_memory_sum_reduction_group += param_information.memory_usage
                elif current_memory_sum_reduction_group + param_information.memory_usage > self.shared_mem_per_block_kb:
                    grouping_dict[f"reduction group {layer_idx}"] = group_reduction
            else:
                raise Exception(f"Unsupported operation: Available Operations: pointwise, reduction")

        return grouping_dict

    def generate_execution(self):
        groups = self.calculate_grouping()

        for key, group in groups.items():
            operation_type = key.split("-")[0]
            operation_name = key.split("-")[1]

            fused_folder = [str(p) for p in Path("library").glob(f"fused_{operation_type}") if p.is_dir()][0]
            fused_kernel = [str(p) for p in Path(fused_folder).glob(f"{operation_name}.py") if p.is_dir()][0]

            self.forward_propagation.append(fused_kernel)

    def train(
        self,
        data: list,
        batch_size: int,
        epochs: int,
        learning_rate: float,
    ):
        batched_data = []
        data_size = len(data)
        num_batches = data_size // batch_size

        batch = []
        for i in range(num_batches):
            batch.append(data[i * batch_size : (i + 1) * batch_size])
            batched_data.append(batch)


        for epoch in range(epochs):
            return