import torch
import json
import importlib
import sys
from pathlib import Path



# noinspection unsupported-operator
class nn:
    def __init__(self, max_group_size: int, error_function):
        self.layers = []
        self.depth_nn = len(self.layers)
        self.error_function = error_function
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

    def __call__(self, data):
        if len(self.forward_propagation) == 0:
            raise Exception("No layers / fused kernels ready for forward propagation!")

        current_data = data

        for kernel_function in self.forward_propagation:
            current_data = kernel_function(current_data)

        return current_data

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

        if "library" not in sys.path:
            sys.path.append("library")

        for key, group in groups.items():
            operation_type = key.split("-")[0]
            operation_name = key.split("-")[1]

            module_name = f"library.fused_{operation_type}.{operation_name}"

            try:
                module = importlib.import_module(module_name)

                kernel_func = getattr(module, 'forward_kernel')

                self.forward_propagation.append(kernel_func)

            except ImportError as e:
                raise Exception(f"Error while importing {module_name}: {e}")
            except AttributeError as e:
                raise Exception(f"The function 'forward_kernel' was not found in {module_name}: {e}")

    def train(
        self,
        train_data: list,
        train_labels: list,
        test_data: list,
        test_labels: list,
        batch_size: int,
        epochs: int,
        learning_rate: float,
    ):
        batched_train_data, batched_test_data = [], []
        train_data_size, test_data_size = len(train_data), len(test_data)

        if train_data_size != test_data_size:
            raise Exception(f"Training and testing data sizes do not match: {train_data_size}, {test_data_size}")

        num_train_batches = train_data_size // batch_size
        num_test_batches = test_data_size // batch_size

        # train batches
        for i in range(num_train_batches):
            batch = [
                train_data[i * batch_size: (i + 1) * batch_size],
                train_labels[i * batch_size: (i + 1) * batch_size],
            ]
            batched_train_data.append(batch)

        # test batches
        for i in range(num_test_batches):
            batch = [
                test_data[i * batch_size: (i + 1) * batch_size],
                test_labels[i * batch_size: (i + 1) * batch_size],
            ]
            batched_test_data.append(batch)

        for epoch_idx in range(epochs):
            forward_passes = []

            # train phase
            for batch_idx in range(num_train_batches):
                forward_pass = []
                current_batch = batched_train_data[batch_idx]

                current_data = current_batch
                for kernel_function in self.forward_propagation:
                    current_data = kernel_function(current_data)
                    forward_pass.append(kernel_function(current_data))

                forward_passes.append(forward_pass)

            # test phase
            for batch_idx in range(num_test_batches):
                current_batch = batched_test_data[batch_idx]

                current_data = current_batch
                for kernel_function in self.forward_propagation:
                    current_data = kernel_function(current_data)

                error = 0
                for i in range(batch_size):
                    error += self.error_function(current_data[i], test_labels[i])

                error /= batch_size

            current_data = error

            for kernel_function in self.backward_propagation:
