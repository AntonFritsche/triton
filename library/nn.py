import torch


class nn:
    def __init__(self):
        self.layers = []
        self.depth_nn = len(self.layers)

        self.dev = torch.cuda.current_device()
        self.props = torch.cuda.get_device_properties(self.dev)

        # shared memory (L1 Cache) per SM
        self.shared_mem_per_block_kb = self.props.shared_memory_per_block / 1024

        # max shared memory per LM with dynamic allocation
        self.shared_mem_optin_kb = getattr(self.props, "shared_memory_per_multiprocessor", None)
        if self.shared_mem_optin_kb:
            self.shared_mem_optin_kb /= 1024

        # sram (L2 Cache)
        self.l2_cache_size_mb = getattr(self.props, "l2_cache_size", 0) / (1024 ** 2)

        # SM and register count
        self.num_sms = self.props.multi_processor_count
        self.regs_per_block = self.props.regs_per_block

        # forward + backward propagation
        self.forward_propagation = []
        self.backward_propagation = []

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
                if current_memory_sum_pointwise_group < self.shared_mem_per_block_kb:
                    group_pointwise.append(layer)
                    current_memory_sum_pointwise_group += param_information.memory_usage
                elif current_memory_sum_pointwise_group + param_information.memory_usage > self.shared_mem_per_block_kb:
                    grouping_dict[f"pointwise group {layer_idx}"] = group_pointwise
            elif param_information.type_operation == 'reduction':
                if current_memory_sum_reduction_group < self.shared_mem_per_block_kb:
                    group_reduction.append(layer)
                    current_memory_sum_reduction_group += param_information.memory_usage
                elif current_memory_sum_reduction_group + param_information.memory_usage > self.shared_mem_per_block_kb:
                    grouping_dict[f"reduction group {layer_idx}"] = group_reduction
            else:
                raise Exception(f"Unsupported operation: Available Operations: pointwise, reduction")

        return grouping_dict