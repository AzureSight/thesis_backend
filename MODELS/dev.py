import torch
print(torch.cuda.is_available())  # True if GPU is available, False otherwise
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the first GPU
print(torch.cuda.current_device)
