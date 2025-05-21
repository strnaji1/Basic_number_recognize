import torch

# Základní info
print("PyTorch verze:", torch.__version__)
print("CUDA dostupná:", torch.cuda.is_available())

# Tenzor + výpočet
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = a + b

print("\nTenzor A:")
print(a)
print("\nTenzor B:")
print(b)
print("\nSoučet A + B:")
print(c)
