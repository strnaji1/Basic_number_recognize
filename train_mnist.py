import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, TensorDataset

# =======================
# Nastavení zařízení
# =======================
device = torch.device("cpu")
print("Zařízení:", device)

# =======================
# Transformace a data
# =======================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# =======================
# Model
# =======================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =======================
# Trénování
# =======================
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Průměrná ztráta: {total_loss/len(train_loader):.4f}")

# =======================
# Vyhodnocení přesnosti
# =======================
correct = 0
total = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f"\nTestovací přesnost: {100 * correct / total:.2f}%")

# =======================
# Vizualizace 50 testovacích obrázků
# =======================
examples = next(iter(test_loader))
images, labels = examples
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(20, 10))
for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"Skutečné: {labels[i]}\nPředpověď: {preds[i]}", fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.show()

# =======================
# Klasifikace vlastního obrázku
# =======================
def predict_image(image_path, model):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
    return pred, img

# Zadej vlastní obrázek
image_path = "my_digit.png"  # Nahraď podle potřeby
predicted, img_for_plot = predict_image(image_path, model)
print(f"Předpovězené číslo: {predicted}")

plt.imshow(img_for_plot, cmap='gray')
plt.title(f"Předpověď: {predicted}", fontsize=16)
plt.axis("off")
plt.show()

# Načtení obrázku
img = Image.open("my_8.png").convert("L").resize((28, 28))

# Převod na tensor + normalizace (jako MNIST)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])
img_tensor = transform(img)

# Vytvoř tensor se správným štítkem (např. 5)
my_label = torch.tensor([8])  # číslo na obrázku
my_tensor = img_tensor.unsqueeze(0)  # přidání batch dimenze

# Vytvoř dataset z jednoho obrázku
my_dataset = TensorDataset(my_tensor, my_label)

# Spoj s původním trénovacím datasetem
combined_dataset = ConcatDataset([train_dataset, my_dataset])

# Nový dataloader
train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

image_path = "my_8.png"
skutecna_hodnota = 8  # Označ sám, co je správná odpověď

predicted, img_for_plot = predict_image(image_path, model)

# Výpis do konzole
print(f"\n🧾 Výsledek pro obrázek '{image_path}':")
print(f"   ✅ Skutečná hodnota: {skutecna_hodnota}")
print(f"   🔮 Předpověď modelu: {predicted}")

# Vykreslení obrázku
plt.imshow(img_for_plot, cmap='gray')
plt.title(f"Skutečné: {skutecna_hodnota} | Předpověď: {predicted}", fontsize=16)
plt.axis("off")
plt.show()