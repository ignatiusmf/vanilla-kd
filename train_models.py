import time
start_time = time.time()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a convolutional teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample to 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample to 8x8

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample to 4x4

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.network(x)

# Define a smaller convolutional student model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    soft_teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    soft_loss = nn.functional.kl_div(soft_student_probs, soft_teacher_probs, reduction='batchmean') * (temperature ** 2)
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Training loop
def train_model(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        print(epoch)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# Training loop for knowledge distillation
def train_kd(teacher_model, student_model, train_loader, optimizer, temperature, alpha, device, epochs):
    student_model.train()
    teacher_model.eval()
    for epoch in range(epochs):
        print(epoch)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            student_logits = student_model(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# Main function
def main():
    batch_size = 64
    learning_rate = 0.001
    temperature = 5.0
    alpha = 0.7
    epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data")
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Creating models")
    # Models
    teacher_model = TeacherModel().to(device)
    student_model_pretrained = StudentModel().to(device)
    student_model_kd = StudentModel().to(device)

    print("Training teacher model")
    # Train teacher model
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_model(teacher_model, train_loader, teacher_optimizer, criterion, device, epochs)
    torch.save(teacher_model.state_dict(), 'pretrained_teacher.pth')

    print("Training student model")
    # Train student model
    student_optimizer = optim.Adam(student_model_pretrained.parameters(), lr=learning_rate)
    train_model(student_model_pretrained, train_loader, student_optimizer, criterion, device, epochs)
    torch.save(student_model_pretrained.state_dict(), 'pretrained_student.pth')

    print("Training kd student model")
    # Train KD student model
    kd_optimizer = optim.Adam(student_model_kd.parameters(), lr=learning_rate)
    train_kd(teacher_model, student_model_kd, train_loader, kd_optimizer, temperature, alpha, device, epochs)
    torch.save(student_model_kd.state_dict(), 'kd_student.pth')

if __name__ == "__main__":
    print("Start")
    main()

total_time = time.time() - start_time
print(f'{total_time=}')
