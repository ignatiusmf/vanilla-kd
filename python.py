import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Define a simple teacher and student model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Train a model
def train_model(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)  # Flatten the images

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """Compute the distillation loss."""
    soft_teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    
    # Soft loss (distilled knowledge)
    soft_loss = nn.functional.kl_div(soft_student_probs, soft_teacher_probs, reduction='batchmean') * (temperature ** 2)

    # Hard loss (ground truth labels)
    hard_loss = nn.functional.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss

# Training loop for KD
def train_student(teacher_model, student_model, train_loader, optimizer, temperature, alpha, device):
    student_model.train()
    teacher_model.eval()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), -1)  # Flatten the images

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)
        loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

# Main function
def main():
    # Configurations
    batch_size = 64
    learning_rate = 0.001
    temperature = 5.0
    alpha = 0.7
    epochs_pretrain = 3
    epochs_kd = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Initializing models...")
    # Initialize models
    teacher_model = TeacherModel().to(device)
    student_model_pretrained = StudentModel().to(device)
    student_model_kd = StudentModel().to(device)

    print("Training teacher model...")
    # Train and save the teacher model
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_model(teacher_model, train_loader, teacher_optimizer, criterion, epochs_pretrain, device)
    torch.save(teacher_model.state_dict(), 'pretrained_teacher.pth')
    print("Teacher model trained and saved as 'pretrained_teacher.pth'.")

    print("Training student model (pretraining)...")
    # Train and save the pretrained student model
    student_optimizer = optim.Adam(student_model_pretrained.parameters(), lr=learning_rate)
    train_model(student_model_pretrained, train_loader, student_optimizer, criterion, epochs_pretrain, device)
    torch.save(student_model_pretrained.state_dict(), 'pretrained_student.pth')
    print("Student model pretrained and saved as 'pretrained_student.pth'.")

    print("Training KD student model...")
    # Train the KD student model
    kd_optimizer = optim.Adam(student_model_kd.parameters(), lr=learning_rate)
    for epoch in range(epochs_kd):
        print(f"Epoch {epoch + 1}/{epochs_kd}:")
        train_student(teacher_model, student_model_kd, train_loader, kd_optimizer, temperature, alpha, device)
        print(f"Epoch {epoch + 1}/{epochs_kd} complete.")

    torch.save(student_model_kd.state_dict(), 'kd_student.pth')
    print("KD student model trained and saved as 'kd_student.pth'.")

if __name__ == "__main__":
    print("Start")
    main()
