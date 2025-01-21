import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_models import TeacherModel, StudentModel

# Function to evaluate a model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# Main function to test models
def main():
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data")
    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Loading models")
    # Load models
    teacher_model = TeacherModel().to(device)
    student_model_pretrained = StudentModel().to(device)
    student_model_kd = StudentModel().to(device)

    teacher_model.load_state_dict(torch.load('pretrained_teacher.pth', weights_only=True))
    student_model_pretrained.load_state_dict(torch.load('pretrained_student.pth', weights_only=True))
    student_model_kd.load_state_dict(torch.load('kd_student.pth', weights_only=True))

    print("Evaluating teacher model")
    evaluate_model(teacher_model, test_loader, device)

    print("Evaluating pretrained student model")
    evaluate_model(student_model_pretrained, test_loader, device)

    print("Evaluating KD student model")
    evaluate_model(student_model_kd, test_loader, device)

if __name__ == "__main__":
    print("Start testing")
    main()
