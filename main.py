import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

global image_v

num_classes = 3  # Assuming you have 3 classes: 'broken_glass', 'dents', and 'scratches'

# Create an instance of the resnet18 model
model = resnet18(pretrained="pretrained")

# Modify the last fully connected layer to match the number of classes
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

# Print the modified model architecture
print(model)

class CarIssueClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CarIssueClassifier, self).__init__()
        self.model = resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Define the class names (optional, if you want to map class indices to class names)
class_names = ['broken_glass', 'dents', 'scratches']

# Create the main application window
window = tk.Tk()
window.title("Image Classification App")
window.geometry("800x600")

# Create a label to display the predicted result
result_label = tk.Label(window, text="Prediction: ")
result_label.grid(row=3, column=0)
result_label.pack()

# Function to preprocess and predict the image
def predict_image():
    # Open a file dialog to select the image
    file_path = filedialog.askopenfilename()

    image_v = file_path
    # Preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(file_path)
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Load the trained model
    model = CarIssueClassifier(num_classes)
    model.load_state_dict(torch.load("/home/sahan/Desktop/VehicleDamageDetectionSoftware/model.pth"))
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = class_names[predicted_idx.item()]

    # Update the result label
    result_label.config(text="Prediction: " + predicted_label)

# Create a button to browse and predict the image
browse_button = tk.Button(window, text="Browse", command=predict_image)
browse_button.pack()


    # Convert the image to Tkinter-compatible format
# tk_image = ImageTk.PhotoImage(image_v)

#     # Create a Tkinter label and display the image
# image_label = tk.Label(window, image=tk_image)
# image_label.pack()

# Run the main application loop
window.mainloop()
