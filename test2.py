import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import customtkinter

num_classes = 3  # Assuming you have 3 classes: 'broken_glass', 'dents', and 'scratches'

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

# Create a label to display the predicted result
result_label = tk.Label(window, text="Prediction: ")
result_label.pack()

# Function to preprocess and predict the image
def predict_image():
    # Open a file dialog to select the image
    file_path = filedialog.askopenfilename()
    
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

# Run the main application loop
window.mainloop()

import tkinter
import tkinter.messagebox
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Damage Detection")
        self.geometry(f"{1100}x{580}")
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Damage Detection", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))
        self.textbox = customtkinter.CTkTextbox(self, width=10)
        self.textbox.grid(row=0, column=1, padx=(10, 10), pady=(10,10), sticky="nsew")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.textbox.insert("0.0", "This is a Vehicle Damage Detection Software created by us, using pytorch")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")


if __name__ == "__main__":
    app = App()
    app.mainloop()