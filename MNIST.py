import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from sklearn.datasets import fetch_openml
import pickle
import NeuralNetwork as NN
from sklearn.model_selection import train_test_split

# Convert number to one-hot encoding
def num_to_target(num):
    target_data = np.zeros(10)
    target_data[num] = 1
    return target_data

# Convert softmax output to class label
def target_to_num(results):
    return np.argmax(results)

nn = NN.NeuralNetwork(784, 1024, 10, 0.05)
print("Initialized new model.")

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist.data / 255.0, mnist.target.astype(int)  # Normalize
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train in batches
batch_size = 128
for epoch in range(5):  # Run multiple epochs
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i + batch_size]
        batch_y = [num_to_target(y) for y in y_train[i:i + batch_size]]

        for j in range(len(batch_x)):
            nn.train(batch_x[j], batch_y[j])

# Save trained model
with open("trained_model1.pkl", "wb") as f:
    pickle.dump(nn, f)
print("Model saved.")

# Evaluate accuracy
correct_answers = sum(target_to_num(nn.feed_forward(x_test[i])) == y_test[i] for i in range(len(x_test)))
accuracy = (correct_answers / len(x_test)) * 100
print(f"Final Test Accuracy: {accuracy:.2f}%")

# GUI for visualization
# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

index = np.random.randint(len(y_test))

# Display initial image
img = ax.imshow(x_test[index].reshape(28, 28), cmap='gray')
title = ax.set_title(f"Label: {y_test[index]} \n AI thinks it's the number {target_to_num(nn.feed_forward(x_test[index]))}")
ax.axis('off')

# Button callback functions
def next_image(event):
    global index
    index = min(len(x_test)-1,np.random.randint(len(x_test)))
    update_image()

def prev_image(event):
    global index
    index = max(0,np.random.randint(len(x_test)))
    update_image()

def update_image():
    img.set_data(x_test[index].reshape(28, 28))
    title.set_text(f"Label: {y_test[index]} \n AI thinks it's a {target_to_num(nn.feed_forward(x_test[index]))}")
    plt.draw()

def print_network(event):
    print(nn.print_parameters())

# Add buttons
axprev = plt.axes([0.1, 0.05, 0.15, 0.075])  # Position of "Previous" button
axnext = plt.axes([0.75, 0.05, 0.15, 0.075])  # Position of "Next" button

axprintnetwork = plt.axes([0.425,0.05,0.15,0.075]) # print network button (useless now)

bnext = Button(axnext, 'Next')
bnext.on_clicked(next_image)

bprev = Button(axprev, 'Previous')
bprev.on_clicked(prev_image)

bnprintnetwork = Button(axprintnetwork, 'print network')
bnprintnetwork.on_clicked(print_network)


plt.show()
