import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Define a function to generate responses from the chatbot
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids=input_ids, max_length=1000, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Create a GUI window
root = tk.Tk()
root.title("Chatbot")

# Create a text widget to display the chat history
history_text = tk.Text(root, height=20, width=50)
history_text.pack()

# Create a label and entry widget for user input
input_label = tk.Label(root, text="User:")
input_label.pack(side=tk.LEFT)
input_entry = tk.Entry(root, width=50)
input_entry.pack(side=tk.LEFT)


# Define a function to handle user input
def handle_input():
    # Get the user input
    user_input = input_entry.get()
    # Add the user input to the chat history
    history_text.insert(tk.END, "User: " + user_input + "\n")
    # Generate a response from the chatbot
    response = generate_response(user_input)
    # Add the chatbot's response to the chat history
    history_text.insert(tk.END, "Chatbot: " + response + "\n")
    # Clear the user input
    input_entry.delete(0, tk.END)


# Create a button to submit user input
submit_button = tk.Button(root, text="Submit", command=handle_input)
submit_button.pack()

# Start the GUI main loop
root.mainloop()
