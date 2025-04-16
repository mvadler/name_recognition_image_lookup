import tkinter as tk
from tkinter import scrolledtext, messagebox
from threading import Thread
from name_recognition_with_google_search import main

def update_status(message):
    status_label.config(text=message)
    status_label.update_idletasks()

def run_face_matching():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return

    # Clear status
    update_status("🔄 Starting search...")
    
    # Run the matching in a separate thread to keep the GUI responsive
    def threaded_main():
        main(text, status_callback=update_status)
        update_status("✅ Done searching and displaying montages!")

    Thread(target=threaded_main, daemon=True).start()

# GUI Setup
root = tk.Tk()
root.title("Face Recognition Montage Viewer")
root.geometry("600x400")

label = tk.Label(root, text="Enter text to extract names and search for faces:")
label.pack(pady=10)

input_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10)
input_box.pack(padx=10, pady=10)

submit_button = tk.Button(root, text="Generate Photo Montages", command=run_face_matching)
submit_button.pack(pady=10)

status_label = tk.Label(root, text="🟡 Waiting for input...", fg="blue")
status_label.pack(pady=5)

close_button = tk.Button(root, text="Close", command=root.destroy)
close_button.pack(pady=10)

# Run GUI loop
root.mainloop()
