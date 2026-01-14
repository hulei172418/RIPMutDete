import os
import json
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# Extract code content from a JSON file
def extract_code(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        origin = data.get("origin", {}).get("item", {}).get("content", {}).get("item", "")
        mutated = data.get("mutated", {}).get("item", {}).get("content", {}).get("item", "")
        return origin.strip(), mutated.strip()
    except Exception as e:
        return f"Failed to read: {e}", f"Failed to read: {e}"

# Read path pairs from Excel
def load_path_pairs(excel_path):
    df = pd.read_excel(excel_path)
    if "original_graph_path" not in df or "mutant_graph_path" not in df:
        raise ValueError("The Excel file must contain 'original_graph_path' and 'mutant_graph_path' columns")
    return df[["original_graph_path", "mutant_graph_path", "equiv_label"]].values.tolist()

# Create the GUI application
def launch_gui(path_pairs):
    root = tk.Tk()
    root.title("Original Program vs Mutant - Equivalence Comparison")

    index = tk.IntVar(value=0)

    origin_view = scrolledtext.ScrolledText(root, height=50, width=100)
    origin_view.grid(row=1, column=0, padx=5, pady=5)

    mutant_view = scrolledtext.ScrolledText(root, height=50, width=100)
    mutant_view.grid(row=1, column=1, padx=5, pady=5)

    label_text = tk.StringVar()
    label_display = tk.Label(root, textvariable=label_text, font=("Arial", 14), fg="blue")
    label_display.grid(row=2, column=0, columnspan=2, pady=10)

    tk.Label(root, text="Original Code").grid(row=0, column=0)
    tk.Label(root, text="Mutated Code").grid(row=0, column=1)

    def show_pair(i):
        og_path, mut_path, equiv_label = path_pairs[i]
        og_json = os.path.join(og_path, "output.json")
        mut_json = os.path.join(mut_path, "output.json")

        og_code, mut_code = extract_code(mut_json)

        origin_view.delete("1.0", tk.END)
        origin_view.insert(tk.END, og_code)

        mutant_view.delete("1.0", tk.END)
        mutant_view.insert(tk.END, mut_code)
        label_text.set(f"Equivalence Label equiv_label: {equiv_label}")

        root.title(f"Comparison View - Pair {i + 1} / {len(path_pairs)}")

    def prev():
        if index.get() > 0:
            index.set(index.get() - 1)
            show_pair(index.get())

    def next_():
        if index.get() < len(path_pairs) - 1:
            index.set(index.get() + 1)
            show_pair(index.get())

    tk.Button(root, text="<< Previous", command=prev).grid(row=2, column=0, sticky="w", padx=10, pady=5)
    tk.Button(root, text="Next >>", command=next_).grid(row=2, column=1, sticky="e", padx=10, pady=5)

    show_pair(index.get())
    root.mainloop()

# Main entry point
def main():
    excel_path = filedialog.askopenfilename(
        title="Select the Excel file containing 'original_graph_path' and 'mutant_graph_path'",
        filetypes=[("Excel 文件", "*.xlsx *.xls")]
    )

    if not excel_path:
        messagebox.showwarning("No File Selected", "You did not select an Excel file.")
        return

    try:
        path_pairs = load_path_pairs(excel_path)
        launch_gui(path_pairs)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load: {e}")

if __name__ == "__main__":
    main()
