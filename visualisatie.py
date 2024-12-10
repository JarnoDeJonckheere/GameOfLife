import tkinter as tk

def create_grid(canvas, width, height, cell_width, cell_height, colored_cells):
    """
    Create a height x width grid on the given canvas.
    :param canvas: Tkinter Canvas widget
    :param cell_size: Size of each cell in the grid
    :param colored_cells: List of (row, column) tuples indicating which cells to color
    """
    for i in range(height):
        for j in range(width):
            x1, y1 = j * cell_width, i * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            if (i, j) in colored_cells:
                canvas.create_rectangle(x1, y1, x2, y2, fill="blue", outline="black")
            else:
                canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")

def update_grid(canvas, width, height, cell_width, cell_height, colored_cells):
    """
    Update the grid with the new colored cells.
    """
    canvas.delete("all")  # Clear the canvas
    create_grid(canvas, width, height, cell_width, cell_height, colored_cells)  # Redraw the grid

def process_lines(canvas, lines, width, height, cell_width, cell_height, delay=50):
    """
    Process each line and update the grid at regular intervals.
    """
    def update_line(i):
        if i < len(lines):
            line_values = list(map(int, filter(None, lines[i].strip().split(";"))))
            positions = [index for index, value in enumerate(line_values) if value == 1]  # Get indices where value is 1
            position_tuples = [(pos // width, pos % width) for pos in positions]
            update_grid(canvas, width, height, cell_width, cell_height, position_tuples)
            # Schedule the next update
            root.after(delay, update_line, i + 1)

    update_line(0)  # Start the recursive updates

width = 0
height = 0

root = tk.Tk()
root.title("Game of Life")

# Open the file
with open("Nostride_Synchronous/outPut_50x50_seed42.txt", "r") as file:
    next(file)  # Skip the first line

    second_line = file.readline().strip()  # Read the second line
    parts = second_line.split("grid_")  # Split into parts based on 'grid_'
    for part in parts:
        if "width;" in part:
            width = int(part.replace("width;", ""))  # Extract the width value
        elif "height;" in part:
            height = int(part.replace("height;", ""))  # Extract the height value
    print(f"grid_height = {height}, grid_width = {width}")

    cell_height = (root.winfo_screenheight() - root.winfo_rooty() - 200) / height
    cell_width = (root.winfo_screenwidth() - 200) / width

    canvas = tk.Canvas(root, width=root.winfo_screenwidth() - 200, height=root.winfo_screenheight() - root.winfo_rooty() - 200)
    canvas.pack()

    create_grid(canvas, width, height, cell_width, cell_height, [])

    lines = file.readlines()[:-1]  # Skip last line
    process_lines(canvas, lines, width, height, cell_width, cell_height)  # Start processing lines

root.mainloop()
