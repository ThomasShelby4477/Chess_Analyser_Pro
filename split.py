import cv2
import os

def split_board(image_path, output_dir, piece_type):
    """
    Split a chess board image into individual squares and save them.
    
    Args:
        image_path: Path to the input chess board image
        output_dir: Directory to save the split squares
        piece_type: Label for the pieces (e.g., 'wP', 'bK', 'empty')
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return
        
    h, w = img.shape[:2]
    square_size = h // 8

    # Create directory for piece type if it doesn't exist
    piece_dir = os.path.join(output_dir, piece_type)
    if not os.path.exists(piece_dir):
        os.makedirs(piece_dir)

    # Split board into 8x8 squares
    for row in range(8):
        for col in range(8):
            y1 = row * square_size
            x1 = col * square_size
            square = img[y1:y1+square_size, x1:x1+square_size]
            cv2.imwrite(f"{piece_dir}/{row}_{col}.png", square)

# List of all chess piece types
piece_types = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'empty', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']

# Base directory containing your chess board images
base_dir = r"E:\Extras\Chess_AI\ss"
output_dir = "dataset"

# Process all piece types automatically
for piece_type in piece_types:
    image_path = os.path.join(base_dir, f"{piece_type}.png")
    print(f"Processing {piece_type}...")
    split_board(image_path, output_dir, piece_type)

print("All chess piece boards processed successfully!")
