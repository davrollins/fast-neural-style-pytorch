import cv2
import transformer
import torch
import utils

# Prompt for Style Transformer Model and set video window top bar name
pth=input( " (1) Bayanihan \n (2) Lazy \n (3) Mosaic \n (4) Starry Night \n (5) Tokyo Ghoul \n (6) Udnie \n (7) Wave \n Enter Style: " )
if pth in ("1", "Bayanihan", "bayanian"):
   STYLE_TRANSFORM_PATH = ("transforms/bayanihan.pth")
   DEMO = ("Bayanihan Style Transformation Demo")
elif pth in ("2", "Lazy", "lazy"):
   STYLE_TRANSFORM_PATH = ("transforms/lazy.pth")
   DEMO = ("Lazy Style Transformation Demo")
elif pth in ("3", "Mosaic", "mosaic"):
   STYLE_TRANSFORM_PATH = ("transforms/mosaic.pth")
   DEMO = ("Mosaic Style Transformation Demo")
elif pth in ("4", "Starry", "starry"):
   STYLE_TRANSFORM_PATH = ("transforms/starry.pth")
   DEMO = ("Starry Night Style Transformation Demo")
elif pth in ("5", "Tokyo_Ghoul", "tokyo_ghoul", "tokyo", "Tokyo"):
   STYLE_TRANSFORM_PATH = ("transforms/tokyo_ghoul.pth")
   DEMO = ("Tokyo_Ghoul Style Transformation Demo")
elif pth in ("6", "Udnie", "udnie"):
   STYLE_TRANSFORM_PATH = ("transforms/udnie.pth")
   DEMO = ("Udnie Style Transformation Demo")
elif pth in ("7", "Wave", "wave"):
   STYLE_TRANSFORM_PATH = ("transforms/wave.pth")
   DEMO = ("The Great Wave off Kanagawa Style Transformation Demo")
PRESERVE_COLOR = False
#WIDTH = 3440
#HEIGHT = 1440
WIDTH = 1280
HEIGHT = 720

def webcam(style_transform_path, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    print("Loading Transformer Network")
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_transform_path))
    net = net.to(device)
    print("Done Loading Transformer Network")

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)

    # Main loop
    with torch.no_grad():
        while True:
            # Get webcam input
            ret_val, img = cam.read()

            # Mirror 
            img = cv2.flip(img, 1)

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            
            # Generate image
            content_tensor = utils.itot(img).to(device)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR):
                generated_image = utils.transfer_color(img, generated_image)

            generated_image = generated_image / 255

            # Show webcam
            cv2.imshow(DEMO, generated_image)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
            
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()

webcam(STYLE_TRANSFORM_PATH, WIDTH, HEIGHT)
