
import pyqtgraph as pg

class ClickableImageView(pg.ImageView):
    def __init__(self, *args, **kwargs):
        super(ClickableImageView, self).__init__(*args, **kwargs)
        self.selectedPixels = set()  # Keep track of selected pixels

    def mouseReleaseEvent(self, event):
        # Convert click position to image coordinates
        pos = self.getImageItem().mapFromScene(event.pos())
        x, y = int(pos.x()), int(pos.y())

        # Toggle pixel selection
        if (x, y) in self.selectedPixels:
            self.selectedPixels.remove((x, y))
        else:
            self.selectedPixels.add((x, y))

        # Update the image with the new selection
        self.updateImageWithSelection()

    def updateImageWithSelection(self):
        # Retrieve the current image data
        img_data = self.image

        # Modify pixels based on selection
        for x, y in self.selectedPixels:
            img_data[y, x] = [0, 0, 255]  # Set selected pixels to blue

        # Update the displayed image
        self.setImage(img_data)
