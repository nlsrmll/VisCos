import os

from packages.utils.settings import settings

AvailableImages = sorted(os.listdir(settings.IMAGE_BASE_PATH))

AvailableImages_Dir = {os.path.splitext(image)[0]: image for image in AvailableImages}
