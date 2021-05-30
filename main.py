from dataclasses import dataclass
import cv2
import numpy as np
import requests
from PIL import Image 
import pickle

try:
    from reloading import reloading
except ImportError:
    
    # mock reloading without functionality, because almost everyone wont have it installd
    def reloading(*args, **kwargs):
        if len(args) > 0 or kwargs.get("forever"):
            fn_or_seq = kwargs.get("forever") or args[0]
            if isinstance(fn_or_seq, types.FunctionType):
                return fn_or_seq
            return fn_or_seq
        return update_wrapper(partial(reloading, **kwargs), reloading)


pixels_api_token = "YOUR API TOKEN HERE"

@dataclass
class Dimensions:
    width: int
    height: int

class PixelsClient:
    def __init__(self):
        self.base_url = "https://pixels.pythondiscord.com/"
        self.session = requests.Session()
        self.session.headers = {"Authorization": "Bearer " + pixels_api_token}
        self.reset = 10
        self.remaining = 5
        self.limit = 5
        self._dims = None

    def get_size(self, bypass=False):
        if not bypass and self._dims:
            return self._dims
        r = self.session.get(self.base_url + "get_size")
        self._dims = Dimensions(**r.json())
        return self._dims

    def get_pixels(self):
        r = self.session.get(self.base_url + "get_pixels")
        h = r.headers
        self.remaining = float(h.get("requests-remaining", 0))
        self.limit = float(h.get("Requests-Limit", 5))
        self.reset = float(h.get("requests-reset", 10))
        return r.content

    def get_image(self):
        size = self.get_size(bypass=True)
        img = Image.frombytes("RGB", (size.width, size.height), self.get_pixels())
        return np.array(img)


class DisplayHandler:
    def __init__(self):
        self.windowname = "laundmos' pixels history viewer"
        cv2.namedWindow(self.windowname)
        self.currentframe = None
        self.back = 40000
        self.images = []
        self.trackbar_val = 0
        self.scale = 6
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.do_jump_back = True
        self.client = PixelsClient()

    @classmethod
    def startup(cls):
        self = cls()
        cv2.createTrackbar(
            "frame-back", self.windowname, 0, self.back, self.handle_trackbar_change
        )
        cv2.setMouseCallback(self.windowname, self.handle_mouse_event)
        self.load_images()
        cv2.imshow(self.windowname, cv2.cvtColor(np.zeros((self.client.get_size().width, self.client.get_size().height, 3), np.uint8), cv2.COLOR_BGR2RGB))
        return self

    def load_images(self):
        try:
            with open("history_dump.pickle", "rb") as f:
                self.images = pickle.load(f)
        except FileNotFoundError:
            self.images = []

    def dump_images(self):
        with open("history_dump.pickle", "wb") as f:
            pickle.dump(self.images, f, protocol=pickle.HIGHEST_PROTOCOL)

    @reloading(every=500)
    def _window_size(self):
        img_shape = self.images[-1].shape
        return (img_shape[1] * self.scale, img_shape[0] * self.scale)

    @property
    def window_size(self):
        return self._window_size()

    @reloading
    def update(self):
        print("refreshed")
        try:
            img = self.client.get_image()
            self.images.append(img)
            self.images = self.images[-self.back:]
        except ValueError as e:
            print(e)
        cv2.setTrackbarMax("frame-back", self.windowname, len(self.images) - 1)
        if self.do_jump_back:
            self.handle_trackbar_change(0)
        if self.client.remaining < 1:
            for _ in range(1000 * round(self.client.reset + 1)):
                cv2.waitKey(1)
        else:
            for _ in range(1000 * round((self.client.reset / self.client.limit) + 1)):
                cv2.waitKey(1)
        self.dump_images()
    
    @reloading(every=500)
    def get_current_image(self):
        return self.images[-(1 + self.trackbar_val)]

    @reloading(every=500)
    def handle_trackbar_change(self, val):
        self.trackbar_val = val
        img = self.get_current_image()
        res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = cv2.resize(res, self.window_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.windowname, res)
        self.currentframe = res

    @reloading
    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_leftdown(x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.do_jump_back = not self.do_jump_back
            print(f"jump back toggled {self.do_jump_back}")

    @reloading
    def outlined_text(self, *args, **kwargs):
        cv2.putText(
            *args,
            (255, 255, 255),
            **{**kwargs, "lineType": cv2.LINE_AA, "thickness": 6},
        )
        cv2.putText(
            *args, (0, 0, 0), **{**kwargs, "lineType": cv2.LINE_AA, "thickness": 2}
        )

    @reloading(every=500)
    def put_text(self, image, x, y, texts, text_scale=0.8):
        t_size_x, t_size_y = 0, 0
        for text in texts:
            (size_x, size_y), retval = cv2.getTextSize(text, self.font, text_scale, 4)
            t_size_y += size_y
            t_size_x = max(size_x, t_size_x)
        if x + t_size_x > self.window_size[0]:
            x -= t_size_x
        if y - t_size_y < 0:
            y += t_size_y
        if y + t_size_y > self.window_size[1]:
            y -= t_size_y
        for i, text in enumerate(texts):
            self.outlined_text(
                image,
                text,
                (x, y + (i * 30)),
                self.font,
                text_scale,
            )

    @reloading(every=500)
    def mouse_leftdown(self, x, y):
        frame_copy = self.currentframe.copy()
        img_x, img_y = x // self.scale, y // self.scale
        img = self.get_current_image()
        color = img[img_y, img_x]
        text = f"({img_x}, {img_y})"
        self.put_text(frame_copy, x, y, [text, str(color)])
        cv2.imshow(self.windowname, frame_copy)

    def run(self):
        self.update()
        while cv2.getWindowProperty(self.windowname, 0) >= 0:
            self.update()


d = DisplayHandler.startup()
d.run()
