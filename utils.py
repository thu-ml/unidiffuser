from absl import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]  # center crop
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)  # resize the center crop from [crop, crop] to [width, height]

    return np.array(img).astype(np.uint8)


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def get_nnet(name, **kwargs):
    if name == 'uvit_multi_post_ln':
        from libs.uvit_multi_post_ln import UViT
        return UViT(**kwargs)
    elif name == 'uvit_multi_post_ln_v1':
        from libs.uvit_multi_post_ln_v1 import UViT
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


def drawRoundRec(draw, color, x, y, w, h, r):
    drawObject = draw

    '''Rounds'''
    drawObject.ellipse((x, y, x + r, y + r), fill=color)
    drawObject.ellipse((x + w - r, y, x + w, y + r), fill=color)
    drawObject.ellipse((x, y + h - r, x + r, y + h), fill=color)
    drawObject.ellipse((x + w - r, y + h - r, x + w, y + h), fill=color)

    '''rec.s'''
    drawObject.rectangle((x + r / 2, y, x + w - (r / 2), y + h), fill=color)
    drawObject.rectangle((x, y + r / 2, x + w, y + h - (r / 2)), fill=color)


def add_water(img, text='UniDiffuser', pos=3):
    width, height = img.size
    scale = 4
    scale_size = 0.5
    img = img.resize((width * scale, height * scale), Image.LANCZOS)
    result = Image.new(img.mode, (width * scale, height * scale), color=(255, 255, 255))
    result.paste(img, box=(0, 0))

    delta_w = int(width * scale * 0.27 * scale_size)  # text width
    delta_h = width * scale * 0.05 * scale_size  # text height
    postions = np.array([[0, 0], [0, height * scale - delta_h], [width * scale - delta_w, 0],
                         [width * scale - delta_w, height * scale - delta_h]])
    postion = postions[pos]
    # 文本
    draw = ImageDraw.Draw(result)
    fillColor = (107, 92, 231)
    setFont = ImageFont.truetype("assets/ArialBoldMT.ttf", int(width * scale * 0.05 * scale_size))
    delta = 20 * scale_size
    padding = 15 * scale_size
    drawRoundRec(draw, (223, 230, 233), postion[0] - delta - padding, postion[1] - delta - padding,
                 w=delta_w + 2 * padding, h=delta_h + 2 * padding, r=50 * scale_size)
    draw.text((postion[0] - delta, postion[1] - delta), text, font=setFont, fill=fillColor)

    return result.resize((width, height), Image.LANCZOS)
