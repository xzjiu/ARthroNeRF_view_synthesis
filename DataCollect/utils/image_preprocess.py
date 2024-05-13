import cv2


def check_image_is_vague(image, threshold=100):
    """
    :param image: image path
    :param threshold: 
    :return: True or False
    """
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    print('image vague is {}'.format(fm))
    if fm > threshold:
        return True
    return False