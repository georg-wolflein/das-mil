from .device import DEVICE, device, detach


def human_format(num: float) -> str:
    """Display a large number in a human readable format. https://stackoverflow.com/a/45846841"""
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
