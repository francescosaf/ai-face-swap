"""Template di allineamento del viso per i modelli a 256px.

L'insightface installato in dlfolder e' troppo vecchio e non espone i
template. HyperSwap e HIFiFace usano il template 'arcface_128': NON e' un
template a 128 punti, ma i 5 landmark del volto (occhi, naso, angoli
bocca) tarati su un crop 256, espressi in 0..1 e da moltiplicare per la
dimensione del crop.

 Fonte: facefusion/face_helper.py (FaceFusion), licenza MIT.
"""

import numpy as np

ARCFACE_128 = np.asarray(
[
    [0.36167657, 0.40387735],
    [0.63696718, 0.40235469],
    [0.50019687, 0.56044221],
    [0.38710392, 0.72160548],
    [0.61507732, 0.72034454],
],
    dtype=np.float32,
)


def warp_template(size: int) -> np.ndarray:
    """I 5 template ArcFace portati al sistema di coordinate del crop."""
    return ARCFACE_128 * float(size)
