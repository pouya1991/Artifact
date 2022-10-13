import pytest

from submodule_cv import SlideCoordsExtractor

class MockOpenSlide(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    @property
    def dimensions(self):
        return (self.width, self.height,)

def test_SlideCoordsExtractor():
    tile_x = 3
    tile_y = 4
    patch_size = 1024
    width = patch_size * tile_x
    height = patch_size * tile_y
    os_slide = MockOpenSlide(width, height)
    sce = SlideCoordsExtractor(os_slide, patch_size)
    assert len(sce) == tile_x * tile_y
    assert next(sce) == (0, 0, 0 * patch_size, 0 * patch_size)
    assert next(sce) == (1, 0, 1 * patch_size, 0 * patch_size)
    assert next(sce) == (2, 0, 2 * patch_size, 0 * patch_size)
    assert next(sce) == (0, 1, 0 * patch_size, 1 * patch_size)
    assert next(sce) == (1, 1, 1 * patch_size, 1 * patch_size)
    assert next(sce) == (2, 1, 2 * patch_size, 1 * patch_size)
    assert next(sce) == (0, 2, 0 * patch_size, 2 * patch_size)
    assert next(sce) == (1, 2, 1 * patch_size, 2 * patch_size)
    assert next(sce) == (2, 2, 2 * patch_size, 2 * patch_size)
    assert next(sce) == (0, 3, 0 * patch_size, 3 * patch_size)
    assert next(sce) == (1, 3, 1 * patch_size, 3 * patch_size)
    assert next(sce) == (2, 3, 2 * patch_size, 3 * patch_size)
