"utility functions for working with czi files"
__all__ = ['get_czi_shape_info', 'build_index']


def get_czi_shape_info(czi_file):
    """get_czi_shape_info

    :param czi_file:
    """
    shape = czi_file.shape
    axes_dict = {axis: idx for idx, axis in enumerate(czi_file.axes)}
    shape_dict = {axis: shape[axes_dict[axis]] for axis in czi_file.axes}
    return axes_dict, shape_dict


def build_index(axes, ix_select):
    """build_index

    :param axes:
    :param ix_select:
    """
    idx = [ix_select.get(ax, 0) for ax in axes]
    return tuple(idx)


def is_movie(czi_file):
    axes, shape = get_czi_shape_info(czi_file)
    times = axes.get('T', 1)
    return times > 1


def has_depth(czi_file):
    axes, shape = get_czi_shape_info(czi_file)
    has_depth = axes.get('Z', 1)
    return depth > 1
