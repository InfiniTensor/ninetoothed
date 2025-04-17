import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size


def visualize(tensor, color=None, save_path=None):
    """Visualize a tensor as a structured grid representation.

    :param tensor: The tensor to be visualized.
    :param color: The color to be used for visualization.
    :param save_path: The path where the visualization should be saved.
    """

    if color is None:
        color = f"C{visualize.count}"

    _, max_pos_x, max_pos_y = _visualize_tensor(plt.gca(), tensor, 0, 0, color)

    width = max_pos_y + 1
    height = max_pos_x + 1

    _, ax = _prepare_figure_and_axes(width, height)

    _visualize_tensor(ax, tensor, 0, 0, color)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)

    plt.close()

    visualize.count += 1


visualize.count = 0


def _prepare_figure_and_axes(width, height):
    outline_width = 0.1
    plt.rcParams["lines.linewidth"] = 72 * outline_width

    fig = plt.figure(figsize=(width + outline_width, height + outline_width))

    h = (Size.Fixed(0), Size.Fixed(width + outline_width))
    v = (Size.Fixed(0), Size.Fixed(height + outline_width))

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    ax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()

    plt.axis("off")

    half_outline_width = outline_width / 2
    plt.xlim((-half_outline_width, width + half_outline_width))
    plt.ylim((-half_outline_width, height + half_outline_width))

    return fig, ax


def _visualize_tensor(ax, tensor, x, y, color, level_spacing=4):
    verts = _visualize_level(ax, tensor, x, y, color)

    if tensor.dtype is None:
        return verts, verts[1][1][0], verts[1][1][1]

    next_x, next_y = verts[0][1]
    next_y += level_spacing + 1

    next_verts, max_pos_x, max_pos_y = _visualize_tensor(
        ax, tensor.dtype, next_x, next_y, color
    )

    conn_verts = _verts_of_rect(1, level_spacing, next_x, next_y - level_spacing)
    conn_verts = [list(vert) for vert in conn_verts]
    conn_verts[2][0] += next_verts[1][0][0]

    pos_y, pos_x = zip(*conn_verts)
    pos_x = pos_x + (pos_x[0],)
    pos_y = pos_y + (pos_y[0],)

    ax.plot(pos_x[1:3], pos_y[1:3], "k--")
    ax.plot(pos_x[3:5], pos_y[3:5], "k--")

    max_pos_x = max(max_pos_x, verts[1][1][0])
    max_pos_y = max(max_pos_y, verts[1][1][1])

    return verts, max_pos_x, max_pos_y


def _visualize_level(ax, level, x, y, color):
    offsets = [1 for _ in range(level.ndim)]

    for dim in range(-3, -level.ndim - 1, -1):
        offsets[dim] = offsets[dim + 2] * level.shape[dim + 2] + 1

    indices = np.indices(level.shape)
    flattened_indices = np.stack(
        [indices[i].flatten() for i in range(level.ndim)], axis=-1
    )

    max_pos_x = x
    max_pos_y = y

    for indices in flattened_indices:
        pos = [x, y]

        for dim, index in enumerate(indices):
            pos[(level.ndim - dim) % 2] += index * offsets[dim]

        max_pos_x = max(max_pos_x, pos[0])
        max_pos_y = max(max_pos_y, pos[1])

        _visualize_unit_square(ax, pos[1], pos[0], color)

    verts = (((x, y), (x, max_pos_y)), ((max_pos_x, y), (max_pos_x, max_pos_y)))

    return verts


def _visualize_unit_square(ax, x, y, color):
    _visualize_rect(ax, 1, 1, x, y, color)


def _visualize_rect(ax, width, height, x, y, color):
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            width,
            height,
            edgecolor="k",
            facecolor=color,
            linewidth=plt.rcParams["lines.linewidth"],
        )
    )


def _verts_of_rect(width, height, x, y):
    return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
