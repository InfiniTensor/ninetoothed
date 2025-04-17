import matplotlib.pyplot as plt

from ninetoothed import Tensor
from ninetoothed.visualization import (
    _prepare_figure_and_axes,
    _visualize_tensor,
    visualize,
)


def visualize_x_tiled(m, n, block_size_m, block_size_n):
    BLOCK_SIZE_M = block_size_m
    BLOCK_SIZE_N = block_size_n

    x = Tensor(shape=(m, n))
    x_tiled = x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    visualize(x_tiled, color="C0", save_path="generated/x-tiled.png")


def visualize_add(size, block_size):
    BLOCK_SIZE = block_size

    x = Tensor(shape=(size,))
    y = Tensor(shape=(size,))
    z = Tensor(shape=(size,))

    x_arranged = x.tile((BLOCK_SIZE,))
    visualize(x_arranged, color="C0", save_path="generated/x-arranged.png")

    y_arranged = y.tile((BLOCK_SIZE,))
    visualize(y_arranged, color="C1", save_path="generated/y-arranged.png")

    z_arranged = z.tile((BLOCK_SIZE,))
    visualize(z_arranged, color="C2", save_path="generated/z-arranged.png")


def visualize_tiled_matrix_multiplication(
    m, n, k, block_size_m, block_size_n, block_size_k
):
    BLOCK_SIZE_M = block_size_m
    BLOCK_SIZE_N = block_size_n
    BLOCK_SIZE_K = block_size_k

    a = Tensor(shape=(m, k))
    b = Tensor(shape=(k, n))
    c = Tensor(shape=(m, n))

    a_tiled = a.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_tiled = b.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    c_tiled = c.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    a_tile = a_tiled.innermost()
    b_tile = b_tiled.innermost()
    c_tile = c_tiled.innermost()

    color_0 = "#1f77b4"
    color_1 = "#ff7f0e"
    color_2 = "#2ca02c"

    a_color = _lighten_color(color_0, 20)
    b_color = _lighten_color(color_1, 20)
    c_color = _lighten_color(color_2, 20)

    a_tile_base_color = color_0
    b_tile_base_color = color_1
    c_tile_base_color = color_2

    def _visualize_matrices(ax):
        a_verts, a_max_pos_x, a_max_pos_y = _visualize_tensor(ax, a, 0, 0, a_color)
        b_verts, b_max_pos_x, b_max_pos_y = _visualize_tensor(
            ax, b, a_max_pos_x + 2, a_max_pos_y + 2, b_color
        )
        c_verts, _, _ = _visualize_tensor(ax, c, 0, a_max_pos_y + 2, c_color)

        a_min_pos_x, a_min_pos_y = a_verts[0][0]
        b_min_pos_x, b_min_pos_y = b_verts[0][0]
        c_min_pos_x, c_min_pos_y = c_verts[0][0]

        percentage = 30

        for i in range(0, a_tiled.shape[1]):
            y_offset = i * a_tile.shape[1]

            x = a_min_pos_x + (a_tiled.shape[0] - 2) * a_tile.shape[0]
            y = a_min_pos_y + y_offset

            darkened_color = _darken_color(
                a_tile_base_color, (i + 1) / a_tiled.shape[1] * percentage
            )

            _visualize_tensor(ax, a_tile, x, y, darkened_color)

        for i in range(0, b_tiled.shape[0]):
            x_offset = i * b_tile.shape[0]

            x = b_min_pos_x + x_offset
            y = b_min_pos_y + b_tile.shape[1]

            darkened_color = _darken_color(
                b_tile_base_color,
                (b_tiled.shape[0] - i) / b_tiled.shape[0] * percentage,
            )

            _visualize_tensor(ax, b_tile, x, y, darkened_color)

        _visualize_tensor(
            ax,
            c_tile,
            c_min_pos_x + (a_tiled.shape[0] - 2) * a_tile.shape[0],
            c_min_pos_y + b_tile.shape[1],
            _darken_color(c_tile_base_color, percentage),
        )

        return b_max_pos_x, b_max_pos_y

    max_pos_x, max_pos_y = _visualize_matrices(plt.gca())

    width = max_pos_y + 1
    height = max_pos_x + 1

    _, ax = _prepare_figure_and_axes(width, height)

    _visualize_matrices(ax)

    save_path = "generated/tiled-matrix-multiplication.png"
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)

    plt.close()


def visualize_mm(m, n, k, block_size_m, block_size_n, block_size_k):
    BLOCK_SIZE_M = block_size_m
    BLOCK_SIZE_N = block_size_n
    BLOCK_SIZE_K = block_size_k

    input = Tensor(shape=(m, k))
    other = Tensor(shape=(k, n))
    output = Tensor(shape=(m, n))

    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    visualize(output_arranged, color="C2", save_path="generated/output-arranged-0.png")

    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    visualize(input_arranged, color="C0", save_path="generated/input-arranged-0.png")
    input_arranged = input_arranged.tile((1, -1))
    visualize(input_arranged, color="C0", save_path="generated/input-arranged-1.png")
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    visualize(input_arranged, color="C0", save_path="generated/input-arranged-2.png")
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    visualize(input_arranged, color="C0", save_path="generated/input-arranged-3.png")

    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    visualize(other_arranged, color="C1", save_path="generated/other-arranged-0.png")
    other_arranged = other_arranged.tile((-1, 1))
    visualize(other_arranged, color="C1", save_path="generated/other-arranged-1.png")
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    visualize(other_arranged, color="C1", save_path="generated/other-arranged-2.png")
    other_arranged.dtype = other_arranged.dtype.squeeze(1)
    visualize(other_arranged, color="C1", save_path="generated/other-arranged-3.png")


def _darken_color(hex_color, percentage):
    rgb = _hex_to_rgb(hex_color)
    factor = (100 - percentage) / 100
    darkened_rgb = tuple(int(c * factor) for c in rgb)

    return _rgb_to_hex(darkened_rgb)


def _lighten_color(hex_color, percentage):
    rgb = _hex_to_rgb(hex_color)
    factor = percentage / 100
    lightened_rgb = tuple(int(c + (255 - c) * factor) for c in rgb)

    return _rgb_to_hex(lightened_rgb)


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")

    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)

    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(*rgb_color)
