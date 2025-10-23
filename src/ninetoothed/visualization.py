import ast
import inspect
import itertools
import tkinter
from tkinter import ttk

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import Divider, Size

from ninetoothed.debugging import simulate_arrangement


def visualize(tensor, color=None, save_path=None):
    """Visualize a tensor as a structured grid representation.

    :param tensor: The tensor to be visualized.
    :param color: The color to be used for visualization.
    :param save_path: The path where the visualization should be saved.
    """

    if color is None:
        color = f"C{visualize.count}"

    _, max_pos_x, max_pos_y = _visualize_tensor(plt.gca(), tensor, 0, 0, color)

    plt.close()

    width = max_pos_y + 1
    height = max_pos_x + 1

    _, ax = _prepare_figure_and_axes(width, height)

    _visualize_tensor(ax, tensor, 0, 0, color)

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)

        plt.close()
    else:
        plt.show()

    visualize.count += 1


visualize.count = 0


def visualize_arrangement(arrangement, tensors):
    """Visualize the arrangement of the tensors.

    :param arrangement: The arrangement of the tensors.
    :param tensors: The tensors.
    """

    source_tensors, target_tensors = simulate_arrangement(arrangement, tensors)

    param_names = inspect.signature(arrangement).parameters.keys()
    tensor_names = tuple(param_names)[: len(source_tensors)]

    root = tkinter.Tk()
    root.title("Arrangement Simulation")

    main_frame = ttk.Frame(root)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    canvas = FigureCanvasTkAgg(Figure(figsize=(6.4, 3)), main_frame)
    control_frame = ttk.Frame(main_frame)

    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    entry_width = 10

    tensor_name_label = ttk.Label(control_frame, text="Tensor Name")
    tensor_name_combo_box = ttk.Combobox(
        control_frame, values=tensor_names, width=entry_width
    )
    source_shape_label = ttk.Label(control_frame, text="Source Shape")
    source_shape_entry = ttk.Entry(control_frame, width=entry_width)
    source_subscript_label = ttk.Label(control_frame, text="Source Subscript")
    source_subscript_entry = ttk.Entry(control_frame, width=entry_width)
    num_programs_label = ttk.Label(control_frame, text="Number of Programs")
    num_programs_entry = ttk.Entry(control_frame, width=entry_width)
    program_id_label = ttk.Label(control_frame, text="Program ID")
    program_id_entry = ttk.Entry(control_frame, width=entry_width)
    target_shape_label = ttk.Label(control_frame, text="Target Shape")
    target_shape_entry = ttk.Entry(control_frame, width=entry_width)
    target_subscript_label = ttk.Label(control_frame, text="Target Subscript")
    target_subscript_entry = ttk.Entry(control_frame, width=entry_width)

    main_frame.grid(
        column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S)
    )

    canvas.get_tk_widget().grid(
        column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S)
    )
    control_frame.grid(
        column=1, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S)
    )

    for row, (label, widget) in enumerate(
        (
            (tensor_name_label, tensor_name_combo_box),
            (source_shape_label, source_shape_entry),
            (source_subscript_label, source_subscript_entry),
            (num_programs_label, num_programs_entry),
            (program_id_label, program_id_entry),
            (target_shape_label, target_shape_entry),
            (target_subscript_label, target_subscript_entry),
        )
    ):
        label.grid(column=0, row=row, sticky=(tkinter.W, tkinter.E))
        widget.grid(column=1, row=row, sticky=(tkinter.W, tkinter.E))

    for child in itertools.chain(
        main_frame.winfo_children(), control_frame.winfo_children()
    ):
        child.grid_configure(padx=5, pady=5)

    def _update_read_only_entry(entry, text):
        entry.configure(state="active")
        entry.delete(0, tkinter.END)
        entry.insert(tkinter.END, text)
        entry.configure(state="readonly")

    def _init():
        source_tensor = source_tensors[0]
        target_tensor = target_tensors[0]

        tensor_name_combo_box.set(tensor_names[0])
        source_subscript_entry.insert(
            tkinter.END, ", ".join(("0",) * (source_tensor.ndim - 3) + ("...",))
        )
        _update_read_only_entry(num_programs_entry, str(target_tensor.shape[0]))
        program_id_entry.insert(tkinter.END, "0")
        target_subscript_entry.insert(
            tkinter.END, ", ".join(("0",) * (target_tensor.ndim - 4) + ("...",))
        )

    _init()

    def _visualize(*_):
        canvas.figure.clear()

        tensor_name = tensor_name_combo_box.get()
        source_subscript = ast.literal_eval(source_subscript_entry.get())
        program_id = ast.literal_eval(program_id_entry.get())
        target_subscript = ast.literal_eval(target_subscript_entry.get())

        if not isinstance(source_subscript, tuple):
            source_subscript = (source_subscript,)

        if not isinstance(target_subscript, tuple):
            target_subscript = (target_subscript,)

        tensor_id = tensor_names.index(tensor_name)

        source_tensor = source_tensors[tensor_id]
        target_tensor = target_tensors[tensor_id]

        _update_read_only_entry(source_shape_entry, str(tuple(source_tensor.shape)))
        _update_read_only_entry(target_shape_entry, str(tuple(target_tensor.shape[1:])))

        source_ax, target_ax = canvas.figure.subplots(
            1, 2, subplot_kw=dict(projection="3d", aspect="equal")
        )

        _visualize_mapping(
            source_ax,
            target_ax,
            source_tensor[source_subscript],
            target_tensor[(program_id, *target_subscript)],
        )

        for ax in (source_ax, target_ax):
            ax.axis(False)

        canvas.draw()

    root.bind("<Return>", _visualize)
    tensor_name_combo_box.bind("<<ComboboxSelected>>", _visualize)

    _visualize()

    root.mainloop()


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


def _visualize_mapping(source_ax, target_ax, source, target):
    def _to_3d_array(tensor):
        ndim = tensor.ndim

        if ndim > 3:
            raise ValueError("Tensor must have 3 or fewer dimensions.")

        shape = tuple(1 for _ in range(3 - ndim)) + tensor.shape

        return tensor.reshape(shape).permute(2, 1, 0).numpy(force=True)

    def _visualize_voxels(ax, filled, colors):
        def _explode(data):
            shape = np.array(data.shape) * 2 - 1

            if data.ndim == 4:
                shape[-1] = data.shape[-1]

            exploded = np.zeros(shape, dtype=data.dtype)
            exploded[::2, ::2, ::2, ...] = data

            return exploded

        filled = _explode(filled)
        colors = _explode(colors)

        ax.set_proj_type("ortho")
        ax.view_init(elev=-75, azim=-45, roll=-45)

        x, y, z = np.indices(np.array(filled.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.001
        y[:, 0::2, :] += 0.001
        z[:, :, 0::2] += 0.001
        x[1::2, :, :] += 0.999
        y[:, 1::2, :] += 0.999
        z[:, :, 1::2] += 0.999

        ax.voxels(
            x, y, z, filled, facecolors=colors, edgecolor=(0, 0, 0, alpha), shade=False
        )

        ax.axis("equal")

    source = _to_3d_array(source)
    target = _to_3d_array(target)

    def _normalize(tensor, epsilon=1e-6):
        target_min = target[target >= 0].min()
        target_max = target[target >= 0].max()

        return (tensor - target_min) / (target_max - target_min + epsilon)

    color_map = mpl.colormaps["viridis"]

    alpha = 0.5

    other_color = (1, 1, 1, alpha)

    source_colors = np.where(
        np.isin(source, target)[..., None],
        color_map(_normalize(source), alpha=alpha),
        other_color,
    )
    target_colors = np.where(
        target[..., None] >= 0, color_map(_normalize(target), alpha=alpha), other_color
    )

    _visualize_voxels(source_ax, np.full_like(source, True), source_colors)
    _visualize_voxels(target_ax, np.full_like(target, True), target_colors)
