import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
 
import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
 
matplotlib.use("TkAgg")
 
 
ELECTRON_MASS = 9.109_383_713_9e-31  # kg,
HBAR = 1.054_571_817e-34  # J s
ELEMENTARY_CHARGE = 1.602_176_634e-19  # C
GROUP_VELOCITY = 5.0e4  # 5 x 10^6 cm/s, expressed in m/s
ENERGY = 0.5 * ELECTRON_MASS * GROUP_VELOCITY**2
 
A_MIN_NM = 0.02
RATIO_MIN = 0.02
GRID_SIZE = 350
TRANSMISSION_FLOOR = 1e-14
 
 
def transmission_coefficient(a_nm: np.ndarray, energy_ratio: np.ndarray) -> np.ndarray:
    """Exact transmission through a rectangular barrier for 0 < E/V0 < 1."""
    barrier_energy = ENERGY / energy_ratio
    kappa = np.sqrt(2.0 * ELECTRON_MASS * (barrier_energy - ENERGY)) / HBAR
    argument = kappa * a_nm * 1e-9
    correction = np.sinh(argument) ** 2 / (4.0 * energy_ratio * (1.0 - energy_ratio))
    return 1.0 / (1.0 + correction)
 
 
class TransmissionMapApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mapa de transmision - Barrera rectangular")
        self.root.geometry("900x680")
        self.root.minsize(700, 520)
        self.pending_update: str | None = None
 
        self._configure_style()
        self._build_layout()
        self.update_map()
 
    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Info.TLabel", font=("Segoe UI", 10), foreground="#374151")
        style.configure("Value.TLabel", font=("Consolas", 10, "bold"), foreground="#0f766e")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
 
    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
 
        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="Coeficiente de transmision", style="Title.TLabel").pack(side=tk.LEFT)
        energy_ev = ENERGY / ELEMENTARY_CHARGE
        ttk.Label(
            header,
            text=f"Grupo 1  |  vs = 5 x 10^6 cm/s  |  E = {energy_ev * 1e3:.3f} meV",
            style="Info.TLabel",
        ).pack(side=tk.RIGHT, pady=(7, 0))
 
        controls = ttk.Frame(outer, padding=(4, 4, 4, 12))
        controls.pack(fill=tk.X)
        controls.columnconfigure(1, weight=1)
 
        self.a_max = tk.DoubleVar(value=15.0)
        self.ratio_max = tk.DoubleVar(value=0.99)
        self.log_scale = tk.BooleanVar(value=True)
        self.a_value = tk.StringVar()
        self.ratio_value = tk.StringVar()
 
        ttk.Label(controls, text="Ancho maximo de a").grid(row=0, column=0, sticky=tk.W, padx=(0, 14))
        ttk.Scale(
            controls,
            from_=0.5,
            to=40.0,
            variable=self.a_max,
            command=self.schedule_update,
        ).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(controls, textvariable=self.a_value, style="Value.TLabel", width=10).grid(
            row=0, column=2, sticky=tk.E, padx=(14, 0)
        )
 
        ttk.Label(controls, text="Altura maxima E/V0").grid(row=1, column=0, sticky=tk.W, padx=(0, 14), pady=(10, 0))
        ttk.Scale(
            controls,
            from_=0.10,
            to=0.999,
            variable=self.ratio_max,
            command=self.schedule_update,
        ).grid(row=1, column=1, sticky=tk.EW, pady=(10, 0))
        ttk.Label(controls, textvariable=self.ratio_value, style="Value.TLabel", width=10).grid(
            row=1, column=2, sticky=tk.E, padx=(14, 0), pady=(10, 0)
        )
        ttk.Checkbutton(
            controls,
            text="Escala logaritmica para T",
            variable=self.log_scale,
            command=self.schedule_update,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
 
        plot_frame = ttk.Frame(outer, relief=tk.SOLID, borderwidth=1)
        plot_frame.pack(fill=tk.BOTH, expand=True)
 
        self.figure = Figure(figsize=(7.2, 4.4), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
 
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False).pack(side=tk.LEFT)
        ttk.Button(
            toolbar_frame,
            text="Guardar PNG",
            command=self.save_png,
            style="Accent.TButton",
        ).pack(side=tk.RIGHT, padx=8, pady=5)
 
    def schedule_update(self, _value: str = "") -> None:
        if self.pending_update is not None:
            self.root.after_cancel(self.pending_update)
        self.pending_update = self.root.after(80, self.update_map)
 
    def update_map(self) -> None:
        self.pending_update = None
        a_max = self.a_max.get()
        ratio_max = self.ratio_max.get()
        self.a_value.set(f"{a_max:.2f} nm")
        self.ratio_value.set(f"{ratio_max:.3f}")
 
        a_values = np.linspace(A_MIN_NM, a_max, GRID_SIZE)
        ratios = np.linspace(RATIO_MIN, ratio_max, GRID_SIZE)
        a_grid, ratio_grid = np.meshgrid(a_values, ratios)
        transmission = transmission_coefficient(a_grid, ratio_grid)
        if self.log_scale.get():
            plot_values = np.clip(transmission, TRANSMISSION_FLOOR, 1.0)
            normalization = LogNorm(vmin=TRANSMISSION_FLOOR, vmax=1.0)
            scale_label = "logaritmica"
        else:
            plot_values = transmission
            normalization = Normalize(vmin=0.0, vmax=1.0)
            scale_label = "lineal, 0 a 1"
 
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        heatmap = axes.pcolormesh(
            a_values,
            ratios,
            plot_values,
            shading="auto",
            cmap="inferno",
            norm=normalization,
        )
        colorbar = self.figure.colorbar(heatmap, ax=axes, pad=0.025)
        colorbar.set_label(f"Coeficiente de transmision T (escala {scale_label})")
        axes.set(
            title="Transmision exacta a traves de una barrera rectangular",
            xlabel="Ancho de barrera a [nm]",
            ylabel="Razon de energias E/V0",
            xlim=(A_MIN_NM, a_max),
            ylim=(RATIO_MIN, ratio_max),
        )
        axes.grid(False)
        self.canvas.draw_idle()
 
    def save_png(self) -> None:
        if self.pending_update is not None:
            self.root.after_cancel(self.pending_update)
            self.update_map()
 
        default_name = f"mapa_transmision_a-{self.a_max.get():.2f}nm_ratio-{self.ratio_max.get():.3f}.png"
        destination = filedialog.asksaveasfilename(
            title="Guardar mapa de calor",
            initialdir=Path.cwd(),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=(("Imagen PNG", "*.png"),),
        )
        if not destination:
            return
 
        try:
            self.figure.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        except OSError as error:
            messagebox.showerror("No se pudo guardar", str(error))
            return
        messagebox.showinfo("Imagen guardada", f"PNG guardado en:\n{destination}")
 
 
def main() -> None:
    root = tk.Tk()
    TransmissionMapApp(root)
    root.mainloop()
 
 
if __name__ == "__main__":
    main()