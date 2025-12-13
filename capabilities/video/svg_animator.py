#!/usr/bin/env python3
"""
THAU SVG Animator
Generador de animaciones SVG para THAU

Crea animaciones vectoriales complejas:
- Logos animados
- Spinners y loaders
- Transiciones de texto
- Iconos interactivos
- Gráficos animados
- Efectos visuales
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import math


@dataclass
class AnimationConfig:
    """Configuración de animación"""
    duration: float = 2.0  # segundos
    repeat: str = "indefinite"  # indefinite, 1, 2, etc.
    easing: str = "ease-in-out"  # linear, ease, ease-in, ease-out, ease-in-out
    delay: float = 0.0
    fill: str = "freeze"  # freeze, remove


class SVGElement:
    """Elemento SVG base con soporte para animaciones"""

    def __init__(self, tag: str, **attrs):
        self.tag = tag
        self.attrs = attrs
        self.children: List['SVGElement'] = []
        self.animations: List[Dict] = []
        self.content: str = ""

    def add_child(self, child: 'SVGElement') -> 'SVGElement':
        self.children.append(child)
        return child

    def animate(self, attribute: str, values: List[Any], config: AnimationConfig = None) -> 'SVGElement':
        """Agrega una animación al elemento"""
        config = config or AnimationConfig()
        self.animations.append({
            "type": "animate",
            "attribute": attribute,
            "values": values,
            "config": config
        })
        return self

    def animate_transform(self, transform_type: str, values: List[Any], config: AnimationConfig = None) -> 'SVGElement':
        """Agrega una transformación animada"""
        config = config or AnimationConfig()
        self.animations.append({
            "type": "animateTransform",
            "transform_type": transform_type,
            "values": values,
            "config": config
        })
        return self

    def animate_motion(self, path: str, config: AnimationConfig = None) -> 'SVGElement':
        """Anima el elemento a lo largo de un path"""
        config = config or AnimationConfig()
        self.animations.append({
            "type": "animateMotion",
            "path": path,
            "config": config
        })
        return self

    def to_svg(self, indent: int = 0) -> str:
        """Convierte el elemento a string SVG"""
        spaces = "  " * indent
        attrs_str = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in self.attrs.items() if v is not None)

        # Generar animaciones
        animation_elements = []
        for anim in self.animations:
            if anim["type"] == "animate":
                cfg = anim["config"]
                values_str = ";".join(str(v) for v in anim["values"])
                animation_elements.append(
                    f'{spaces}  <animate attributeName="{anim["attribute"]}" '
                    f'values="{values_str}" dur="{cfg.duration}s" '
                    f'repeatCount="{cfg.repeat}" fill="{cfg.fill}" '
                    f'calcMode="{cfg.easing}" begin="{cfg.delay}s"/>'
                )
            elif anim["type"] == "animateTransform":
                cfg = anim["config"]
                values_str = ";".join(str(v) for v in anim["values"])
                animation_elements.append(
                    f'{spaces}  <animateTransform attributeName="transform" '
                    f'type="{anim["transform_type"]}" values="{values_str}" '
                    f'dur="{cfg.duration}s" repeatCount="{cfg.repeat}" '
                    f'fill="{cfg.fill}" begin="{cfg.delay}s"/>'
                )
            elif anim["type"] == "animateMotion":
                cfg = anim["config"]
                animation_elements.append(
                    f'{spaces}  <animateMotion path="{anim["path"]}" '
                    f'dur="{cfg.duration}s" repeatCount="{cfg.repeat}" '
                    f'fill="{cfg.fill}" begin="{cfg.delay}s"/>'
                )

        # Construir elemento
        if self.children or self.content or animation_elements:
            result = f'{spaces}<{self.tag} {attrs_str}>\n' if attrs_str else f'{spaces}<{self.tag}>\n'
            for anim in animation_elements:
                result += anim + "\n"
            if self.content:
                result += f'{spaces}  {self.content}\n'
            for child in self.children:
                result += child.to_svg(indent + 1) + "\n"
            result += f'{spaces}</{self.tag}>'
        else:
            result = f'{spaces}<{self.tag} {attrs_str}/>' if attrs_str else f'{spaces}<{self.tag}/>'

        return result


class SVGAnimator:
    """
    Generador principal de animaciones SVG

    Uso:
        animator = SVGAnimator(width=200, height=200)
        animator.create_spinner()
        svg_code = animator.render()
    """

    def __init__(self, width: int = 100, height: int = 100, viewbox: str = None):
        self.width = width
        self.height = height
        self.viewbox = viewbox or f"0 0 {width} {height}"
        self.root = SVGElement("svg",
            xmlns="http://www.w3.org/2000/svg",
            width=str(width),
            height=str(height),
            viewBox=self.viewbox
        )
        self.defs = SVGElement("defs")
        self.root.add_child(self.defs)

    def add_gradient(self, gradient_id: str, colors: List[Tuple[str, str]], gradient_type: str = "linear",
                     x1: str = "0%", y1: str = "0%", x2: str = "100%", y2: str = "0%") -> str:
        """Agrega un gradiente y retorna su ID para uso"""
        if gradient_type == "linear":
            grad = SVGElement("linearGradient", id=gradient_id, x1=x1, y1=y1, x2=x2, y2=y2)
        else:
            grad = SVGElement("radialGradient", id=gradient_id)

        for offset, color in colors:
            stop = SVGElement("stop", offset=offset, stop_color=color)
            grad.add_child(stop)

        self.defs.add_child(grad)
        return f"url(#{gradient_id})"

    def add_filter(self, filter_id: str, filter_type: str = "blur", **params) -> str:
        """Agrega un filtro SVG"""
        filt = SVGElement("filter", id=filter_id)

        if filter_type == "blur":
            blur = SVGElement("feGaussianBlur", stdDeviation=params.get("amount", "3"))
            filt.add_child(blur)
        elif filter_type == "glow":
            blur = SVGElement("feGaussianBlur", stdDeviation=params.get("amount", "3"), result="blur")
            filt.add_child(blur)
            merge = SVGElement("feMerge")
            merge.add_child(SVGElement("feMergeNode", **{"in": "blur"}))
            merge.add_child(SVGElement("feMergeNode", **{"in": "SourceGraphic"}))
            filt.add_child(merge)
        elif filter_type == "shadow":
            offset = SVGElement("feOffset", dx=params.get("dx", "2"), dy=params.get("dy", "2"), result="offset")
            blur = SVGElement("feGaussianBlur", stdDeviation=params.get("blur", "2"), result="blur")
            merge = SVGElement("feMerge")
            merge.add_child(SVGElement("feMergeNode", **{"in": "blur"}))
            merge.add_child(SVGElement("feMergeNode", **{"in": "SourceGraphic"}))
            filt.add_child(offset)
            filt.add_child(blur)
            filt.add_child(merge)

        self.defs.add_child(filt)
        return f"url(#{filter_id})"

    def circle(self, cx: float, cy: float, r: float, **attrs) -> SVGElement:
        """Crea un círculo"""
        elem = SVGElement("circle", cx=str(cx), cy=str(cy), r=str(r), **attrs)
        self.root.add_child(elem)
        return elem

    def rect(self, x: float, y: float, width: float, height: float, rx: float = 0, **attrs) -> SVGElement:
        """Crea un rectángulo"""
        elem = SVGElement("rect", x=str(x), y=str(y), width=str(width), height=str(height), rx=str(rx), **attrs)
        self.root.add_child(elem)
        return elem

    def path(self, d: str, **attrs) -> SVGElement:
        """Crea un path"""
        elem = SVGElement("path", d=d, **attrs)
        self.root.add_child(elem)
        return elem

    def text(self, content: str, x: float, y: float, **attrs) -> SVGElement:
        """Crea texto"""
        elem = SVGElement("text", x=str(x), y=str(y), **attrs)
        elem.content = content
        self.root.add_child(elem)
        return elem

    def group(self, **attrs) -> SVGElement:
        """Crea un grupo"""
        elem = SVGElement("g", **attrs)
        self.root.add_child(elem)
        return elem

    def line(self, x1: float, y1: float, x2: float, y2: float, **attrs) -> SVGElement:
        """Crea una línea"""
        elem = SVGElement("line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), **attrs)
        self.root.add_child(elem)
        return elem

    def polygon(self, points: List[Tuple[float, float]], **attrs) -> SVGElement:
        """Crea un polígono"""
        points_str = " ".join(f"{x},{y}" for x, y in points)
        elem = SVGElement("polygon", points=points_str, **attrs)
        self.root.add_child(elem)
        return elem

    # ==================== PRESETS DE ANIMACIÓN ====================

    def create_spinner(self, style: str = "circle", color: str = "#3498db", size: float = None) -> 'SVGAnimator':
        """
        Crea un spinner animado

        Estilos: circle, dots, bars, pulse, orbit
        """
        size = size or min(self.width, self.height) * 0.8
        cx, cy = self.width / 2, self.height / 2
        r = size / 2

        if style == "circle":
            # Spinner circular clásico
            circle = self.circle(cx, cy, r * 0.8,
                fill="none",
                stroke=color,
                stroke_width=str(r * 0.15),
                stroke_dasharray=f"{r * 3.14} {r * 1.57}",
                stroke_linecap="round"
            )
            circle.animate_transform("rotate", ["0 50 50", "360 50 50"],
                AnimationConfig(duration=1.0, easing="linear"))

        elif style == "dots":
            # Spinner de puntos
            num_dots = 8
            for i in range(num_dots):
                angle = (i / num_dots) * 2 * math.pi - math.pi / 2
                dot_x = cx + r * 0.7 * math.cos(angle)
                dot_y = cy + r * 0.7 * math.sin(angle)
                dot_r = r * 0.12

                dot = self.circle(dot_x, dot_y, dot_r, fill=color)
                # Fade animation con delay
                config = AnimationConfig(duration=0.8, delay=i * 0.1)
                dot.animate("opacity", ["0.3", "1", "0.3"], config)

        elif style == "bars":
            # Spinner de barras
            num_bars = 12
            for i in range(num_bars):
                angle = (i / num_bars) * 360
                bar = self.rect(cx - r * 0.05, cy - r * 0.9, r * 0.1, r * 0.3, rx=r * 0.05,
                    fill=color,
                    transform=f"rotate({angle} {cx} {cy})"
                )
                config = AnimationConfig(duration=1.0, delay=i * (1.0 / num_bars))
                bar.animate("opacity", ["1", "0.2", "1"], config)

        elif style == "pulse":
            # Pulso expandiéndose
            for i in range(3):
                circle = self.circle(cx, cy, r * 0.2,
                    fill="none",
                    stroke=color,
                    stroke_width="2"
                )
                config = AnimationConfig(duration=1.5, delay=i * 0.5)
                circle.animate("r", [str(r * 0.2), str(r)], config)
                circle.animate("opacity", ["1", "0"], config)

        elif style == "orbit":
            # Punto orbitando
            # Círculo base
            self.circle(cx, cy, r * 0.7, fill="none", stroke="#e0e0e0", stroke_width="2")
            # Punto orbitante
            dot = self.circle(cx + r * 0.7, cy, r * 0.15, fill=color)
            dot.animate_transform("rotate", ["0 50 50", "360 50 50"],
                AnimationConfig(duration=1.5, easing="linear"))

        return self

    def create_loading_text(self, text: str = "Loading", color: str = "#333") -> 'SVGAnimator':
        """Crea texto con animación de loading (puntos)"""
        cx, cy = self.width / 2, self.height / 2

        # Texto principal
        self.text(text, cx - 30, cy,
            fill=color,
            font_family="Arial, sans-serif",
            font_size="16",
            dominant_baseline="middle"
        )

        # Puntos animados
        for i in range(3):
            dot = self.circle(cx + 25 + i * 8, cy, 2, fill=color)
            config = AnimationConfig(duration=1.0, delay=i * 0.2)
            dot.animate("opacity", ["0", "1", "0"], config)

        return self

    def create_progress_bar(self, progress: float = 0.7, color: str = "#3498db",
                           bg_color: str = "#e0e0e0", animated: bool = True) -> 'SVGAnimator':
        """Crea una barra de progreso"""
        margin = 10
        bar_height = 20
        bar_y = (self.height - bar_height) / 2

        # Fondo
        self.rect(margin, bar_y, self.width - 2 * margin, bar_height, rx=bar_height / 2,
            fill=bg_color)

        # Progreso
        progress_width = (self.width - 2 * margin) * progress
        progress_bar = self.rect(margin, bar_y, progress_width, bar_height, rx=bar_height / 2,
            fill=color)

        if animated:
            # Animación de brillo
            config = AnimationConfig(duration=1.5)
            progress_bar.animate("opacity", ["0.8", "1", "0.8"], config)

        return self

    def create_heartbeat(self, color: str = "#e74c3c") -> 'SVGAnimator':
        """Crea un corazón con animación de latido"""
        cx, cy = self.width / 2, self.height / 2
        scale = min(self.width, self.height) / 100

        # Path del corazón
        heart_path = f"M {cx} {cy + 10 * scale} " \
                    f"C {cx - 20 * scale} {cy - 10 * scale}, {cx - 40 * scale} {cy + 10 * scale}, {cx} {cy + 40 * scale} " \
                    f"C {cx + 40 * scale} {cy + 10 * scale}, {cx + 20 * scale} {cy - 10 * scale}, {cx} {cy + 10 * scale}"

        heart = self.path(heart_path, fill=color)
        config = AnimationConfig(duration=0.8)
        heart.animate_transform("scale", ["1 1", "1.1 1.1", "1 1"], config)

        return self

    def create_wave(self, color: str = "#3498db", waves: int = 3) -> 'SVGAnimator':
        """Crea ondas animadas"""
        for i in range(waves):
            y_offset = 10 + i * 15
            path_d = f"M 0 {self.height / 2 + y_offset} " \
                    f"Q {self.width / 4} {self.height / 2 - 10 + y_offset}, {self.width / 2} {self.height / 2 + y_offset} " \
                    f"T {self.width} {self.height / 2 + y_offset}"

            wave = self.path(path_d,
                fill="none",
                stroke=color,
                stroke_width="3",
                stroke_opacity=str(1 - i * 0.2)
            )
            config = AnimationConfig(duration=2.0 + i * 0.5, easing="linear")
            # Animación de desplazamiento horizontal
            wave.animate("transform", [
                f"translate(0, 0)",
                f"translate({-self.width}, 0)"
            ], config)

        return self

    def create_typing_cursor(self, color: str = "#333") -> 'SVGAnimator':
        """Crea un cursor de escritura parpadeante"""
        cursor = self.rect(self.width / 2, self.height / 2 - 10, 2, 20, fill=color)
        config = AnimationConfig(duration=1.0)
        cursor.animate("opacity", ["1", "0", "1"], config)
        return self

    def create_logo_reveal(self, text: str = "THAU", color: str = "#667eea") -> 'SVGAnimator':
        """Crea una animación de revelación de logo/texto"""
        cx, cy = self.width / 2, self.height / 2

        # Gradiente
        grad_url = self.add_gradient("logoGrad",
            [("0%", "#667eea"), ("100%", "#764ba2")],
            x1="0%", y1="0%", x2="100%", y2="100%"
        )

        # Texto con máscara animada
        text_elem = self.text(text, cx, cy,
            fill=grad_url,
            font_family="Arial Black, sans-serif",
            font_size="32",
            font_weight="bold",
            text_anchor="middle",
            dominant_baseline="middle"
        )

        # Animación de opacidad y escala
        config = AnimationConfig(duration=1.5, repeat="1", fill="freeze")
        text_elem.animate("opacity", ["0", "1"], config)
        text_elem.animate_transform("scale", ["0.5 0.5", "1 1"], config)

        return self

    def create_check_animation(self, color: str = "#2ecc71") -> 'SVGAnimator':
        """Crea una animación de checkmark (completado)"""
        cx, cy = self.width / 2, self.height / 2
        r = min(self.width, self.height) * 0.35

        # Círculo de fondo
        circle = self.circle(cx, cy, r, fill=color)
        config = AnimationConfig(duration=0.3, repeat="1", fill="freeze")
        circle.animate_transform("scale", ["0 0", "1 1"], config)

        # Checkmark
        check_path = f"M {cx - r * 0.4} {cy} L {cx - r * 0.1} {cy + r * 0.3} L {cx + r * 0.4} {cy - r * 0.3}"
        check = self.path(check_path,
            fill="none",
            stroke="white",
            stroke_width="4",
            stroke_linecap="round",
            stroke_linejoin="round",
            stroke_dasharray="100",
            stroke_dashoffset="100"
        )
        config2 = AnimationConfig(duration=0.5, delay=0.3, repeat="1", fill="freeze")
        check.animate("stroke-dashoffset", ["100", "0"], config2)

        return self

    def create_error_animation(self, color: str = "#e74c3c") -> 'SVGAnimator':
        """Crea una animación de error (X)"""
        cx, cy = self.width / 2, self.height / 2
        r = min(self.width, self.height) * 0.35

        # Círculo de fondo
        circle = self.circle(cx, cy, r, fill=color)
        config = AnimationConfig(duration=0.3, repeat="1", fill="freeze")
        circle.animate_transform("scale", ["0 0", "1 1"], config)

        # X
        line_len = r * 0.5
        for angle in [45, -45]:
            x1 = cx - line_len * math.cos(math.radians(angle))
            y1 = cy - line_len * math.sin(math.radians(angle))
            x2 = cx + line_len * math.cos(math.radians(angle))
            y2 = cy + line_len * math.sin(math.radians(angle))

            line = self.line(x1, y1, x2, y2,
                stroke="white",
                stroke_width="4",
                stroke_linecap="round"
            )
            config2 = AnimationConfig(duration=0.3, delay=0.3, repeat="1", fill="freeze")
            line.animate("opacity", ["0", "1"], config2)

        return self

    def render(self) -> str:
        """Renderiza el SVG completo"""
        return self.root.to_svg()

    def save(self, filepath: str) -> str:
        """Guarda el SVG a un archivo"""
        svg_content = self.render()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        return filepath


# Función de conveniencia para crear animaciones rápidas
def create_animation(animation_type: str, **kwargs) -> str:
    """
    Crea una animación SVG predefinida

    Args:
        animation_type: Tipo de animación (spinner, loading, progress, heartbeat, wave, check, error, logo)
        **kwargs: Parámetros adicionales (width, height, color, etc.)

    Returns:
        String con el código SVG
    """
    width = kwargs.get("width", 100)
    height = kwargs.get("height", 100)
    color = kwargs.get("color", "#3498db")

    animator = SVGAnimator(width, height)

    if animation_type == "spinner":
        style = kwargs.get("style", "circle")
        animator.create_spinner(style=style, color=color)
    elif animation_type == "loading":
        text = kwargs.get("text", "Loading")
        animator.create_loading_text(text=text, color=color)
    elif animation_type == "progress":
        progress = kwargs.get("progress", 0.7)
        animator.create_progress_bar(progress=progress, color=color)
    elif animation_type == "heartbeat":
        animator.create_heartbeat(color=kwargs.get("color", "#e74c3c"))
    elif animation_type == "wave":
        animator.create_wave(color=color)
    elif animation_type == "check":
        animator.create_check_animation(color=kwargs.get("color", "#2ecc71"))
    elif animation_type == "error":
        animator.create_error_animation(color=kwargs.get("color", "#e74c3c"))
    elif animation_type == "logo":
        text = kwargs.get("text", "THAU")
        animator.create_logo_reveal(text=text, color=color)
    elif animation_type == "cursor":
        animator.create_typing_cursor(color=color)

    return animator.render()


# Test
if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent.parent / "data" / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎬 THAU SVG Animator - Test")
    print("=" * 60)

    animations = [
        ("spinner_circle", {"animation_type": "spinner", "style": "circle"}),
        ("spinner_dots", {"animation_type": "spinner", "style": "dots"}),
        ("spinner_bars", {"animation_type": "spinner", "style": "bars"}),
        ("spinner_pulse", {"animation_type": "spinner", "style": "pulse"}),
        ("spinner_orbit", {"animation_type": "spinner", "style": "orbit"}),
        ("loading_text", {"animation_type": "loading", "width": 150}),
        ("progress_bar", {"animation_type": "progress", "width": 200, "height": 40}),
        ("heartbeat", {"animation_type": "heartbeat"}),
        ("wave", {"animation_type": "wave", "width": 200, "height": 100}),
        ("check", {"animation_type": "check"}),
        ("error", {"animation_type": "error"}),
        ("logo_thau", {"animation_type": "logo", "width": 200, "height": 100}),
        ("typing_cursor", {"animation_type": "cursor", "width": 50, "height": 50}),
    ]

    print(f"\n📁 Guardando animaciones en: {output_dir}\n")

    for name, params in animations:
        svg = create_animation(**params)
        filepath = output_dir / f"{name}.svg"
        with open(filepath, 'w') as f:
            f.write(svg)
        print(f"   ✅ {name}.svg")

    # Crear HTML de preview
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>THAU SVG Animations</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: white; padding: 20px; }
        h1 { text-align: center; color: #667eea; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; text-align: center; }
        .card h3 { margin: 10px 0 0; font-size: 14px; color: #a0a0a0; }
        .card svg { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>🎬 THAU SVG Animations</h1>
    <div class="grid">
"""

    for name, params in animations:
        svg = create_animation(**params)
        html_content += f"""
        <div class="card">
            {svg}
            <h3>{name}</h3>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    preview_path = output_dir / "preview.html"
    with open(preview_path, 'w') as f:
        f.write(html_content)

    print(f"\n🌐 Preview HTML: {preview_path}")
    print("\n✅ Test completado!")
