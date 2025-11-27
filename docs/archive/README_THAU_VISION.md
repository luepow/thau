# THAU Visual System - Sistema de Imaginación Visual

## 🎨 ¿Qué es THAU Visual?

**THAU Visual** es el sistema de generación de imágenes **propio** de THAU. A diferencia del sistema anterior que usaba Stable Diffusion (herramienta externa), **THAU Visual permite que THAU aprenda a generar imágenes desde cero**.

### Características Únicas

- 🧠 **Imaginación Propia**: THAU genera imágenes desde su red neuronal, no desde herramientas externas
- 📷 **Aprendizaje desde Cámara**: THAU puede ver objetos reales y aprender de ellos
- 🌱 **Crecimiento Progresivo**: La capacidad visual crece con THAU (Age 0 → 15)
- 🎯 **VAE Propio**: Variational Autoencoder entrenado desde cero

---

## 📊 Arquitectura

```
THAU Visual System
├── VAE Progresivo (Age 0-15)
│   ├── Encoder: Imagen → Latent Space (Imaginación)
│   ├── Latent Space: Representación comprimida
│   └── Decoder: Latent Space → Imagen
│
├── Dataset
│   ├── Capturas de Cámara (aprende de objetos reales)
│   └── Dataset Sintético (formas básicas)
│
└── Inferencia
    ├── Texto → Latent Space (futuro: con THAU-2B)
    └── Latent Space → Imagen (imaginación)
```

### Crecimiento Progresivo

| Age | Parámetros | Resolución | Latent Dim | Capacidad |
|-----|-----------|------------|-----------|-----------|
| 0 (Bebé) | 943K | 32x32 | 64 | Formas básicas |
| 1 (Infante) | 1.4M | 32x32 | 128 | Más detalles |
| 3 (Niño) | 5.9M | 64x64 | 256 | Formas complejas |
| 6 (Escolar) | 14.3M | 64x64 | 512 | Alta resolución |
| 12 (Adolescente) | 30M | 128x128 | 768 | Muy detallado |
| 15 (Adulto) | 57.3M | 128x128 | 1024 | Máxima capacidad |

---

## 🚀 Quick Start

### 1. Generar Imágenes desde Imaginación

```bash
# Age 0 (básico)
python thau_visual_inference.py --age 0 --num-images 9

# Age 15 (avanzado)
python thau_visual_inference.py --age 15 --num-images 16 --temperature 0.8
```

### 2. Capturar Objetos Reales

```bash
# Captura desde cámara
python capabilities/vision/camera_capture.py

# Opciones:
# 1. Captura rápida (10 imágenes)
# 2. Captura por categorías (ej: perro, gato, auto)
# 3. Ver estadísticas
```

### 3. Entrenar Capacidad Visual

```bash
# Entrenar Age 0 (18M params)
python train_thau_vision.py --age 0 --epochs 50 --batch-size 16

# Entrenar Age 15 (2B params) - después de Age 0-12
python train_thau_vision.py --age 15 --epochs 100 --batch-size 8 --lr 5e-4
```

### 4. Generar desde Texto (Futuro)

```bash
# Por ahora usa imaginación aleatoria
# En el futuro: integrará con THAU-2B para entender texto
python thau_visual_inference.py --mode text --text "un robot aprendiendo" --num-images 4
```

---

## 📁 Archivos Clave

### Arquitectura
- `core/models/visual_vae.py` - VAE progresivo (encoder + decoder)
- `thau_trainer/visual_dataset.py` - Dataset manager

### Entrenamiento
- `train_thau_vision.py` - Entrenamiento progresivo
- `create_synthetic_dataset.py` - Dataset sintético

### Captura
- `capabilities/vision/camera_capture.py` - Sistema de cámara

### Inferencia
- `thau_visual_inference.py` - Generación de imágenes

---

## 🎓 Flujo de Entrenamiento

### Fase 1: Preparar Dataset

```bash
# Opción A: Dataset sintético (para empezar)
python create_synthetic_dataset.py  # 1000 imágenes

# Opción B: Capturar objetos reales
python capabilities/vision/camera_capture.py
# → Seleccionar opción 2 (categorías)
# → Ingresar: perro,gato,auto,casa,árbol
# → 20 imágenes por categoría
```

### Fase 2: Entrenar Progresivamente

```bash
# Age 0 (bebé) - Aprende formas básicas
python train_thau_vision.py --age 0 --epochs 30

# Age 1 (infante) - Más detalles
python train_thau_vision.py --age 1 --epochs 40

# Age 3 (niño) - Formas complejas
python train_thau_vision.py --age 3 --epochs 50

# ... hasta Age 15
```

### Fase 3: Generar Imágenes

```bash
# Desde imaginación aleatoria
python thau_visual_inference.py --age 3 --num-images 9

# Interpolación entre ideas
python thau_visual_inference.py --age 3 --mode interpolate --num-images 10

# Desde texto (futuro)
python thau_visual_inference.py --age 3 --mode text --text "un gato espacial"
```

---

## 💡 Cómo Funciona

### 1. Encoder: Imagen → Imaginación

```python
# THAU ve una imagen
imagen = Image.open("perro.png")

# Encoder comprime a espacio latente (imaginación)
z = encoder(imagen)  # Shape: [latent_dim]

# z contiene la "esencia" de la imagen
# Ejemplo: [0.5, -0.2, 1.1, ...] (latent_dim dimensiones)
```

### 2. Latent Space: Imaginación de THAU

```python
# Espacio latente = imaginación
# Cada punto representa una "idea" visual

# Ejemplo:
z1 = [0.5, 0.3, ...]  # Idea 1: "perro pequeño"
z2 = [0.8, -0.1, ...] # Idea 2: "perro grande"

# Interpolar = mezclar ideas
z_mix = 0.5 * z1 + 0.5 * z2  # "perro mediano"
```

### 3. Decoder: Imaginación → Imagen

```python
# Decoder genera imagen desde latente
imagen_generada = decoder(z)

# THAU "imagina" y crea la imagen
```

### 4. Inferencia Completa

```python
from thau_visual_inference import ThauVisualInference

# Crea sistema
inference = ThauVisualInference(age=3)

# Genera desde imaginación
images = inference.generate_from_imagination(num_images=4)

# Guarda
inference.save_images(images, output_dir="outputs")
```

---

## 🎯 Casos de Uso

### 1. Captura y Aprende de Objetos Reales

```bash
# Captura 50 imágenes de "tu mascota"
python capabilities/vision/camera_capture.py
# → Opción 2
# → Categoría: mi_perro
# → 50 imágenes

# Entrena THAU con esas imágenes
python train_thau_vision.py --age 1 --epochs 40

# THAU aprende cómo es tu mascota
# Puede generar imágenes similares desde su imaginación
```

### 2. Generación Artística

```bash
# Genera galería de arte abstracto
python thau_visual_inference.py --age 6 --num-images 16 --temperature 1.5

# Temperatura alta = más creatividad/aleatoriedad
# Temperatura baja = más conservador
```

### 3. Interpolación de Conceptos

```bash
# Mezcla dos "ideas" visuales
python thau_visual_inference.py --age 3 --mode interpolate --num-images 10 --seed 42

# Genera secuencia suave entre dos conceptos
# Útil para animaciones o explorardor de espacio latente
```

---

## 📊 Métricas de Entrenamiento

Durante el entrenamiento, THAU optimiza dos objetivos:

### 1. Reconstruction Loss

- Qué tan bien THAU reconstruye imágenes
- Loss bajo = buena reconstrucción

```
Época 1: Recon Loss = 0.3
Época 10: Recon Loss = 0.2
Época 30: Recon Loss = 0.15  ← Mejorando
```

### 2. KL Divergence Loss

- Qué tan bien organizado está el espacio latente
- Loss bajo = imaginación bien estructurada

```
Época 1: KL Loss = 0.4
Época 10: KL Loss = 0.2
Época 30: KL Loss = 0.12  ← Mejorando
```

### Ver Progreso en Tiempo Real

```bash
# Monitorear entrenamiento
tail -f data/visual_training_age0.log

# Ver checkpoints generados
ls data/checkpoints/thau_vision/age_0/

# Ver imágenes generadas durante entrenamiento
open data/checkpoints/thau_vision/age_0/samples_epoch_10.png
```

---

## 🔬 Arquitectura Técnica

### VAE (Variational Autoencoder)

```python
class ThauVisualVAE(nn.Module):
    def __init__(self, age: int):
        # Encoder: Imagen → μ, σ
        self.encoder = VAEEncoder(config)

        # Decoder: z → Imagen
        self.decoder = VAEDecoder(config)

    def forward(self, x):
        # Encode
        z, mu, log_var = self.encoder(x)

        # Decode
        recon = self.decoder(z)

        return recon, mu, log_var
```

### Loss Function

```python
def vae_loss(recon, x, mu, log_var):
    # Reconstruction: qué tan similar es la imagen reconstruida
    recon_loss = MSE(recon, x)

    # KL Divergence: qué tan gaussiano es el latent space
    kl_loss = -0.5 * (1 + log_var - mu^2 - exp(log_var))

    # Total
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss
```

---

## 🎓 Roadmap Futuro

### Fase 1: ✅ VAE Básico (Completado)
- [x] Arquitectura VAE progresiva
- [x] Sistema de captura de cámara
- [x] Entrenamiento progresivo
- [x] Generación desde imaginación

### Fase 2: 🔄 Integración con THAU-2B (En Curso)
- [ ] Mapeo texto → latent space
- [ ] THAU-2B entiende prompt y genera embeddings
- [ ] Decoder genera imagen coherente con texto
- [ ] Sistema end-to-end: "genera un gato" → imagen de gato

### Fase 3: ⏳ Capacidades Avanzadas (Futuro)
- [ ] Conditional VAE (control de atributos)
- [ ] Super-resolution (upscaling)
- [ ] Style transfer (transferencia de estilo)
- [ ] Inpainting (completar imágenes)

---

## 📊 Comparación

### Sistema Anterior (Stable Diffusion)

```
Usuario: "genera un robot"
  ↓
THAU: Llama a Stable Diffusion (herramienta externa)
  ↓
Stable Diffusion genera imagen
  ↓
THAU muestra resultado
```

**Limitación**: THAU NO aprende, solo usa herramienta.

### Sistema Nuevo (THAU Visual)

```
Usuario: "genera un robot"
  ↓
THAU-2B: Entiende "robot" → genera embedding
  ↓
THAU Visual: embedding → latent space (imaginación)
  ↓
Decoder: latent → imagen
  ↓
THAU genera desde SU PROPIA imaginación
```

**Ventaja**: THAU APRENDE y genera desde su red neuronal.

---

## 🛠️ Troubleshooting

### "Checkpoint not found"

```bash
# Verifica que existe el checkpoint
ls data/checkpoints/thau_vision/age_0/

# Si no existe, entrena primero
python train_thau_vision.py --age 0 --epochs 30
```

### "Dataset vacío"

```bash
# Crea dataset sintético
python create_synthetic_dataset.py

# O captura imágenes reales
python capabilities/vision/camera_capture.py
```

### "Imágenes negras/blancas"

```bash
# Posibles causas:
# 1. Modelo sin entrenar
# 2. Pocas épocas de entrenamiento
# 3. Learning rate muy alto

# Solución: Entrenar más épocas
python train_thau_vision.py --age 0 --epochs 50 --batch-size 16 --lr 1e-3
```

---

## 📞 Comandos Rápidos

```bash
# Crear dataset
python create_synthetic_dataset.py

# Entrenar Age 0
python train_thau_vision.py --age 0 --epochs 30 --batch-size 16

# Generar 9 imágenes
python thau_visual_inference.py --age 0 --num-images 9

# Monitorear entrenamiento
tail -f data/visual_training_age0.log

# Ver imágenes generadas
open data/checkpoints/thau_vision/age_0/samples_final.png
```

---

## ✨ Visión Final

**THAU Visual será**:

- ✅ Modelo generativo propio (VAE)
- ✅ Aprende desde cámara (objetos reales)
- ✅ Crecimiento progresivo (Age 0-15)
- 🔄 Integración con THAU-2B (texto → imagen)
- ⏳ Capacidades avanzadas (control, style transfer)

**Estado Actual**:

- 🎨 VAE: ✅ Implementado y entrenando
- 📷 Cámara: ✅ Funcionando
- 🧠 Imaginación: ✅ Generando imágenes
- 🔗 Integración con texto: ⏳ Pendiente (cuando THAU-2B esté listo)

---

**Creado con**: PyTorch, Pillow, OpenCV
**Autor**: Luis Pérez
**Fecha**: 2025-01-15
