# Instalación de THAU

Guía completa para instalar y configurar THAU en tu sistema.

---

## Requisitos del Sistema

### Requisitos Mínimos

- **SO**: Linux, macOS, o Windows 10/11
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disco**: 10GB de espacio libre
- **Python**: 3.10 o superior
- **Node.js**: 18+ (para THAU Code Desktop)

### Requisitos Recomendados

- **GPU**: NVIDIA GPU con 8GB+ VRAM o Apple Silicon M1/M2/M3
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disco**: 50GB SSD
- **Python**: 3.11+
- **Node.js**: 20+

---

## Instalación Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/your-org/thau.git
cd thau
```

### 2. Crear Entorno Virtual

#### En Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### En Windows:

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias de Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Si tienes GPU NVIDIA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Si usas Apple Silicon:

```bash
pip install torch torchvision torchaudio
```

### 4. Configurar Variables de Entorno

```bash
cp .env.example .env
```

Edita `.env` con tus configuraciones:

```bash
# Modelo base
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Dispositivo (auto detecta automáticamente)
DEVICE=auto  # o cuda, mps, cpu

# Training
BATCH_SIZE=4
MAX_LENGTH=2048
LEARNING_RATE=5e-5

# Generación
TEMPERATURE=0.7
TOP_P=0.9
MAX_NEW_TOKENS=512

# Memoria
USE_QUANTIZATION=false  # true para 8-bit quantization
MEMORY_DB_PATH=data/memory/chroma_db

# API
API_PORT=8001
API_HOST=0.0.0.0
```

### 5. Crear Estructura de Datos

```bash
mkdir -p data/{checkpoints,memory,logs}
```

### 6. Instalar en Modo Desarrollo (Opcional)

```bash
pip install -e .
```

---

## Configuraciones Específicas por Plataforma

### GPU NVIDIA (CUDA)

1. Instalar CUDA Toolkit 11.8+:
```bash
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Verificar instalación
nvcc --version
nvidia-smi
```

2. Instalar PyTorch con CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Verificar detección:
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
```

### Apple Silicon (MPS)

PyTorch detecta MPS automáticamente en M1/M2/M3:

```python
import torch
print(torch.backends.mps.is_available())  # Should return True
```

No se requiere configuración adicional.

### CPU Only

Si no tienes GPU, THAU funcionará en CPU:

```bash
# .env
DEVICE=cpu
BATCH_SIZE=2  # Reducir batch size
```

---

## Instalación de THAU Code Desktop

### Requisitos

- Node.js 18+
- npm 9+

### Pasos

1. Navegar al directorio:
```bash
cd thau-code-desktop
```

2. Instalar dependencias:
```bash
npm install
```

3. Iniciar en modo desarrollo:
```bash
npm run dev
```

4. Build para producción:
```bash
npm run build
npm run build:electron
```

Los instaladores estarán en `dist-electron/`:
- **macOS**: `.dmg` y `.zip`
- **Windows**: `.exe`
- **Linux**: `.AppImage` y `.deb`

---

## Verificación de Instalación

### Test 1: Importar Módulos

```python
python -c "
from core.models.base_transformer import TinyLLM
from adapters.device_manager import DeviceManager
from memory.manager import MemoryManager
print('✅ All modules imported successfully')
"
```

### Test 2: Detectar Dispositivo

```python
python -c "
from adapters.device_manager import DeviceManager
dm = DeviceManager()
print(f'Device: {dm.get_device_info()}')
"
```

### Test 3: Cargar Modelo

```python
python -c "
from thau_trainer.own_model_manager import ThauOwnModelManager
manager = ThauOwnModelManager(age=0)
print('✅ Model loaded successfully')
print(f'Parameters: {manager.get_model_size():,}')
"
```

### Test 4: Generar Texto

```python
from thau_trainer.own_model_manager import ThauOwnModelManager

manager = ThauOwnModelManager(age=0)
response = manager.generate("Hola, ¿cómo estás?", max_new_tokens=50)
print(f"THAU: {response}")
```

---

## Instalación de Componentes Opcionales

### ChromaDB (Memoria a Largo Plazo)

Ya incluido en requirements.txt, pero si necesitas reinstalar:

```bash
pip install chromadb
```

### Quantization (8-bit/4-bit)

Para modelos más grandes con menos memoria:

```bash
pip install bitsandbytes
```

Configurar en `.env`:
```bash
USE_QUANTIZATION=true
QUANTIZATION_BITS=8  # o 4
```

### FastAPI con ASGI Server

Para producción:

```bash
pip install uvicorn[standard] gunicorn
```

---

## Troubleshooting

### Error: "ModuleNotFoundError"

**Solución:**
```bash
pip install -e .
```

### Error: "CUDA out of memory"

**Soluciones:**
1. Reducir batch size:
```bash
# .env
BATCH_SIZE=2
```

2. Usar quantization:
```bash
# .env
USE_QUANTIZATION=true
```

3. Reducir max_length:
```bash
# .env
MAX_LENGTH=1024
```

### Error: "No module named 'torch'"

**Solución:**
```bash
pip install torch torchvision torchaudio
```

### Error: ChromaDB no se conecta

**Solución:**
```bash
rm -rf data/memory/chroma_db
mkdir -p data/memory/chroma_db
```

### Error: Puerto 8001 en uso

**Solución:**
```bash
# .env
API_PORT=8002
```

O encontrar y matar el proceso:
```bash
lsof -ti:8001 | xargs kill -9
```

---

## Actualización

Para actualizar a la última versión:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## Desinstalación

```bash
# Desactivar entorno virtual
deactivate

# Eliminar todo
cd ..
rm -rf thau
```

---

## Próximos Pasos

Una vez instalado THAU:

1. Lee la [Guía de Inicio Rápido](quickstart.md)
2. Explora la [Arquitectura](../architecture/overview.md)
3. Entrena tu primer modelo con la [Guía de Entrenamiento](../guides/training-thau-2b.md)
4. Genera imágenes con [THAU Vision](../guides/image-generation.md)

---

## Soporte

Si encuentras problemas durante la instalación:

- **GitHub Issues**: [https://github.com/your-org/thau/issues](https://github.com)
- **Discussions**: [https://github.com/your-org/thau/discussions](https://github.com)
- **Email**: support@thau-ai.com

---

**¡Felicidades! THAU está instalado y listo para usar.**
