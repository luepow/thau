"""
Entrenamiento Especializado: THAU Software Engineering Expert

Entrena a THAU para ser un experto en:
- Desarrollo Backend (APIs, bases de datos, arquitectura)
- Desarrollo Frontend (React, Vue, UI/UX)
- Mejores Prácticas (SOLID, Clean Code, Design Patterns)
- Algoritmos y Estructuras de Datos
- Código simple y mantenible
- Sentido común en decisiones técnicas
"""

import torch
from pathlib import Path
from typing import List, Dict
from thau_trainer.own_model_manager import ThauOwnModelManager
from config.base_config import get_config

config = get_config()


class SoftwareEngineeringExpertTrainer:
    """
    Entrena a THAU en conocimiento de ingeniería de software de clase mundial
    """

    def __init__(self, target_age: int = 15):
        self.manager = ThauOwnModelManager()
        self.manager.initialize_model(cognitive_age=target_age)
        self.training_examples = []
        self.target_age = target_age

    def load_best_practices_data(self):
        """
        Carga ejemplos de mejores prácticas en desarrollo de software
        """

        # 1. SOLID Principles
        solid_examples = [
            {
                "question": "¿Cómo aplicarías el principio de Responsabilidad Única?",
                "answer": """El principio de Responsabilidad Única (SRP) dice que una clase debe tener una sola razón para cambiar.

**Mal ejemplo:**
```python
class UserManager:
    def create_user(self, data): pass
    def send_email(self, user): pass  # ❌ Responsabilidad adicional
    def generate_report(self, user): pass  # ❌ Responsabilidad adicional
```

**Buen ejemplo:**
```python
class UserRepository:
    def create_user(self, data): pass
    def find_user(self, id): pass

class EmailService:
    def send_email(self, user): pass

class ReportGenerator:
    def generate_report(self, user): pass
```

Cada clase tiene **una sola responsabilidad**, haciendo el código más fácil de mantener y testear."""
            },
            {
                "question": "¿Cuál es la forma más simple de implementar paginación en una API REST?",
                "answer": """La forma más simple y estándar:

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/users")
async def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    offset = (page - 1) * page_size
    users = db.query(User).offset(offset).limit(page_size).all()
    total = db.query(User).count()

    return {
        "data": users,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    }
```

**Principios aplicados:**
1. **Valores por defecto sensatos** (page=1, page_size=20)
2. **Validación automática** (ge=1, le=100)
3. **Respuesta consistente** con data y metadata
4. **Sentido común**: Límite máximo para evitar queries pesadas"""
            }
        ]

        # 2. Backend Best Practices
        backend_examples = [
            {
                "question": "¿Cómo diseñarías una API REST para un sistema de e-commerce?",
                "answer": """Diseño REST siguiendo mejores prácticas:

```python
# 1. Estructura de recursos jerárquica
GET    /api/products              # Listar productos
GET    /api/products/{id}          # Detalles de producto
POST   /api/products              # Crear producto (admin)
PUT    /api/products/{id}          # Actualizar producto
DELETE /api/products/{id}          # Eliminar producto

GET    /api/products/{id}/reviews # Reviews de un producto
POST   /api/products/{id}/reviews # Crear review

# 2. Carrito de compras (recurso de usuario)
GET    /api/cart                  # Ver carrito actual
POST   /api/cart/items            # Añadir item
PUT    /api/cart/items/{id}       # Actualizar cantidad
DELETE /api/cart/items/{id}       # Eliminar item

# 3. Orders
POST   /api/orders                # Crear orden desde carrito
GET    /api/orders                # Mis órdenes
GET    /api/orders/{id}           # Detalle de orden

# 4. Paginación, filtros, búsqueda
GET /api/products?page=1&page_size=20&category=electronics&min_price=100
```

**Principios de sentido común:**
1. URLs descriptivas y consistentes
2. Verbos HTTP semánticos (GET/POST/PUT/DELETE)
3. Versionado (/api/v1/)
4. Paginación por defecto
5. Autenticación JWT en header
6. Rate limiting para protección
7. Respuestas consistentes con status codes apropiados"""
            },
            {
                "question": "¿Cómo manejarías transacciones en una base de datos para un sistema de pagos?",
                "answer": """Uso de transacciones con rollback automático:

```python
from sqlalchemy.orm import Session
from fastapi import HTTPException

async def process_payment(
    db: Session,
    user_id: int,
    amount: float,
    order_id: int
):
    try:
        # Iniciar transacción
        with db.begin():
            # 1. Verificar saldo
            user = db.query(User).filter(User.id == user_id).with_for_update().first()
            if not user or user.balance < amount:
                raise ValueError("Saldo insuficiente")

            # 2. Decrementar saldo del usuario
            user.balance -= amount

            # 3. Crear registro de pago
            payment = Payment(
                user_id=user_id,
                order_id=order_id,
                amount=amount,
                status="completed"
            )
            db.add(payment)

            # 4. Actualizar estado de orden
            order = db.query(Order).filter(Order.id == order_id).first()
            order.status = "paid"

            # Commit automático si todo OK

    except Exception as e:
        # Rollback automático en caso de error
        raise HTTPException(status_code=400, detail=str(e))

    return payment

```

**Sentido común aplicado:**
1. `with_for_update()` → Lock pesimista previene race conditions
2. Transacción atómica → Todo o nada
3. Verificaciones antes de modificar
4. Rollback automático en excepciones
5. Logging de operaciones críticas (agregarlo)"""
            }
        ]

        # 3. Frontend Best Practices
        frontend_examples = [
            {
                "question": "¿Cuál es la forma más simple y eficiente de manejar estado en React?",
                "answer": """Depende del caso, usa la solución más simple:

**Caso 1: Estado local simple → useState**
```tsx
function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}
```

**Caso 2: Estado compartido entre componentes cercanos → Context**
```tsx
const ThemeContext = createContext();

function App() {
  const [theme, setTheme] = useState('light');
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      <Header />
      <Content />
    </ThemeContext.Provider>
  );
}

function Header() {
  const { theme, setTheme } = useContext(ThemeContext);
  return <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
    Toggle
  </button>;
}
```

**Caso 3: Estado complejo global → Zustand (más simple que Redux)**
```tsx
import create from 'zustand';

const useStore = create((set) => ({
  user: null,
  cart: [],
  addToCart: (item) => set((state) => ({
    cart: [...state.cart, item]
  })),
  clearCart: () => set({ cart: [] })
}));

function CartButton() {
  const { cart, addToCart } = useStore();
  return <button>Cart ({cart.length})</button>;
}
```

**Sentido común:**
- Empieza simple (useState)
- Solo añade complejidad cuando sea necesario
- Context para temas/autenticación
- Zustand/Redux para estado global complejo
- Evita prop drilling con Context"""
            }
        ]

        # 4. Database Design
        database_examples = [
            {
                "question": "¿Cómo diseñarías el esquema de base de datos para un sistema de blog con autores, posts, comentarios y tags?",
                "answer": """Diseño normalizado siguiendo mejores prácticas:

```sql
-- Tabla de usuarios/autores
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para búsquedas frecuentes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- Tabla de posts
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    author_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    published BOOLEAN DEFAULT FALSE,
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_posts_author ON posts(author_id);
CREATE INDEX idx_posts_slug ON posts(slug);
CREATE INDEX idx_posts_published ON posts(published, published_at);

-- Tabla de comentarios
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_comments_post ON comments(post_id, created_at DESC);

-- Tabla de tags
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    slug VARCHAR(50) UNIQUE NOT NULL
);

-- Tabla de relación many-to-many posts-tags
CREATE TABLE post_tags (
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, tag_id)
);

CREATE INDEX idx_post_tags_tag ON post_tags(tag_id);
```

**Principios aplicados:**
1. **Normalización**: Sin datos duplicados
2. **Relaciones claras**: FK constraints
3. **Cascade delete**: Limpieza automática
4. **Índices estratégicos**: En campos de búsqueda frecuente
5. **Timestamps**: Auditoría
6. **Slugs**: URLs amigables
7. **Sentido común**: user_id nullable en comments (usuario borrado pero comentario queda)"""
            }
        ]

        # 5. Algorithms & Data Structures
        algorithms_examples = [
            {
                "question": "¿Cuál es la forma más eficiente de encontrar elementos duplicados en un array?",
                "answer": """La forma más simple y eficiente es usar un Set:

```python
def find_duplicates(arr):
    \"\"\"
    Encuentra duplicados en O(n) tiempo, O(n) espacio
    \"\"\"
    seen = set()
    duplicates = set()

    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

# Ejemplo
nums = [1, 2, 3, 2, 4, 5, 3]
print(find_duplicates(nums))  # [2, 3]
```

**Análisis de complejidad:**
- Tiempo: O(n) - Una sola pasada
- Espacio: O(n) - En el peor caso, todos únicos

**Alternativa más Pythonic:**
```python
from collections import Counter

def find_duplicates(arr):
    counts = Counter(arr)
    return [item for item, count in counts.items() if count > 1]
```

**Sentido común:**
- No reinventes la rueda: usa Counter de collections
- O(n) es óptimo para este problema
- Código legible > Código "clever"
- Documentar complejidad temporal/espacial"""
            }
        ]

        # 6. Code Simplicity & Common Sense
        simplicity_examples = [
            {
                "question": "Tengo un endpoint que tarda 5 segundos. ¿Cómo lo optimizarías?",
                "answer": """Proceso sistemático de optimización con sentido común:

**Paso 1: MEDIR primero (no asumas)**
```python
import time
import logging

@app.get("/slow-endpoint")
async def slow_endpoint():
    start = time.time()

    # Medir cada operación
    t1 = time.time()
    data1 = await fetch_from_db()
    logging.info(f"DB query: {time.time() - t1:.2f}s")

    t2 = time.time()
    data2 = await external_api_call()
    logging.info(f"API call: {time.time() - t2:.2f}s")

    t3 = time.time()
    result = process_data(data1, data2)
    logging.info(f"Processing: {time.time() - t3:.2f}s")

    logging.info(f"Total: {time.time() - start:.2f}s")
    return result
```

**Paso 2: Optimizar lo que realmente es lento**

Si DB query es lento (3s):
```python
# Añadir índices
CREATE INDEX idx_users_email ON users(email);

# O usar caché
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user(email):
    return db.query(User).filter(User.email == email).first()
```

Si API externa es lenta (4s):
```python
# Paralelizar llamadas independientes
import asyncio

results = await asyncio.gather(
    fetch_from_db(),
    external_api_call()
)
```

Si procesamiento es lento:
```python
# Usar Pandas para operaciones batch
import pandas as pd
df = pd.DataFrame(data)
result = df.groupby('category').agg({'amount': 'sum'})
```

**Sentido común:**
1. **Medir antes de optimizar** - No asumas
2. **Optimiza el cuello de botella** real
3. **Low-hanging fruit primero**: Índices, caché
4. **Paraleliza lo independiente**
5. **Considera complejidad** de mantenimiento vs ganancia"""
            }
        ]

        # Combinar todos los ejemplos
        self.training_examples = (
            solid_examples +
            backend_examples +
            frontend_examples +
            database_examples +
            algorithms_examples +
            simplicity_examples
        )

        print(f"✅ Cargados {len(self.training_examples)} ejemplos de mejores prácticas")

    def add_custom_training_data(self, examples: List[Dict]):
        """
        Añade ejemplos personalizados de entrenamiento
        """
        self.training_examples.extend(examples)
        print(f"✅ Añadidos {len(examples)} ejemplos personalizados")

    def train(self, epochs: int = 3, learning_rate: float = 5e-5):
        """
        Entrena el modelo con los ejemplos de mejores prácticas
        """
        print(f"\n🚀 Iniciando entrenamiento Software Engineering Expert")
        print(f"   Ejemplos: {len(self.training_examples)}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning Rate: {learning_rate}\n")

        # Configurar LoRA para fine-tuning eficiente
        self.manager.setup_lora(
            r=32,  # Rank más alto para capturar patrones complejos
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        total_steps = len(self.training_examples) * epochs
        step = 0

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{epochs}")
            print(f"{'='*60}\n")

            epoch_loss = 0

            for i, example in enumerate(self.training_examples):
                # Formatear como conversación
                training_text = f"""<|user|>
{example['question']}

<|assistant|>
{example['answer']}
<|endoftext|>"""

                # Entrenar con este ejemplo
                loss = self.manager.train_step(
                    training_text,
                    learning_rate=learning_rate,
                    num_steps=5  # 5 steps por ejemplo
                )

                epoch_loss += loss
                step += 1

                # Mostrar progreso
                if (i + 1) % 10 == 0:
                    avg_loss = epoch_loss / (i + 1)
                    progress = (step / total_steps) * 100
                    print(f"   Ejemplo {i+1}/{len(self.training_examples)} | "
                          f"Loss: {loss:.4f} | "
                          f"Avg: {avg_loss:.4f} | "
                          f"Progreso: {progress:.1f}%")

            avg_epoch_loss = epoch_loss / len(self.training_examples)
            print(f"\n   📊 Epoch {epoch + 1} Loss Promedio: {avg_epoch_loss:.4f}")

            # Guardar checkpoint cada epoch
            checkpoint_name = f"software_expert_epoch_{epoch+1}.pt"
            self.manager.save_checkpoint(checkpoint_name)
            print(f"   💾 Checkpoint guardado: {checkpoint_name}")

        print(f"\n✅ Entrenamiento completado!")
        print(f"   Modelo especializado en Software Engineering Best Practices")

    def test_knowledge(self):
        """
        Prueba el conocimiento adquirido
        """
        print(f"\n🧪 Probando conocimiento adquirido...\n")

        test_questions = [
            "¿Cómo implementarías autenticación JWT en FastAPI?",
            "¿Cuál es la diferencia entre SQL JOIN y subquery?",
            "¿Cómo optimizarías una consulta SQL lenta?",
            "¿Qué patrón de diseño usarías para un sistema de notificaciones?"
        ]

        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"❓ {question}")
            print(f"{'='*60}\n")

            response = self.manager.generate(
                question,
                max_new_tokens=512,
                temperature=0.7
            )

            print(f"🤖 THAU: {response}\n")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     THAU Software Engineering Expert Training System         ║
╚══════════════════════════════════════════════════════════════╝

Este módulo entrena a THAU para ser un experto en:

✅ Backend Development (FastAPI, Django, Node.js)
✅ Frontend Development (React, Vue, TypeScript)
✅ Database Design (SQL, NoSQL, Optimización)
✅ API Design (REST, GraphQL, WebSocket)
✅ Best Practices (SOLID, Clean Code, Design Patterns)
✅ Algorithms & Data Structures
✅ Code Simplicity & Maintainability
✅ Common Sense Decision Making

""")

    # Inicializar trainer
    trainer = SoftwareEngineeringExpertTrainer(target_age=15)

    # Cargar datos de mejores prácticas
    trainer.load_best_practices_data()

    # Entrenar
    trainer.train(epochs=3, learning_rate=5e-5)

    # Probar conocimiento
    trainer.test_knowledge()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                     ✅ TRAINING COMPLETE                      ║
╚══════════════════════════════════════════════════════════════╝

THAU ahora tiene conocimiento especializado en:
- Desarrollo de software full-stack
- Mejores prácticas de ingeniería
- Código simple y mantenible
- Decisiones técnicas con sentido común

Puedes usar este modelo para:
1. Revisión de código
2. Sugerencias de arquitectura
3. Optimización de performance
4. Diseño de APIs
5. Resolución de problemas técnicos

El modelo busca siempre:
- La solución más simple que funcione
- Código legible y mantenible
- Eficiencia sin sacrificar claridad
- Sentido común en decisiones técnicas
""")


if __name__ == "__main__":
    main()
