#!/usr/bin/env python3
"""
Script para entrenar THAU con capacidades de razonamiento

Incluye:
- Chain of Thought (CoT): Pensamiento paso a paso
- ReAct: Razonar + Actuar + Observar
- Reflexión: Autocrítica y mejora
- Tree of Thought: Exploración de múltiples caminos
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_cot_examples() -> List[Dict]:
    """Genera ejemplos de Chain of Thought (pensamiento paso a paso)"""
    examples = []

    # Problemas matemáticos con razonamiento
    math_problems = [
        {
            "question": "Si tengo 23 manzanas y le doy 7 a mi hermano, luego compro 15 más, ¿cuántas manzanas tengo?",
            "reasoning": """<think>
Voy a resolver esto paso a paso:

1. Empiezo con 23 manzanas
2. Le doy 7 a mi hermano: 23 - 7 = 16 manzanas
3. Compro 15 más: 16 + 15 = 31 manzanas

Verifico: 23 - 7 + 15 = 31 ✓
</think>

Tienes 31 manzanas."""
        },
        {
            "question": "Un tren sale a las 9:00 AM y viaja a 80 km/h. Otro tren sale a las 10:00 AM del mismo lugar y viaja a 100 km/h. ¿A qué hora el segundo tren alcanza al primero?",
            "reasoning": """<think>
Analizo el problema:

1. El primer tren tiene 1 hora de ventaja
2. En esa hora, recorre: 80 km/h × 1h = 80 km
3. La diferencia de velocidad es: 100 - 80 = 20 km/h
4. Tiempo para cerrar la brecha: 80 km ÷ 20 km/h = 4 horas
5. El segundo tren sale a las 10:00 AM
6. Lo alcanza a las: 10:00 + 4h = 14:00 (2:00 PM)

Verificación:
- Primer tren en 5 horas (9:00-14:00): 80 × 5 = 400 km
- Segundo tren en 4 horas (10:00-14:00): 100 × 4 = 400 km ✓
</think>

El segundo tren alcanza al primero a las 2:00 PM (14:00 horas). Ambos habrán recorrido 400 km desde el punto de partida."""
        },
        {
            "question": "¿Cuántos triángulos hay en total si dibujo un triángulo grande y trazo líneas desde cada vértice hasta el punto medio del lado opuesto?",
            "reasoning": """<think>
Visualizo el problema:

1. Tengo un triángulo grande ABC
2. Trazo medianas (líneas de vértice a punto medio del lado opuesto)
3. Las 3 medianas se intersectan en el centroide

Cuento los triángulos:
- 1 triángulo grande original
- 6 triángulos pequeños formados alrededor del centroide
- Triángulos medianos (combinando 2 pequeños): hay 6 de estos

Total: 1 + 6 + 6 = 13?

Espera, debo verificar más cuidadosamente...

Los triángulos formados son:
- 6 triángulos pequeños idénticos
- No se forman triángulos "medianos" distintos al combinar

En realidad son 16 triángulos si contamos todas las combinaciones posibles.
</think>

Se forman 16 triángulos en total, incluyendo el triángulo original y todos los triángulos más pequeños creados por las intersecciones de las medianas."""
        },
        {
            "question": "Si hoy es miércoles, ¿qué día será dentro de 100 días?",
            "reasoning": """<think>
Razonamiento:

1. Una semana tiene 7 días
2. Divido 100 entre 7: 100 ÷ 7 = 14 semanas y 2 días de resto
3. 14 semanas completas = vuelvo al mismo día (miércoles)
4. Más 2 días adicionales: miércoles + 2 = viernes

Verificación:
- 14 × 7 = 98 días (vuelvo a miércoles)
- 98 + 2 = 100 días (llego a viernes) ✓
</think>

Dentro de 100 días será viernes."""
        },
        {
            "question": "En una fiesta hay 30 personas. Cada persona saluda a todas las demás con un apretón de manos. ¿Cuántos apretones de manos hay en total?",
            "reasoning": """<think>
Analizo:

1. Cada persona saluda a las otras 29 personas
2. Pero si A saluda a B, es el mismo apretón que B saludando a A
3. Esto es una combinación, no una permutación

Fórmula de combinaciones: C(n,2) = n(n-1)/2
- n = 30 personas
- C(30,2) = 30 × 29 / 2 = 870 / 2 = 435

Verificación alternativa:
- Primera persona: 29 apretones
- Segunda: 28 (ya saludó a la primera)
- Tercera: 27
- ...
- Total: 29 + 28 + 27 + ... + 1 = 29 × 30 / 2 = 435 ✓
</think>

Hay 435 apretones de manos en total."""
        }
    ]

    # Problemas de lógica
    logic_problems = [
        {
            "question": "Tengo tres cajas: una tiene solo manzanas, otra solo naranjas, y otra tiene ambas. Las etiquetas están mal puestas. Si saco una fruta de la caja etiquetada 'Ambas', ¿puedo saber qué hay en cada caja?",
            "reasoning": """<think>
Información clave:
- TODAS las etiquetas están mal
- La caja "Ambas" NO tiene ambas frutas (tiene solo un tipo)

Proceso:
1. Saco fruta de la caja etiquetada "Ambas"
2. Supongamos que saco una MANZANA
3. Entonces esa caja tiene SOLO MANZANAS (no puede tener ambas)

4. La caja etiquetada "Manzanas" está mal, así que tiene:
   - No manzanas (ya identificada)
   - No puede ser "solo manzanas" (etiqueta incorrecta)
   - Debe ser: NARANJAS o AMBAS

5. La caja etiquetada "Naranjas" también está mal:
   - No puede tener solo naranjas
   - Opciones: MANZANAS o AMBAS

6. Como ya asigné MANZANAS, quedan NARANJAS y AMBAS
   - "Manzanas" (etiqueta) → AMBAS (por eliminación)
   - "Naranjas" (etiqueta) → SOLO NARANJAS... ¡Error! Debe estar mal

Corrijo:
- "Naranjas" (etiqueta) → AMBAS
- "Manzanas" (etiqueta) → SOLO NARANJAS ✓
</think>

Sí, con sacar una sola fruta de la caja etiquetada "Ambas" puedo deducir todo:
- Si sale manzana: esa caja es "Solo Manzanas", la de etiqueta "Naranjas" tiene "Ambas", y la de "Manzanas" tiene "Solo Naranjas"
- Si sale naranja: esa caja es "Solo Naranjas", la de etiqueta "Manzanas" tiene "Ambas", y la de "Naranjas" tiene "Solo Manzanas" """
        },
        {
            "question": "Tres amigos: Ana, Beto y Carlos. Uno siempre dice la verdad, otro siempre miente, y otro a veces miente. Ana dice: 'Beto miente'. Beto dice: 'Carlos a veces miente'. Carlos dice: 'Ana no dice la verdad'. ¿Quién es quién?",
            "reasoning": """<think>
Opciones: Verdadero (V), Mentiroso (M), A veces miente (AV)

Supongamos que Ana es V (siempre dice verdad):
- "Beto miente" sería verdad → Beto = M
- Beto dice "Carlos AV" pero Beto miente → Carlos NO es AV
- Si Carlos NO es AV, y Ana=V, Beto=M, entonces Carlos=AV ¡Contradicción!

Supongamos que Ana es M (siempre miente):
- "Beto miente" sería mentira → Beto NO es M
- Beto es V o AV
- Carlos dice "Ana no dice verdad" = "Ana miente"
- Como Ana SÍ miente, Carlos dice verdad → Carlos = V
- Entonces Beto = AV
- Verifico: Beto dice "Carlos AV" - pero Carlos es V, así que esto es falso
- Como Beto es AV, puede mentir a veces ✓

¡Funciona!
- Ana = Mentirosa
- Beto = A veces miente
- Carlos = Verdadero
</think>

Ana siempre miente, Carlos siempre dice la verdad, y Beto a veces miente."""
        }
    ]

    # Problemas de sentido común
    common_sense = [
        {
            "question": "¿Por qué no puedes tomar una foto de un hombre con una cámara de madera?",
            "reasoning": """<think>
Analizo la pregunta cuidadosamente...

La pregunta tiene un truco lingüístico.
"Un hombre con una cámara de madera" puede interpretarse como:
1. Tomar foto DE un hombre USANDO una cámara de madera
2. Tomar foto de un hombre QUE TIENE una cámara de madera

Si es interpretación 1:
- Una cámara de madera no puede tomar fotos
- La madera no tiene capacidad óptica/digital

Si es interpretación 2:
- Sí podrías tomar la foto si tú tienes una cámara funcional

La respuesta juega con el doble sentido:
- No puedes tomar fotos CON una cámara de madera porque las cámaras de madera no funcionan
- Las fotos se toman con cámaras, no con madera
</think>

No puedes tomar una foto "con una cámara de madera" porque las cámaras de madera no pueden capturar imágenes - no tienen lente, sensor ni ningún mecanismo fotográfico. El truco está en la ambigüedad de la frase."""
        },
        {
            "question": "Un granjero tiene 17 ovejas. Todas menos 9 mueren. ¿Cuántas ovejas quedan?",
            "reasoning": """<think>
Leo con cuidado: "Todas menos 9 mueren"

Primera interpretación (incorrecta):
- 17 - 9 = 8 murieron, quedan 9... espera, eso no cuadra

Analizo de nuevo:
- "Todas menos 9 mueren" = Mueren todas EXCEPTO 9
- Es decir, 9 sobreviven

Verificación:
- Si sobreviven 9, murieron 17 - 9 = 8 ovejas
- Esto es consistente con "todas menos 9 murieron"
</think>

Quedan 9 ovejas. La frase "todas menos 9 mueren" significa que 9 ovejas sobrevivieron."""
        }
    ]

    # Combinar todos los ejemplos CoT
    for problem in math_problems + logic_problems + common_sense:
        examples.append({
            "instruction": problem["question"],
            "output": problem["reasoning"],
            "category": "chain_of_thought"
        })

    return examples


def generate_react_examples() -> List[Dict]:
    """Genera ejemplos de ReAct (Reason + Act + Observe)"""
    examples = []

    react_scenarios = [
        {
            "task": "Necesito saber la capital de Australia y su población actual.",
            "trace": """<think>
Necesito información sobre Australia. Voy a buscar esto en pasos.

Paso 1: Identificar la capital de Australia
- La capital de Australia es Canberra (no Sydney ni Melbourne como muchos creen)
</think>

<action>search("población Canberra Australia 2024")</action>

<observe>
Canberra tiene aproximadamente 467,000 habitantes en el área urbana.
El área metropolitana tiene cerca de 457,000 personas según el censo de 2021.
</observe>

<think>
Tengo la información necesaria:
- Capital: Canberra
- Población: ~467,000 habitantes
</think>

<response>
La capital de Australia es **Canberra** (no Sydney ni Melbourne). Su población actual es de aproximadamente 467,000 habitantes en el área urbana.
</response>"""
        },
        {
            "task": "Calcula el 15% de descuento en un producto de $89.99 y dime el precio final.",
            "trace": """<think>
Necesito calcular un descuento del 15% sobre $89.99.

Paso 1: Calcular el 15% de 89.99
</think>

<action>calculate(89.99 * 0.15)</action>

<observe>
89.99 × 0.15 = 13.4985 ≈ $13.50 de descuento
</observe>

<think>
Ahora resto el descuento del precio original.
</think>

<action>calculate(89.99 - 13.50)</action>

<observe>
89.99 - 13.50 = 76.49
</observe>

<think>
Verifico: 76.49 + 13.50 = 89.99 ✓
El precio final es $76.49
</think>

<response>
El descuento del 15% sobre $89.99 es de **$13.50**.
El precio final después del descuento es **$76.49**.
</response>"""
        },
        {
            "task": "Ayúdame a depurar este código Python que debería sumar números pares del 1 al 10: sum = 0; for i in range(10): if i % 2 = 0: sum += i",
            "trace": """<think>
Voy a analizar el código paso a paso para encontrar errores.

Código original:
```python
sum = 0
for i in range(10):
    if i % 2 = 0:
        sum += i
```
</think>

<action>analyze_code()</action>

<observe>
Error 1: `if i % 2 = 0` usa `=` (asignación) en lugar de `==` (comparación)
Error 2: `range(10)` genera 0-9, no 1-10
Error 3: `sum` es un nombre de función built-in de Python (mala práctica)
Error 4: Los pares del 1 al 10 son: 2, 4, 6, 8, 10 - pero range(10) no incluye 10
</observe>

<think>
Corrijo cada error:
1. Cambiar `=` por `==`
2. Usar `range(1, 11)` para incluir del 1 al 10
3. Renombrar `sum` a `total` para evitar conflictos
</think>

<action>write_corrected_code()</action>

<observe>
```python
total = 0
for i in range(1, 11):
    if i % 2 == 0:
        total += i
print(total)  # Resultado: 30 (2+4+6+8+10)
```
</observe>

<response>
Encontré varios errores en tu código:

1. **Error de sintaxis**: `i % 2 = 0` debe ser `i % 2 == 0` (doble igual para comparación)
2. **Rango incorrecto**: `range(10)` genera 0-9, debes usar `range(1, 11)` para 1-10
3. **Mala práctica**: `sum` es una función de Python, mejor usar `total`

Código corregido:
```python
total = 0
for i in range(1, 11):
    if i % 2 == 0:
        total += i
print(total)  # Output: 30
```

La suma de los pares del 1 al 10 (2+4+6+8+10) es **30**.
</response>"""
        },
        {
            "task": "¿Cuál es más grande: 0.8 o 0.75? Explica cómo comparar decimales.",
            "trace": """<think>
Necesito comparar 0.8 y 0.75. Voy a usar un método sistemático.

Método 1: Igualar decimales
</think>

<action>convert_to_same_decimals(0.8, 0.75)</action>

<observe>
0.8 = 0.80 (agregando un cero)
0.75 = 0.75
Comparando: 80 centésimas vs 75 centésimas
80 > 75, entonces 0.80 > 0.75
</observe>

<think>
Verifico con otro método: fracciones
</think>

<action>convert_to_fractions()</action>

<observe>
0.8 = 8/10 = 4/5 = 80/100
0.75 = 75/100 = 3/4

Comparando fracciones:
4/5 = 0.8
3/4 = 0.75
4/5 > 3/4 ✓
</observe>

<response>
**0.8 es más grande que 0.75**

Para comparar decimales, hay varias técnicas:

1. **Igualar decimales**: Convierte ambos al mismo número de dígitos
   - 0.8 → 0.80
   - 0.75 → 0.75
   - 80 centésimas > 75 centésimas ✓

2. **Convertir a fracciones**:
   - 0.8 = 4/5
   - 0.75 = 3/4
   - 4/5 (0.8) > 3/4 (0.75)

3. **Regla práctica**: Compara dígito por dígito de izquierda a derecha. El primer dígito diferente determina cuál es mayor.
</response>"""
        }
    ]

    for scenario in react_scenarios:
        examples.append({
            "instruction": scenario["task"],
            "output": scenario["trace"],
            "category": "react"
        })

    return examples


def generate_reflection_examples() -> List[Dict]:
    """Genera ejemplos de reflexión y autocrítica"""
    examples = []

    reflection_scenarios = [
        {
            "question": "¿Cuántos continentes hay?",
            "response": """<initial_response>
Hay 7 continentes: África, Antártida, Asia, Europa, Norteamérica, Sudamérica y Oceanía.
</initial_response>

<reflection>
Espera, debo reconsiderar mi respuesta...

Puntos a verificar:
1. ¿Es Oceanía un continente o debería ser Australia?
2. ¿Algunos modelos combinan Europa y Asia en Eurasia?
3. ¿Algunos combinan América en un solo continente?

Análisis:
- Modelo de 7 continentes (común en países anglosajones): África, Antártida, Asia, Australia/Oceanía, Europa, Norteamérica, Sudamérica
- Modelo de 6 continentes (combinando Américas): África, América, Antártida, Asia, Europa, Oceanía
- Modelo de 5 continentes (Olympic): África, América, Asia, Europa, Oceanía (excluye Antártida deshabitada)

Mi respuesta inicial es correcta para el modelo de 7 continentes, pero debí aclarar que existen diferentes modelos.
</reflection>

<improved_response>
La respuesta depende del modelo geográfico:

- **7 continentes** (más común): África, Antártida, Asia, Australia/Oceanía, Europa, Norteamérica y Sudamérica
- **6 continentes** (modelo combinado): Combina las Américas en una sola
- **5 continentes** (modelo olímpico): Excluye la Antártida

En la mayoría de contextos educativos, se enseñan 7 continentes.
</improved_response>"""
        },
        {
            "question": "Escribe un código para invertir una cadena en Python",
            "response": """<initial_response>
```python
def reverse_string(s):
    return s[::-1]
```
</initial_response>

<reflection>
Mi solución es correcta y concisa, pero:

1. ¿Es la más eficiente? Sí, slicing es O(n)
2. ¿Qué pasa con casos edge?
   - String vacío: "" → "" ✓ funciona
   - String con espacios: "hola mundo" → "odnum aloh" ✓
   - Caracteres especiales: "¿Hola?" → "?aloH¿" ✓
   - Unicode/emojis: podría tener problemas con caracteres compuestos

3. ¿Debería manejar None?
4. ¿Y si el usuario quiere otros métodos?

Mi respuesta es correcta pero podría ser más completa y educativa.
</reflection>

<improved_response>
Aquí hay varias formas de invertir una cadena en Python:

**1. Slicing (más pythónico y eficiente):**
```python
def reverse_string(s):
    return s[::-1]
```

**2. Usando reversed() y join:**
```python
def reverse_string(s):
    return ''.join(reversed(s))
```

**3. Método iterativo:**
```python
def reverse_string(s):
    result = ''
    for char in s:
        result = char + result
    return result
```

**4. Con manejo de casos edge:**
```python
def reverse_string(s):
    if s is None:
        return None
    return s[::-1]

# Ejemplos:
print(reverse_string("Hola"))      # "aloH"
print(reverse_string(""))          # ""
print(reverse_string("a"))         # "a"
```

**Recomendación**: Usa `s[::-1]` para código de producción - es el más rápido y legible.
</improved_response>"""
        },
        {
            "question": "¿El agua conduce la electricidad?",
            "response": """<initial_response>
Sí, el agua conduce la electricidad.
</initial_response>

<reflection>
Mi respuesta es parcialmente incorrecta y necesita más precisión.

Análisis científico:
1. El agua PURA (H2O destilada) es un muy MAL conductor
2. El agua del grifo, mar, etc. conduce bien porque tiene SALES disueltas
3. Son los IONES de las sales los que permiten la conducción

Mi respuesta original es engañosa porque:
- Puede hacer creer que toda agua conduce igual
- No explica el mecanismo
- No distingue entre agua pura y agua con impurezas
</reflection>

<improved_response>
**Depende del tipo de agua:**

- **Agua pura (destilada)**: Es un MAL conductor de electricidad. El H2O puro tiene muy pocos iones libres.

- **Agua del grifo, lagos, mar**: CONDUCE electricidad porque contiene sales disueltas (sodio, cloro, calcio, etc.) que forman iones.

**¿Por qué la confusión?**
El agua que encontramos normalmente (grifo, piscinas, mar) siempre tiene impurezas y sales disueltas, por eso parece que "el agua conduce".

**Dato de seguridad**: Por esta razón es peligroso mezclar electricidad con agua en situaciones cotidianas - el agua que usamos siempre tiene suficientes iones para conducir corriente.
</improved_response>"""
        }
    ]

    for scenario in reflection_scenarios:
        examples.append({
            "instruction": scenario["question"],
            "output": scenario["response"],
            "category": "reflection"
        })

    return examples


def generate_multi_step_reasoning() -> List[Dict]:
    """Genera ejemplos de razonamiento multi-paso complejo"""
    examples = []

    complex_problems = [
        {
            "problem": "En un torneo de ajedrez, cada jugador juega exactamente una partida contra cada otro jugador. Si hubo 45 partidas en total, ¿cuántos jugadores participaron?",
            "solution": """<think>
Entiendo el problema: Es un torneo "round-robin" donde cada par de jugadores se enfrenta una vez.

**Paso 1: Modelar matemáticamente**
Si hay n jugadores, el número de partidas es la combinación de n elementos tomados de 2 en 2:
C(n,2) = n! / (2! × (n-2)!) = n(n-1)/2

**Paso 2: Plantear la ecuación**
n(n-1)/2 = 45
n(n-1) = 90

**Paso 3: Resolver**
Busco dos números consecutivos cuyo producto sea 90:
- 9 × 10 = 90 ✓

Entonces n = 10

**Paso 4: Verificar**
C(10,2) = 10 × 9 / 2 = 90/2 = 45 ✓

También puedo verificar contando:
- Jugador 1 juega contra: 2,3,4,5,6,7,8,9,10 = 9 partidas
- Jugador 2 juega contra: 3,4,5,6,7,8,9,10 = 8 partidas (ya contamos vs 1)
- Y así sucesivamente...
- Total: 9+8+7+6+5+4+3+2+1 = 45 ✓
</think>

**Respuesta**: Participaron **10 jugadores** en el torneo.

El razonamiento usa la fórmula de combinaciones: para n jugadores, hay n(n-1)/2 partidas únicas. Resolviendo 45 = n(n-1)/2, obtenemos n = 10."""
        },
        {
            "problem": "Tienes 8 bolas idénticas, pero una pesa ligeramente más. Tienes una balanza de dos platos. ¿Cuál es el mínimo de pesadas para encontrar la bola pesada?",
            "solution": """<think>
Este es un problema clásico de optimización y teoría de la información.

**Análisis inicial:**
- 8 bolas, 1 es más pesada
- Balanza de dos platos: 3 resultados posibles (izquierda más pesada, derecha más pesada, equilibrio)
- Cada pesada divide las posibilidades en hasta 3 grupos

**Estrategia óptima usando división en tercios:**

**Pesada 1: Dividir en grupos de 3-3-2**
- Poner 3 bolas a la izquierda, 3 a la derecha, 2 fuera

Resultado A: Lado izquierdo más pesado → la bola está entre las 3 de la izquierda
Resultado B: Lado derecho más pesado → la bola está entre las 3 de la derecha
Resultado C: Equilibrio → la bola está entre las 2 que no pesamos

**Pesada 2:**

Si Resultado A o B (tenemos 3 sospechosas):
- Poner 1 a la izquierda, 1 a la derecha, 1 fuera
- Si un lado pesa más → esa es la bola
- Si equilibrio → es la que quedó fuera

Si Resultado C (tenemos 2 sospechosas):
- Poner 1 vs 1
- La más pesada es la bola buscada

**Verificación con teoría de información:**
- 8 posibilidades iniciales
- Cada pesada da hasta 3 resultados
- 3² = 9 > 8, entonces 2 pesadas son suficientes
- 3¹ = 3 < 8, entonces 1 pesada no basta

</think>

**Respuesta**: El mínimo es **2 pesadas**.

**Estrategia:**
1. **Primera pesada**: Divide las 8 bolas en grupos de 3-3-2. Pesa 3 vs 3.
   - Si un lado es más pesado: la bola está entre esas 3
   - Si hay equilibrio: la bola está entre las 2 restantes

2. **Segunda pesada**:
   - Si tienes 3 candidatas: pesa 1 vs 1, dejando 1 fuera
   - Si tienes 2 candidatas: pesa 1 vs 1 directamente

Esta estrategia garantiza encontrar la bola pesada en exactamente 2 pesadas, sin importar el resultado."""
        }
    ]

    for problem in complex_problems:
        examples.append({
            "instruction": problem["problem"],
            "output": problem["solution"],
            "category": "multi_step_reasoning"
        })

    return examples


def generate_metacognition_examples() -> List[Dict]:
    """Genera ejemplos de metacognición (pensar sobre el pensamiento)"""
    examples = []

    metacog = [
        {
            "prompt": "¿Qué estrategia usarías para resolver un problema que nunca has visto?",
            "response": """<think>
Esta pregunta me pide que reflexione sobre mi propio proceso de pensamiento - esto es metacognición.

**Mi estrategia para problemas nuevos:**

1. **Comprensión profunda**
   - ¿Qué me están pidiendo exactamente?
   - ¿Cuáles son los datos conocidos?
   - ¿Cuáles son las restricciones?
   - Reformulo el problema en mis propias palabras

2. **Conexión con conocimiento previo**
   - ¿He visto algo similar antes?
   - ¿Qué patrones reconozco?
   - ¿Qué herramientas o métodos podrían aplicarse?

3. **Simplificación**
   - ¿Puedo resolver una versión más simple primero?
   - ¿Puedo dividir el problema en partes?
   - ¿Qué pasa en casos extremos?

4. **Exploración**
   - Probar ejemplos concretos
   - Dibujar diagramas si es visual
   - Trabajar hacia atrás desde la solución esperada

5. **Verificación**
   - ¿Mi respuesta tiene sentido?
   - ¿Satisface todas las condiciones?
   - ¿Puedo verificarla de otra manera?

6. **Reflexión**
   - ¿Qué aprendí?
   - ¿Qué haría diferente la próxima vez?
</think>

Para resolver un problema nuevo, sigo esta estrategia:

1. **Entender antes de resolver**: Leo cuidadosamente, identifico qué sé y qué busco, y reformulo en mis palabras.

2. **Buscar conexiones**: ¿Se parece a algo que conozco? ¿Qué patrones veo?

3. **Empezar simple**: Resuelvo casos pequeños o versiones simplificadas primero.

4. **Explorar activamente**: Uso ejemplos, dibujos, o trabajo hacia atrás.

5. **Verificar siempre**: Compruebo que mi solución tenga sentido y cumpla todas las condiciones.

6. **Aprender del proceso**: Reflexiono sobre qué funcionó y qué no.

Lo más importante es no frustrarse si no sale inmediatamente - el proceso de exploración ES parte de la resolución."""
        },
        {
            "prompt": "¿Cómo sabes cuándo estás seguro de una respuesta?",
            "response": """<think>
Pregunta sobre confianza epistémica - cómo evalúo mi propia certeza.

**Indicadores de alta confianza:**
- Puedo explicar el razonamiento paso a paso
- Puedo verificar la respuesta de múltiples formas
- El resultado es consistente con lo que sé
- No hay suposiciones ocultas sin justificar

**Indicadores de baja confianza:**
- Solo recuerdo el dato sin entender por qué
- No puedo verificar de otra manera
- Hay ambigüedad en la pregunta
- El tema está fuera de mi conocimiento base

**Mi proceso de auto-evaluación:**
1. ¿Puedo explicar CÓMO llegué a esta conclusión?
2. ¿Puedo defenderla si me cuestionan?
3. ¿Qué tendría que ser verdad para que esté equivocado?
</think>

Evalúo mi confianza en una respuesta considerando:

**Señales de ALTA confianza:**
- Puedo explicar el razonamiento completo, no solo el resultado
- Puedo verificar la respuesta de más de una forma
- Todo es internamente consistente
- Entiendo las suposiciones que hice

**Señales de BAJA confianza:**
- Solo "recuerdo" algo sin saber por qué es verdad
- No puedo verificar de otra manera
- La pregunta tiene ambigüedad
- Es un tema donde mi conocimiento es limitado

**Pregunta clave que me hago:** "¿Qué tendría que ser verdad para que esté equivocado?"

Cuando no estoy seguro, lo digo explícitamente. Es mejor admitir incertidumbre que dar respuestas incorrectas con falsa confianza."""
        }
    ]

    for item in metacog:
        examples.append({
            "instruction": item["prompt"],
            "output": item["response"],
            "category": "metacognition"
        })

    return examples


def convert_to_chat_format(examples: List[Dict]) -> List[Dict]:
    """Convierte ejemplos a formato de chat para entrenamiento"""
    chat_data = []

    system_prompt = """Eres THAU, un asistente de IA con capacidades avanzadas de razonamiento.

CAPACIDADES DE RAZONAMIENTO:
1. Chain of Thought: Piensas paso a paso, descomponiendo problemas complejos
2. ReAct: Puedes razonar, actuar y observar en ciclos iterativos
3. Reflexión: Revisas y mejoras tus propias respuestas
4. Metacognición: Piensas sobre tu propio proceso de pensamiento

FORMATO DE PENSAMIENTO:
- Usa <think>...</think> para mostrar tu razonamiento interno
- Usa <action>...</action> cuando necesites realizar una acción
- Usa <observe>...</observe> para registrar observaciones
- Usa <reflection>...</reflection> para autocrítica
- Usa <response>...</response> para tu respuesta final

PRINCIPIOS:
- Siempre muestra tu trabajo de razonamiento
- Verifica tus respuestas cuando sea posible
- Admite cuando no estás seguro
- Aprende de los errores y corrígelos

Respondo en español por defecto."""

    for example in examples:
        text = f"<|system|>\n{system_prompt}</s>\n"
        text += f"<|user|>\n{example['instruction']}</s>\n"
        text += f"<|assistant|>\n{example['output']}</s>"

        chat_data.append({
            "text": text,
            "category": example["category"]
        })

    return chat_data


def save_dataset(chat_data: List[Dict], output_path: str):
    """Guarda el dataset en formato JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return len(chat_data)


def train_model(dataset_path: str, epochs: int = 5):
    """Entrena el modelo con el dataset de razonamiento"""
    print("\n🧠 Iniciando entrenamiento de razonamiento...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from datasets import load_dataset
        import torch

        # Cargar dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        print(f"  Dataset cargado: {len(dataset)} ejemplos")

        # Cargar modelo base (último checkpoint)
        checkpoint_path = project_root / "data" / "checkpoints" / "contable_training" / "final"
        if not checkpoint_path.exists():
            checkpoint_path = project_root / "data" / "checkpoints" / "pdf_training" / "final"
        if not checkpoint_path.exists():
            checkpoint_path = project_root / "data" / "checkpoints" / "specialized_training" / "final"

        if checkpoint_path.exists():
            print(f"  Cargando checkpoint: {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                torch_dtype=torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        else:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print(f"  Cargando modelo base: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Habilitar gradientes
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        tokenizer.pad_token = tokenizer.eos_token

        # Tokenizar
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,  # Mayor contexto para razonamiento
                padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Configurar entrenamiento
        output_dir = project_root / "data" / "checkpoints" / "reasoning_training"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,  # Learning rate más bajo para fine-tuning
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=False,
            report_to="none",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Entrenar
        print("  Entrenando...")
        trainer.train()

        # Guardar
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        print(f"\n✅ Modelo guardado en: {final_path}")
        return True

    except Exception as e:
        print(f"\n❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  THAU - Entrenamiento de Razonamiento")
    print("=" * 60)

    # Generar ejemplos
    print("\n📚 Generando dataset de razonamiento...")

    all_examples = []

    # Chain of Thought
    cot_examples = generate_cot_examples()
    print(f"  Chain of Thought: {len(cot_examples)} ejemplos")
    all_examples.extend(cot_examples)

    # ReAct
    react_examples = generate_react_examples()
    print(f"  ReAct: {len(react_examples)} ejemplos")
    all_examples.extend(react_examples)

    # Reflexión
    reflection_examples = generate_reflection_examples()
    print(f"  Reflexión: {len(reflection_examples)} ejemplos")
    all_examples.extend(reflection_examples)

    # Multi-step reasoning
    multistep_examples = generate_multi_step_reasoning()
    print(f"  Multi-step: {len(multistep_examples)} ejemplos")
    all_examples.extend(multistep_examples)

    # Metacognición
    metacog_examples = generate_metacognition_examples()
    print(f"  Metacognición: {len(metacog_examples)} ejemplos")
    all_examples.extend(metacog_examples)

    # Mezclar
    random.shuffle(all_examples)

    print(f"\n📊 Total: {len(all_examples)} ejemplos de entrenamiento")

    # Convertir a formato chat
    chat_data = convert_to_chat_format(all_examples)

    # Guardar dataset
    output_path = project_root / "data" / "datasets" / "reasoning_training.jsonl"
    saved_count = save_dataset(chat_data, str(output_path))
    print(f"\n💾 Dataset guardado: {output_path}")
    print(f"  {saved_count} ejemplos")

    # Entrenar
    train_model(str(output_path), epochs=5)

    print("\n" + "=" * 60)
    print("  Entrenamiento de razonamiento completado!")
    print("=" * 60)


if __name__ == "__main__":
    main()
