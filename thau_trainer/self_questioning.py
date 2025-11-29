"""
Sistema de Auto-Cuestionamiento Autónomo para THAU
Permite que THAU genere preguntas para sí mismo y aprenda de forma autónoma
"""

import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import random
import requests


class SelfQuestioningSystem:
    """
    Sistema que permite a THAU hacerse preguntas a sí mismo
    para aprender de forma autónoma con límites de seguridad
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:1.5b-base",
        max_questions_per_hour: int = 60,  # Aumentado de 10 a 60 (1 por minuto)
        max_questions_per_day: int = 1000,  # Aumentado de 100 a 1000
        data_dir: Path = Path("./data/self_questioning"),
        min_seconds_between_questions: int = 5  # Reducido de 30 a 5 segundos
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.max_questions_per_hour = max_questions_per_hour
        self.max_questions_per_day = max_questions_per_day
        self.min_seconds_between_questions = min_seconds_between_questions
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Registro de actividad
        self.activity_log = self._load_activity_log()

        # Temas de exploración progresivos por edad cognitiva - EXPANDIDO
        self.topic_templates = {
            0: [  # Recién nacido - Conceptos básicos
                "¿Qué es {concept}?",
                "¿Cómo se usa {concept}?",
                "¿Para qué sirve {concept}?",
                "Define {concept} de forma simple",
                "¿Qué significa {concept}?",
            ],
            1: [  # 1 año - Comparaciones simples
                "¿Cuál es la diferencia entre {concept1} y {concept2}?",
                "¿Cómo funciona {concept}?",
                "¿Qué hace {concept}?",
                "Dame un ejemplo de {concept}",
                "¿Cuándo usar {concept}?",
                "Explica {concept} con un ejemplo",
            ],
            2: [  # 2 años - Contexto y uso
                "¿Por qué es importante {concept}?",
                "¿Cuándo se usa {concept}?",
                "¿Dónde encuentro {concept}?",
                "¿Qué pasa si no uso {concept}?",
                "¿Cómo implemento {concept} en código?",
                "Lista 3 casos de uso de {concept}",
            ],
            3: [  # 3 años - Relaciones
                "¿Cómo se relaciona {concept1} con {concept2}?",
                "¿Cuáles son las ventajas de {concept}?",
                "¿Qué problemas resuelve {concept}?",
                "Compara {concept1} vs {concept2}: ventajas y desventajas",
                "¿Cómo {concept} mejora el rendimiento?",
                "¿Qué alternativas hay a {concept}?",
            ],
            6: [  # 6 años - Análisis técnico
                "Explica la arquitectura de {concept}",
                "¿Cómo se implementa {concept} internamente?",
                "¿Cuáles son los trade-offs de usar {concept}?",
                "¿Cómo optimizar {concept} para producción?",
                "¿Qué patrones de diseño usa {concept}?",
                "Explica el ciclo de vida de {concept}",
                "¿Cómo hacer testing de {concept}?",
            ],
            11: [  # 11 años - Razonamiento avanzado
                "Diseña un sistema usando {concept1} y {concept2}",
                "¿Cómo escalar {concept} para millones de usuarios?",
                "Analiza los problemas de seguridad en {concept}",
                "¿Cómo depurar problemas complejos con {concept}?",
                "Explica {concept} desde múltiples perspectivas",
                "¿Cuál es el futuro de {concept}?",
                "Critica la implementación estándar de {concept}",
            ],
            15: [  # 15 años - Expertise
                "Escribe código profesional implementando {concept}",
                "Diseña la arquitectura completa de un sistema con {concept}",
                "¿Cómo contribuiría al desarrollo de {concept}?",
                "Analiza paper académico sobre {concept}",
                "¿Cómo enseñarías {concept} a principiantes?",
                "Propón mejoras al estado del arte de {concept}",
            ]
        }

        # Base de conocimiento expandida - 500+ conceptos organizados por categoría
        self.exploration_concepts = self._build_knowledge_base()

        # Categorías temáticas para exploración dirigida
        self.knowledge_categories = self._build_categories()

    def _build_knowledge_base(self) -> List[str]:
        """Construye base de conocimiento masiva con 500+ conceptos"""
        concepts = []

        # ===== PROGRAMACIÓN FUNDAMENTAL =====
        concepts.extend([
            "variable", "constante", "tipo de dato", "string", "entero", "flotante", "booleano",
            "función", "método", "parámetro", "argumento", "retorno", "return", "void",
            "loop", "for", "while", "do-while", "foreach", "iteración", "bucle infinito",
            "condicional", "if", "else", "switch", "case", "operador ternario",
            "array", "lista", "diccionario", "mapa", "conjunto", "set", "tupla", "cola", "pila",
            "objeto", "clase", "instancia", "constructor", "destructor", "propiedad", "atributo",
            "herencia", "polimorfismo", "encapsulamiento", "abstracción", "interfaz", "clase abstracta",
            "módulo", "paquete", "import", "export", "namespace", "scope", "ámbito",
            "excepción", "try", "catch", "finally", "throw", "error handling",
            "recursión", "callback", "closure", "lambda", "función anónima", "higher-order function",
        ])

        # ===== ESTRUCTURAS DE DATOS =====
        concepts.extend([
            "árbol binario", "árbol de búsqueda", "árbol AVL", "árbol rojo-negro", "B-tree",
            "grafo", "grafo dirigido", "grafo no dirigido", "grafo ponderado", "matriz de adyacencia",
            "lista enlazada", "lista doblemente enlazada", "lista circular",
            "heap", "min-heap", "max-heap", "priority queue",
            "hash table", "hash map", "colisión", "función hash", "tabla de dispersión",
            "trie", "suffix tree", "bloom filter", "skip list",
            "stack", "queue", "deque", "circular buffer", "ring buffer",
        ])

        # ===== ALGORITMOS =====
        concepts.extend([
            "algoritmo", "complejidad algorítmica", "Big O", "O(1)", "O(n)", "O(log n)", "O(n²)",
            "bubble sort", "quick sort", "merge sort", "heap sort", "insertion sort", "selection sort",
            "binary search", "linear search", "depth-first search", "breadth-first search",
            "dijkstra", "A*", "bellman-ford", "floyd-warshall", "algoritmo de Prim", "algoritmo de Kruskal",
            "programación dinámica", "divide y vencerás", "greedy algorithm", "backtracking",
            "two pointers", "sliding window", "prefix sum", "memoización",
        ])

        # ===== BASES DE DATOS =====
        concepts.extend([
            "base de datos", "SQL", "NoSQL", "relacional", "no relacional",
            "tabla", "columna", "fila", "registro", "campo", "clave primaria", "clave foránea",
            "SELECT", "INSERT", "UPDATE", "DELETE", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
            "índice", "índice compuesto", "índice único", "full-text index",
            "normalización", "1NF", "2NF", "3NF", "BCNF", "desnormalización",
            "transacción", "ACID", "atomicidad", "consistencia", "aislamiento", "durabilidad",
            "stored procedure", "trigger", "view", "materialized view",
            "MongoDB", "PostgreSQL", "MySQL", "Redis", "Cassandra", "DynamoDB", "Elasticsearch",
            "sharding", "replicación", "particionamiento", "clustering",
            "ORM", "query builder", "migration", "schema", "seed",
        ])

        # ===== DESARROLLO WEB =====
        concepts.extend([
            "HTML", "CSS", "JavaScript", "DOM", "evento", "listener",
            "HTTP", "HTTPS", "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS",
            "REST", "RESTful API", "GraphQL", "WebSocket", "Server-Sent Events",
            "JSON", "XML", "YAML", "Protobuf", "serialización", "deserialización",
            "cookie", "session", "localStorage", "sessionStorage", "IndexedDB",
            "CORS", "CSRF", "XSS", "SQL injection", "sanitización",
            "frontend", "backend", "fullstack", "SPA", "SSR", "SSG", "ISR",
            "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt.js",
            "Node.js", "Express", "FastAPI", "Django", "Flask", "Spring Boot",
            "webpack", "vite", "babel", "transpilación", "bundling", "minificación",
            "responsive design", "mobile-first", "flexbox", "grid", "media queries",
        ])

        # ===== CLOUD Y DEVOPS =====
        concepts.extend([
            "cloud computing", "AWS", "Azure", "Google Cloud", "DigitalOcean",
            "servidor", "instancia", "virtual machine", "contenedor", "Docker", "Kubernetes",
            "microservicio", "monolito", "serverless", "FaaS", "Lambda", "Cloud Functions",
            "CI/CD", "integración continua", "despliegue continuo", "pipeline",
            "Git", "GitHub", "GitLab", "branch", "merge", "pull request", "code review",
            "Jenkins", "GitHub Actions", "CircleCI", "Travis CI",
            "Terraform", "Ansible", "Puppet", "Chef", "Infrastructure as Code",
            "load balancer", "reverse proxy", "CDN", "Nginx", "Apache",
            "monitoreo", "logging", "métricas", "alertas", "Prometheus", "Grafana",
            "escalabilidad horizontal", "escalabilidad vertical", "auto-scaling",
        ])

        # ===== SEGURIDAD =====
        concepts.extend([
            "autenticación", "autorización", "JWT", "OAuth", "OAuth2", "OpenID Connect",
            "encriptación", "hash", "salt", "bcrypt", "argon2", "SHA-256", "MD5",
            "SSL", "TLS", "certificado", "HTTPS", "Let's Encrypt",
            "firewall", "WAF", "DDoS", "rate limiting", "throttling",
            "OWASP", "vulnerabilidad", "CVE", "penetration testing", "ethical hacking",
            "2FA", "MFA", "biometría", "token", "API key", "secret management",
            "cifrado simétrico", "cifrado asimétrico", "RSA", "AES", "clave pública", "clave privada",
        ])

        # ===== INTELIGENCIA ARTIFICIAL Y ML =====
        concepts.extend([
            "inteligencia artificial", "machine learning", "deep learning", "neural network",
            "supervised learning", "unsupervised learning", "reinforcement learning",
            "clasificación", "regresión", "clustering", "dimensionality reduction",
            "decision tree", "random forest", "SVM", "k-means", "k-NN",
            "red neuronal", "perceptrón", "capa oculta", "función de activación", "backpropagation",
            "CNN", "RNN", "LSTM", "GRU", "transformer", "attention mechanism",
            "GPT", "BERT", "embedding", "tokenización", "word2vec", "fine-tuning",
            "overfitting", "underfitting", "regularización", "dropout", "batch normalization",
            "gradient descent", "learning rate", "optimizer", "Adam", "SGD",
            "loss function", "accuracy", "precision", "recall", "F1 score", "ROC", "AUC",
            "PyTorch", "TensorFlow", "Keras", "scikit-learn", "Hugging Face",
            "NLP", "computer vision", "speech recognition", "generative AI", "LLM",
            "RAG", "vector database", "embedding", "semantic search", "ChromaDB", "Pinecone",
        ])

        # ===== MATEMÁTICAS FUNDAMENTALES =====
        concepts.extend([
            # Aritmética
            "suma", "resta", "multiplicación", "división", "potencia", "raíz cuadrada",
            "número natural", "número entero", "número racional", "número irracional", "número real",
            "número complejo", "fracción", "decimal", "porcentaje", "proporción", "razón",
            "múltiplo", "divisor", "máximo común divisor", "mínimo común múltiplo",
            "número primo", "número compuesto", "factorización", "factorial", "fibonacci",

            # Álgebra
            "álgebra", "ecuación", "inecuación", "variable", "constante", "coeficiente",
            "polinomio", "monomio", "binomio", "trinomio", "grado de polinomio",
            "ecuación lineal", "ecuación cuadrática", "fórmula cuadrática", "discriminante",
            "sistema de ecuaciones", "método de sustitución", "método de eliminación",
            "función", "dominio", "rango", "imagen", "función inversa", "composición de funciones",
            "función lineal", "función cuadrática", "función exponencial", "función logarítmica",
            "logaritmo", "logaritmo natural", "logaritmo base 10", "propiedades de logaritmos",
            "sucesión", "serie", "progresión aritmética", "progresión geométrica",
            "límite", "convergencia", "divergencia", "serie infinita",

            # Álgebra Lineal
            "álgebra lineal", "matriz", "vector", "escalar", "producto punto", "producto cruz",
            "determinante", "matriz inversa", "matriz transpuesta", "matriz identidad",
            "autovalor", "autovector", "diagonalización", "espacio vectorial",
            "transformación lineal", "núcleo", "imagen", "rango de matriz",
            "sistema de ecuaciones lineales", "método de Gauss", "eliminación gaussiana",

            # Geometría
            "geometría", "punto", "línea", "plano", "segmento", "rayo", "ángulo",
            "ángulo agudo", "ángulo recto", "ángulo obtuso", "ángulo llano",
            "triángulo", "cuadrilátero", "polígono", "círculo", "circunferencia",
            "triángulo equilátero", "triángulo isósceles", "triángulo escaleno",
            "triángulo rectángulo", "teorema de Pitágoras", "hipotenusa", "cateto",
            "perímetro", "área", "volumen", "superficie",
            "cuadrado", "rectángulo", "rombo", "trapecio", "paralelogramo",
            "cubo", "esfera", "cilindro", "cono", "pirámide", "prisma",
            "coordenadas cartesianas", "coordenadas polares", "distancia entre puntos",
            "pendiente", "ecuación de la recta", "rectas paralelas", "rectas perpendiculares",

            # Trigonometría
            "trigonometría", "seno", "coseno", "tangente", "cotangente", "secante", "cosecante",
            "identidades trigonométricas", "ley de senos", "ley de cosenos",
            "radianes", "grados", "círculo unitario", "funciones trigonométricas inversas",

            # Cálculo
            "cálculo", "límite", "continuidad", "derivada", "integral",
            "derivada parcial", "gradiente", "hessiano", "jacobiano",
            "regla de la cadena", "regla del producto", "regla del cociente",
            "integral definida", "integral indefinida", "teorema fundamental del cálculo",
            "integración por partes", "integración por sustitución",
            "ecuación diferencial", "ecuación diferencial ordinaria", "ecuación diferencial parcial",
            "serie de Taylor", "serie de Fourier", "transformada de Laplace",

            # Probabilidad y Estadística
            "probabilidad", "evento", "espacio muestral", "probabilidad condicional",
            "independencia", "teorema de Bayes", "ley de probabilidad total",
            "variable aleatoria", "distribución de probabilidad", "valor esperado",
            "varianza", "desviación estándar", "covarianza", "correlación",
            "distribución normal", "distribución binomial", "distribución de Poisson",
            "distribución uniforme", "distribución exponencial", "distribución t de Student",
            "media", "mediana", "moda", "cuartiles", "percentiles",
            "histograma", "diagrama de caja", "regresión lineal", "regresión múltiple",
            "hipótesis nula", "hipótesis alternativa", "p-valor", "intervalo de confianza",
            "estadística descriptiva", "estadística inferencial", "muestreo",

            # Matemáticas Discretas
            "matemáticas discretas", "teoría de conjuntos", "unión", "intersección", "complemento",
            "relación", "función", "inyectiva", "sobreyectiva", "biyectiva",
            "teoría de grafos", "vértice", "arista", "camino", "ciclo", "árbol",
            "combinatoria", "permutación", "combinación", "principio del palomar",
            "inducción matemática", "recursión", "relación de recurrencia",
            "lógica matemática", "proposición", "conectivos lógicos", "tabla de verdad",
            "cuantificador universal", "cuantificador existencial", "demostración",

            # Teoría de Números
            "teoría de números", "divisibilidad", "congruencia", "aritmética modular",
            "teorema de Fermat", "teorema de Euler", "función phi de Euler",
            "números de Mersenne", "números perfectos", "conjetura de Goldbach",
        ])

        # ===== FÍSICA =====
        concepts.extend([
            # Mecánica
            "física", "mecánica", "cinemática", "dinámica", "estática",
            "posición", "velocidad", "aceleración", "desplazamiento", "trayectoria",
            "movimiento rectilíneo uniforme", "movimiento rectilíneo uniformemente acelerado",
            "caída libre", "tiro parabólico", "movimiento circular", "velocidad angular",
            "fuerza", "masa", "peso", "newton", "leyes de Newton",
            "primera ley de Newton", "segunda ley de Newton", "tercera ley de Newton",
            "fricción", "fuerza normal", "tensión", "fuerza centrípeta",
            "trabajo", "energía", "energía cinética", "energía potencial", "potencia",
            "conservación de la energía", "teorema trabajo-energía",
            "momento lineal", "impulso", "conservación del momento", "colisión elástica",
            "torque", "momento de inercia", "rotación", "centro de masa",

            # Termodinámica
            "termodinámica", "temperatura", "calor", "energía interna",
            "primera ley de la termodinámica", "segunda ley de la termodinámica",
            "entropía", "entalpía", "proceso isotérmico", "proceso adiabático",
            "proceso isobárico", "proceso isocórico", "ciclo de Carnot",
            "transferencia de calor", "conducción", "convección", "radiación",
            "capacidad calorífica", "calor específico", "cambio de fase",

            # Ondas y Sonido
            "onda", "frecuencia", "período", "longitud de onda", "amplitud",
            "onda transversal", "onda longitudinal", "onda mecánica", "onda electromagnética",
            "sonido", "velocidad del sonido", "efecto Doppler", "resonancia",
            "interferencia", "difracción", "reflexión", "refracción",

            # Óptica
            "óptica", "luz", "espectro electromagnético", "fotón",
            "reflexión de la luz", "refracción de la luz", "ley de Snell",
            "lente convergente", "lente divergente", "espejo cóncavo", "espejo convexo",
            "prisma", "dispersión", "polarización", "láser",

            # Electricidad y Magnetismo
            "electricidad", "carga eléctrica", "coulomb", "campo eléctrico",
            "ley de Coulomb", "potencial eléctrico", "voltaje", "corriente eléctrica",
            "resistencia", "ley de Ohm", "circuito eléctrico", "circuito en serie", "circuito en paralelo",
            "capacitor", "inductor", "corriente alterna", "corriente continua",
            "magnetismo", "campo magnético", "fuerza de Lorentz", "ley de Ampère",
            "ley de Faraday", "inducción electromagnética", "motor eléctrico", "generador",
            "ecuaciones de Maxwell", "onda electromagnética",

            # Física Moderna
            "física moderna", "relatividad especial", "relatividad general",
            "dilatación del tiempo", "contracción de la longitud", "E=mc²",
            "mecánica cuántica", "principio de incertidumbre", "dualidad onda-partícula",
            "efecto fotoeléctrico", "modelo atómico de Bohr", "ecuación de Schrödinger",
            "spin", "principio de exclusión de Pauli", "tunelamiento cuántico",
            "física nuclear", "radiactividad", "fisión nuclear", "fusión nuclear",
            "partícula subatómica", "protón", "neutrón", "electrón", "quark", "bosón",
            "modelo estándar", "antimateria", "agujero negro", "Big Bang",
        ])

        # ===== QUÍMICA =====
        concepts.extend([
            # Química General
            "química", "átomo", "molécula", "elemento", "compuesto", "mezcla",
            "tabla periódica", "grupo", "período", "número atómico", "masa atómica",
            "electrón", "protón", "neutrón", "orbital", "configuración electrónica",
            "enlace químico", "enlace covalente", "enlace iónico", "enlace metálico",
            "electronegatividad", "valencia", "número de oxidación",
            "reacción química", "reactivo", "producto", "ecuación química", "balanceo",
            "mol", "número de Avogadro", "masa molar", "estequiometría",
            "solución", "soluto", "solvente", "concentración", "molaridad",
            "ácido", "base", "pH", "neutralización", "indicador",
            "oxidación", "reducción", "reacción redox", "electrólisis",

            # Química Orgánica
            "química orgánica", "hidrocarburo", "alcano", "alqueno", "alquino",
            "grupo funcional", "alcohol", "aldehído", "cetona", "ácido carboxílico",
            "éster", "amina", "amida", "polímero", "monómero",
            "isómero", "quiralidad", "estereoquímica",

            # Química Inorgánica
            "química inorgánica", "metal", "no metal", "metaloide",
            "óxido", "hidróxido", "sal", "ácido", "catión", "anión",

            # Bioquímica
            "bioquímica", "carbohidrato", "lípido", "proteína", "ácido nucleico",
            "aminoácido", "enzima", "ATP", "metabolismo", "catabolismo", "anabolismo",
            "ADN", "ARN", "nucleótido", "doble hélice", "replicación",
        ])

        # ===== BIOLOGÍA =====
        concepts.extend([
            # Biología Celular
            "biología", "célula", "célula procariota", "célula eucariota",
            "membrana celular", "citoplasma", "núcleo", "ribosoma", "mitocondria",
            "retículo endoplasmático", "aparato de Golgi", "lisosoma", "vacuola",
            "cloroplasto", "pared celular", "citoesqueleto",
            "mitosis", "meiosis", "ciclo celular", "cromosoma", "gen",

            # Genética
            "genética", "ADN", "ARN", "gen", "alelo", "genotipo", "fenotipo",
            "herencia", "dominante", "recesivo", "leyes de Mendel",
            "mutación", "recombinación genética", "ingeniería genética", "CRISPR",
            "genoma", "transcripción", "traducción", "código genético",

            # Evolución
            "evolución", "selección natural", "adaptación", "especiación",
            "Darwin", "teoría de la evolución", "ancestro común", "filogenia",
            "fósil", "registro fósil", "extinción", "biodiversidad",

            # Ecología
            "ecología", "ecosistema", "bioma", "hábitat", "nicho ecológico",
            "cadena alimenticia", "red trófica", "productor", "consumidor", "descomponedor",
            "población", "comunidad", "biosfera", "ciclo del carbono", "ciclo del nitrógeno",
            "fotosíntesis", "respiración celular", "simbiosis", "depredación", "competencia",

            # Anatomía y Fisiología
            "anatomía", "fisiología", "órgano", "tejido", "sistema",
            "sistema nervioso", "neurona", "cerebro", "médula espinal", "sinapsis",
            "sistema circulatorio", "corazón", "sangre", "arteria", "vena",
            "sistema respiratorio", "pulmón", "respiración", "oxígeno", "dióxido de carbono",
            "sistema digestivo", "estómago", "intestino", "hígado", "páncreas",
            "sistema inmunológico", "anticuerpo", "antígeno", "vacuna", "inmunidad",
            "sistema endocrino", "hormona", "glándula", "tiroides", "insulina",
            "sistema reproductor", "fecundación", "embarazo", "desarrollo embrionario",

            # Microbiología
            "microbiología", "bacteria", "virus", "hongo", "protista",
            "patógeno", "infección", "antibiótico", "resistencia antibiótica",
        ])

        # ===== LITERATURA =====
        concepts.extend([
            # Géneros Literarios
            "literatura", "género literario", "narrativa", "poesía", "drama", "ensayo",
            "novela", "cuento", "fábula", "leyenda", "mito", "epopeya",
            "poema", "verso", "estrofa", "rima", "métrica", "soneto", "haiku",
            "tragedia", "comedia", "tragicomedia", "teatro", "monólogo", "diálogo",

            # Elementos Narrativos
            "narrador", "punto de vista", "narrador omnisciente", "narrador protagonista",
            "personaje", "protagonista", "antagonista", "personaje secundario",
            "trama", "argumento", "conflicto", "clímax", "desenlace",
            "tiempo narrativo", "espacio narrativo", "flashback", "flashforward",
            "diálogo", "descripción", "monólogo interior", "flujo de conciencia",

            # Figuras Literarias
            "figura literaria", "metáfora", "símil", "personificación", "hipérbole",
            "alegoría", "ironía", "paradoja", "oxímoron", "antítesis",
            "aliteración", "onomatopeya", "anáfora", "epífora", "elipsis",
            "metonimia", "sinécdoque", "eufemismo", "litote",

            # Movimientos Literarios
            "movimiento literario", "clasicismo", "romanticismo", "realismo", "naturalismo",
            "modernismo", "vanguardismo", "surrealismo", "existencialismo",
            "boom latinoamericano", "realismo mágico", "posmodernismo",

            # Autores y Obras
            "Cervantes", "Don Quijote", "Shakespeare", "Hamlet", "García Márquez",
            "Cien años de soledad", "Borges", "Pablo Neruda", "Octavio Paz",
            "Gabriel García Márquez", "Mario Vargas Llosa", "Julio Cortázar",
            "Homer", "Ilíada", "Odisea", "Dante", "Divina Comedia",
        ])

        # ===== LENGUA Y LINGÜÍSTICA =====
        concepts.extend([
            # Gramática
            "gramática", "sintaxis", "morfología", "fonética", "fonología", "semántica",
            "oración", "sujeto", "predicado", "verbo", "sustantivo", "adjetivo",
            "adverbio", "pronombre", "preposición", "conjunción", "artículo",
            "complemento directo", "complemento indirecto", "complemento circunstancial",
            "oración simple", "oración compuesta", "oración subordinada", "oración coordinada",
            "voz activa", "voz pasiva", "modo indicativo", "modo subjuntivo", "modo imperativo",
            "tiempo verbal", "presente", "pasado", "futuro", "condicional",
            "género gramatical", "número gramatical", "concordancia",

            # Ortografía
            "ortografía", "acento", "tilde", "diptongo", "hiato", "triptongo",
            "mayúscula", "minúscula", "puntuación", "coma", "punto", "punto y coma",
            "dos puntos", "comillas", "paréntesis", "guion", "signos de interrogación",

            # Lingüística
            "lingüística", "lenguaje", "lengua", "habla", "idioma", "dialecto",
            "signo lingüístico", "significante", "significado", "Saussure",
            "fonema", "morfema", "lexema", "palabra", "oración",
            "pragmática", "discurso", "contexto", "acto de habla",
            "sociolingüística", "psicolingüística", "neurolingüística",
            "etimología", "préstamo lingüístico", "neologismo", "arcaísmo",
            "familia lingüística", "lenguas romances", "lenguas germánicas",

            # Comunicación
            "comunicación", "emisor", "receptor", "mensaje", "canal", "código",
            "comunicación verbal", "comunicación no verbal", "comunicación escrita",
            "retórica", "argumentación", "persuasión", "falacia",
        ])

        # ===== HISTORIA =====
        concepts.extend([
            # Prehistoria y Antigüedad
            "historia", "prehistoria", "Edad de Piedra", "Paleolítico", "Neolítico",
            "Edad del Bronce", "Edad del Hierro", "revolución neolítica",
            "Mesopotamia", "Egipto antiguo", "faraón", "pirámide", "jeroglífico",
            "Grecia antigua", "polis", "democracia ateniense", "Esparta", "Alejandro Magno",
            "Roma antigua", "República romana", "Imperio romano", "Julio César",
            "civilización", "imperio", "ciudad-estado", "monarquía",

            # Edad Media
            "Edad Media", "feudalismo", "señor feudal", "vasallo", "siervo",
            "castillo", "caballero", "Cruzadas", "Iglesia medieval",
            "Imperio bizantino", "Islam medieval", "Al-Ándalus",
            "Peste Negra", "Renacimiento", "humanismo",

            # Edad Moderna
            "Edad Moderna", "descubrimiento de América", "Cristóbal Colón",
            "conquista de América", "colonización", "Reforma protestante", "Lutero",
            "Contrarreforma", "absolutismo", "Luis XIV", "Ilustración",
            "Revolución Francesa", "Declaración de los Derechos del Hombre",
            "Napoleón", "independencia americana", "Simón Bolívar",

            # Edad Contemporánea
            "Edad Contemporánea", "Revolución Industrial", "capitalismo", "socialismo",
            "Primera Guerra Mundial", "Segunda Guerra Mundial", "Holocausto",
            "Guerra Fría", "comunismo", "Unión Soviética", "Estados Unidos",
            "descolonización", "globalización", "Unión Europea",
            "revolución tecnológica", "era digital", "internet",

            # Historia de América Latina
            "civilización maya", "civilización azteca", "civilización inca",
            "virreinato", "independencia latinoamericana", "caudillismo",
            "revolución mexicana", "revolución cubana",
        ])

        # ===== GEOGRAFÍA =====
        concepts.extend([
            # Geografía Física
            "geografía", "geografía física", "geografía humana", "cartografía",
            "continente", "océano", "mar", "río", "lago", "montaña", "volcán",
            "llanura", "meseta", "valle", "desierto", "selva", "tundra",
            "clima", "tiempo atmosférico", "temperatura", "precipitación", "humedad",
            "zona climática", "trópico", "ecuador", "polo", "hemisferio",
            "tectónica de placas", "terremoto", "tsunami", "erupción volcánica",
            "erosión", "sedimentación", "ciclo del agua", "cuenca hidrográfica",

            # Geografía Humana
            "población", "densidad de población", "migración", "urbanización",
            "ciudad", "metrópoli", "área rural", "área urbana",
            "país", "nación", "estado", "frontera", "territorio",
            "economía", "PIB", "desarrollo humano", "pobreza", "desigualdad",
            "recursos naturales", "energía renovable", "energía no renovable",
            "agricultura", "industria", "servicios", "comercio internacional",

            # Regiones del Mundo
            "América del Norte", "América Central", "América del Sur", "Europa",
            "Asia", "África", "Oceanía", "Antártida",
            "Medio Oriente", "Sudeste Asiático", "Europa Oriental",
        ])

        # ===== FILOSOFÍA Y LÓGICA =====
        concepts.extend([
            # Ramas de la Filosofía
            "filosofía", "metafísica", "epistemología", "ética", "estética", "lógica",
            "ontología", "filosofía de la mente", "filosofía política", "filosofía del lenguaje",

            # Conceptos Filosóficos
            "ser", "existencia", "realidad", "verdad", "conocimiento", "creencia",
            "razón", "experiencia", "percepción", "conciencia", "libre albedrío",
            "determinismo", "dualismo", "monismo", "materialismo", "idealismo",
            "empirismo", "racionalismo", "pragmatismo", "existencialismo",
            "bien", "mal", "virtud", "felicidad", "justicia", "libertad",
            "derechos humanos", "contrato social", "utilitarismo", "deontología",

            # Filósofos
            "Sócrates", "Platón", "Aristóteles", "Descartes", "Kant", "Hegel",
            "Nietzsche", "Marx", "Heidegger", "Wittgenstein", "Sartre",

            # Lógica
            "lógica formal", "lógica proposicional", "lógica de predicados",
            "argumento", "premisa", "conclusión", "validez", "verdad",
            "silogismo", "modus ponens", "modus tollens", "falacia",
            "falacia ad hominem", "falacia de autoridad", "falacia de generalización",
            "razonamiento deductivo", "razonamiento inductivo", "razonamiento abductivo",
            "paradoja", "paradoja del mentiroso", "paradoja de Zenón",
        ])

        # ===== PSICOLOGÍA =====
        concepts.extend([
            "psicología", "mente", "conducta", "cognición", "emoción", "motivación",
            "percepción", "atención", "memoria", "aprendizaje", "inteligencia",
            "personalidad", "temperamento", "carácter", "autoestima",
            "desarrollo cognitivo", "Piaget", "desarrollo moral", "Kohlberg",
            "psicoanálisis", "Freud", "inconsciente", "ego", "superego", "id",
            "conductismo", "condicionamiento clásico", "condicionamiento operante",
            "psicología cognitiva", "procesamiento de información",
            "psicología social", "conformidad", "obediencia", "prejuicio",
            "trastorno psicológico", "ansiedad", "depresión", "esquizofrenia",
            "terapia", "psicoterapia", "terapia cognitivo-conductual",
        ])

        # ===== ECONOMÍA =====
        concepts.extend([
            "economía", "microeconomía", "macroeconomía", "oferta", "demanda",
            "precio", "mercado", "competencia", "monopolio", "oligopolio",
            "PIB", "inflación", "deflación", "recesión", "crecimiento económico",
            "política fiscal", "política monetaria", "banco central", "tasa de interés",
            "comercio internacional", "exportación", "importación", "balanza comercial",
            "capitalismo", "socialismo", "economía mixta", "libre mercado",
            "Adam Smith", "Keynes", "Marx", "escasez", "costo de oportunidad",
        ])

        # ===== ARTE Y MÚSICA =====
        concepts.extend([
            # Arte
            "arte", "pintura", "escultura", "arquitectura", "fotografía",
            "dibujo", "color", "composición", "perspectiva", "luz y sombra",
            "arte rupestre", "arte clásico", "arte medieval", "renacimiento",
            "barroco", "impresionismo", "expresionismo", "cubismo", "surrealismo",
            "arte abstracto", "arte contemporáneo", "arte digital",
            "Leonardo da Vinci", "Miguel Ángel", "Picasso", "Van Gogh", "Dalí",

            # Música
            "música", "melodía", "armonía", "ritmo", "tempo", "compás",
            "nota musical", "escala", "acorde", "tonalidad", "clave",
            "instrumento musical", "voz", "orquesta", "banda", "coro",
            "música clásica", "jazz", "rock", "pop", "música electrónica",
            "ópera", "sinfonía", "concierto", "sonata", "fuga",
            "Bach", "Mozart", "Beethoven", "compositor", "director de orquesta",
        ])

        # ===== ARQUITECTURA DE SOFTWARE =====
        concepts.extend([
            "arquitectura de software", "patrón de diseño", "design pattern",
            "singleton", "factory", "abstract factory", "builder", "prototype",
            "adapter", "bridge", "composite", "decorator", "facade", "proxy",
            "observer", "strategy", "command", "state", "template method", "visitor",
            "MVC", "MVP", "MVVM", "clean architecture", "hexagonal architecture",
            "SOLID", "DRY", "KISS", "YAGNI", "separation of concerns",
            "dependency injection", "inversion of control", "service locator",
            "event-driven architecture", "CQRS", "event sourcing", "saga pattern",
            "domain-driven design", "bounded context", "aggregate", "entity", "value object",
        ])

        # ===== SISTEMAS OPERATIVOS =====
        concepts.extend([
            "sistema operativo", "kernel", "proceso", "hilo", "thread",
            "memoria virtual", "paginación", "segmentación", "cache", "buffer",
            "sistema de archivos", "inode", "file descriptor", "pipe", "socket",
            "scheduling", "context switch", "deadlock", "race condition", "mutex", "semáforo",
            "Linux", "Unix", "Windows", "macOS", "shell", "bash", "terminal",
            "chmod", "chown", "grep", "sed", "awk", "pipe", "redirección",
        ])

        # ===== REDES =====
        concepts.extend([
            "red", "protocolo", "TCP", "UDP", "IP", "IPv4", "IPv6",
            "DNS", "DHCP", "NAT", "subnet", "máscara de red", "gateway",
            "router", "switch", "hub", "bridge", "firewall",
            "modelo OSI", "capa física", "capa de enlace", "capa de red", "capa de transporte",
            "handshake", "ACK", "SYN", "FIN", "checksum",
            "latencia", "throughput", "bandwidth", "jitter", "packet loss",
            "VPN", "proxy", "túnel", "port forwarding",
        ])

        # ===== TESTING =====
        concepts.extend([
            "testing", "unit test", "integration test", "end-to-end test", "e2e",
            "test-driven development", "TDD", "behavior-driven development", "BDD",
            "mock", "stub", "spy", "fixture", "test double",
            "code coverage", "branch coverage", "line coverage",
            "pytest", "jest", "mocha", "junit", "cypress", "selenium",
            "assertion", "expect", "matcher", "test suite", "test case",
            "regression test", "smoke test", "sanity test", "load test", "stress test",
        ])

        # ===== CONCEPTOS GENERALES DE COMPUTACIÓN =====
        concepts.extend([
            "bit", "byte", "kilobyte", "megabyte", "gigabyte", "terabyte",
            "binario", "hexadecimal", "octal", "decimal", "ASCII", "Unicode", "UTF-8",
            "CPU", "GPU", "RAM", "ROM", "SSD", "HDD", "bus", "registro",
            "compilador", "intérprete", "JIT", "bytecode", "código máquina",
            "concurrencia", "paralelismo", "asíncrono", "síncrono", "blocking", "non-blocking",
            "API", "SDK", "framework", "librería", "biblioteca",
            "open source", "licencia MIT", "licencia GPL", "Apache License",
            "versión semántica", "changelog", "release", "deprecation",
        ])

        # ===== METODOLOGÍAS Y PRÁCTICAS =====
        concepts.extend([
            "Agile", "Scrum", "Kanban", "sprint", "daily standup", "retrospectiva",
            "product owner", "scrum master", "backlog", "user story", "epic",
            "pair programming", "code review", "refactoring", "technical debt",
            "documentación", "README", "API documentation", "swagger", "OpenAPI",
            "estimación", "planning poker", "velocity", "burndown chart",
        ])

        # ===== LENGUAJES DE PROGRAMACIÓN =====
        concepts.extend([
            "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#",
            "Go", "Rust", "Kotlin", "Swift", "Ruby", "PHP", "Perl",
            "Haskell", "Scala", "Clojure", "Elixir", "Erlang",
            "tipado estático", "tipado dinámico", "tipado fuerte", "tipado débil",
            "paradigma funcional", "paradigma orientado a objetos", "paradigma imperativo",
        ])

        # ===== CONCEPTOS AVANZADOS =====
        concepts.extend([
            "blockchain", "smart contract", "cryptocurrency", "consensus", "proof of work",
            "IoT", "edge computing", "fog computing", "embedded systems",
            "quantum computing", "qubit", "superposición", "entrelazamiento",
            "realidad virtual", "realidad aumentada", "metaverso",
            "big data", "data lake", "data warehouse", "ETL", "data pipeline",
            "streaming", "Apache Kafka", "Apache Spark", "Apache Flink",
        ])

        # ===== MCP (MODEL CONTEXT PROTOCOL) TOOLS =====
        concepts.extend([
            # MCP Fundamentals
            "MCP", "Model Context Protocol", "MCP server", "MCP client", "MCP tool",
            "MCP resource", "MCP prompt", "MCP capability", "MCP transport",
            "MCP stdio transport", "MCP SSE transport", "MCP WebSocket",
            "tool execution", "tool definition", "tool schema", "JSON Schema",
            "tool parameters", "tool result", "tool error handling",

            # MCP Architecture
            "MCP host", "MCP provider", "MCP registry", "MCP discovery",
            "MCP authentication", "MCP authorization", "MCP session",
            "MCP request", "MCP response", "MCP notification",
            "MCP initialize", "MCP shutdown", "MCP capabilities negotiation",

            # Common MCP Tools
            "file read tool", "file write tool", "file search tool",
            "bash execution tool", "command line tool", "shell tool",
            "web fetch tool", "HTTP request tool", "API call tool",
            "database query tool", "SQL execution tool",
            "code execution tool", "Python REPL tool",
            "browser automation tool", "screenshot tool",
            "git tool", "version control tool",

            # MCP Implementation
            "MCP SDK", "MCP Python SDK", "MCP TypeScript SDK",
            "@modelcontextprotocol/sdk", "mcp package",
            "FastMCP", "MCP server implementation", "MCP client implementation",
            "tool handler", "resource handler", "prompt handler",
            "MCP decorator", "@mcp.tool decorator",

            # Tool Calling Patterns
            "function calling", "tool calling", "tool use",
            "structured output", "JSON mode", "tool choice",
            "parallel tool calls", "sequential tool calls",
            "tool call chain", "agentic workflow",
            "ReAct pattern", "reasoning and acting",
            "tool selection", "tool routing",
        ])

        # ===== PROGRAMACIÓN AVANZADA Y PATRONES =====
        concepts.extend([
            # Design Patterns
            "singleton pattern", "factory pattern", "abstract factory",
            "builder pattern", "prototype pattern", "adapter pattern",
            "bridge pattern", "composite pattern", "decorator pattern",
            "facade pattern", "flyweight pattern", "proxy pattern",
            "chain of responsibility", "command pattern", "interpreter pattern",
            "iterator pattern", "mediator pattern", "memento pattern",
            "observer pattern", "state pattern", "strategy pattern",
            "template method", "visitor pattern", "dependency injection",

            # Architecture Patterns
            "clean architecture", "hexagonal architecture", "onion architecture",
            "layered architecture", "event-driven architecture", "CQRS",
            "event sourcing", "domain-driven design", "DDD",
            "bounded context", "aggregate", "entity", "value object",
            "repository pattern", "unit of work", "specification pattern",

            # Code Quality
            "SOLID principles", "single responsibility", "open/closed principle",
            "Liskov substitution", "interface segregation", "dependency inversion",
            "DRY principle", "KISS principle", "YAGNI principle",
            "code smell", "refactoring", "technical debt",
            "code review", "pair programming", "mob programming",
            "TDD", "test-driven development", "BDD", "behavior-driven development",
            "unit test", "integration test", "end-to-end test", "acceptance test",

            # API Design
            "REST API design", "RESTful principles", "HATEOAS",
            "API versioning", "API documentation", "OpenAPI", "Swagger",
            "rate limiting", "pagination", "filtering", "sorting",
            "API authentication", "API key", "Bearer token", "OAuth flow",
            "GraphQL schema", "GraphQL query", "GraphQL mutation", "GraphQL subscription",
            "gRPC", "Protocol Buffers", "service definition",

            # Concurrency
            "thread", "process", "coroutine", "async/await",
            "mutex", "semaphore", "lock", "deadlock", "race condition",
            "thread pool", "process pool", "event loop",
            "concurrent programming", "parallel programming",
            "actor model", "message passing", "shared memory",

            # Performance
            "profiling", "benchmarking", "optimization",
            "caching strategy", "memoization", "lazy loading",
            "connection pooling", "database optimization",
            "query optimization", "index optimization",
            "memory management", "garbage collection",
            "CPU bound", "I/O bound", "async I/O",
        ])

        # ===== LLM Y AGENTES =====
        concepts.extend([
            # LLM Concepts
            "large language model", "LLM", "foundation model",
            "tokenizer", "vocabulary", "subword tokenization", "BPE",
            "context window", "max tokens", "temperature", "top_p", "top_k",
            "prompt engineering", "system prompt", "user prompt", "assistant message",
            "few-shot learning", "zero-shot learning", "chain of thought",
            "instruction tuning", "RLHF", "DPO", "constitutional AI",

            # LLM APIs
            "OpenAI API", "Anthropic API", "Claude API",
            "chat completion", "text completion", "streaming response",
            "function calling", "tool use", "structured output",
            "embeddings API", "moderation API",

            # Agent Architecture
            "AI agent", "autonomous agent", "agent loop",
            "agent memory", "agent planning", "agent execution",
            "ReAct agent", "Plan and Execute agent",
            "agent tools", "agent capabilities",
            "multi-agent system", "agent collaboration",
            "agent orchestration", "agent routing",

            # RAG (Retrieval Augmented Generation)
            "RAG", "retrieval augmented generation",
            "document chunking", "chunk size", "chunk overlap",
            "embedding model", "vector store", "similarity search",
            "semantic search", "hybrid search", "reranking",
            "ChromaDB", "Pinecone", "Weaviate", "Milvus", "FAISS",

            # Frameworks
            "LangChain", "LlamaIndex", "AutoGPT", "BabyAGI",
            "CrewAI", "Autogen", "semantic kernel",
            "Ollama", "llama.cpp", "vLLM", "text-generation-inference",
        ])

        # ===== LENGUAJE C =====
        concepts.extend([
            # C Fundamentals
            "lenguaje C", "C programming", "C compiler", "gcc", "clang",
            "C standard", "C89", "C99", "C11", "C17", "C23",
            "main function", "stdio.h", "stdlib.h", "string.h", "math.h",
            "printf", "scanf", "fprintf", "fscanf", "sprintf",
            "fopen", "fclose", "fread", "fwrite", "fseek", "ftell",

            # C Data Types
            "C data types", "int", "char", "float", "double", "long", "short",
            "unsigned", "signed", "size_t", "ptrdiff_t", "void",
            "struct", "union", "enum", "typedef", "sizeof",
            "bit field", "padding", "alignment", "memory layout",

            # C Pointers & Memory
            "pointer", "pointer arithmetic", "null pointer", "void pointer",
            "pointer to pointer", "array pointer", "function pointer",
            "malloc", "calloc", "realloc", "free", "memory leak",
            "stack memory", "heap memory", "static memory", "memory allocation",
            "buffer overflow", "segmentation fault", "dangling pointer",
            "memcpy", "memset", "memmove", "memcmp",

            # C Arrays & Strings
            "C array", "multidimensional array", "array decay",
            "C string", "null terminator", "string literal",
            "strlen", "strcpy", "strcat", "strcmp", "strncpy", "strtok",

            # C Control Flow
            "goto statement", "setjmp", "longjmp", "signal handling",
            "SIGINT", "SIGTERM", "SIGSEGV", "signal handler",

            # C Preprocessor
            "preprocessor", "macro", "#define", "#include", "#ifdef", "#ifndef",
            "#pragma", "header guard", "include guard", "conditional compilation",
            "macro function", "variadic macro", "__FILE__", "__LINE__",

            # C Advanced
            "inline function", "static function", "extern", "volatile", "const",
            "restrict keyword", "register keyword", "static inline",
            "variadic function", "va_list", "va_start", "va_arg", "va_end",
            "function overloading", "generic selection", "_Generic",
            "bit manipulation", "bitwise operators", "bit mask",
            "endianness", "big endian", "little endian",

            # C File I/O
            "file handling in C", "binary file", "text file",
            "file modes", "rb", "wb", "ab", "r+", "w+", "a+",
            "buffered I/O", "unbuffered I/O", "setvbuf",

            # C Build System
            "make", "Makefile", "cmake", "CMakeLists.txt",
            "static library", "dynamic library", "shared object", ".so", ".a",
            "header file", ".h file", "source file", ".c file",
            "linking", "linker", "object file", ".o file",
            "compilation stages", "preprocessing", "compilation", "assembly",
        ])

        # ===== LENGUAJE C++ =====
        concepts.extend([
            # C++ Fundamentals
            "C++", "C++ programming", "g++", "C++ compiler",
            "C++11", "C++14", "C++17", "C++20", "C++23", "modern C++",
            "iostream", "cout", "cin", "cerr", "clog",
            "namespace", "using namespace std", "std namespace",

            # C++ OOP
            "C++ class", "C++ object", "member function", "member variable",
            "constructor", "destructor", "copy constructor", "move constructor",
            "assignment operator", "copy assignment", "move assignment",
            "rule of three", "rule of five", "rule of zero",
            "inheritance in C++", "multiple inheritance", "virtual inheritance",
            "virtual function", "pure virtual function", "abstract class",
            "override keyword", "final keyword", "virtual destructor",
            "polymorphism in C++", "dynamic dispatch", "vtable", "vptr",
            "friend function", "friend class",

            # C++ Access Control
            "public", "private", "protected",
            "encapsulation in C++", "getter", "setter",

            # C++ Templates
            "template", "function template", "class template",
            "template specialization", "partial specialization",
            "template parameter", "typename", "template metaprogramming",
            "SFINAE", "enable_if", "type traits", "concepts C++20",
            "variadic template", "parameter pack", "fold expression",

            # C++ STL
            "STL", "Standard Template Library",
            "vector", "deque", "list", "forward_list",
            "set", "multiset", "map", "multimap",
            "unordered_set", "unordered_map", "hash container",
            "stack STL", "queue STL", "priority_queue",
            "array STL", "pair", "tuple", "optional", "variant",
            "string C++", "string_view", "wstring",

            # C++ Iterators
            "iterator", "const_iterator", "reverse_iterator",
            "begin", "end", "cbegin", "cend", "rbegin", "rend",
            "iterator categories", "random access iterator", "bidirectional iterator",

            # C++ Algorithms
            "algorithm header", "sort", "find", "binary_search",
            "transform", "accumulate", "for_each", "copy", "move",
            "remove", "unique", "reverse", "rotate",
            "min_element", "max_element", "nth_element",
            "partition", "stable_partition", "merge",

            # C++ Smart Pointers
            "smart pointer", "unique_ptr", "shared_ptr", "weak_ptr",
            "make_unique", "make_shared", "custom deleter",
            "RAII", "Resource Acquisition Is Initialization",

            # C++ Modern Features
            "auto keyword", "decltype", "nullptr",
            "range-based for loop", "initializer list",
            "lambda expression", "lambda capture", "closure",
            "move semantics", "rvalue reference", "lvalue reference",
            "perfect forwarding", "std::forward", "std::move",
            "constexpr", "consteval", "constinit",
            "structured bindings", "if constexpr",
            "std::any", "std::variant", "std::optional",
            "coroutines C++", "co_await", "co_yield", "co_return",
            "modules C++", "import", "export module",

            # C++ Concurrency
            "thread C++", "std::thread", "std::async", "std::future",
            "mutex C++", "std::mutex", "std::lock_guard", "std::unique_lock",
            "condition_variable", "atomic", "std::atomic",
            "thread pool C++", "parallel algorithms",

            # C++ Exception Handling
            "exception handling C++", "try catch C++", "throw",
            "std::exception", "std::runtime_error", "std::logic_error",
            "noexcept", "exception safety", "strong guarantee",
        ])

        # ===== PYTHON AVANZADO =====
        concepts.extend([
            # Python Fundamentals
            "Python", "Python 3", "CPython", "PyPy", "Cython",
            "pip", "pip install", "requirements.txt", "setup.py", "pyproject.toml",
            "virtual environment", "venv", "virtualenv", "conda",
            "Python REPL", "IPython", "Jupyter", "Jupyter notebook",

            # Python Data Types
            "Python list", "Python dict", "Python set", "Python tuple",
            "list comprehension", "dict comprehension", "set comprehension",
            "generator expression", "slice notation", "unpacking",
            "f-string", "format string", "string methods",

            # Python OOP
            "Python class", "self keyword", "__init__", "__new__",
            "dunder methods", "magic methods", "__str__", "__repr__",
            "__eq__", "__hash__", "__lt__", "__gt__", "__add__",
            "__len__", "__iter__", "__next__", "__getitem__", "__setitem__",
            "__enter__", "__exit__", "context manager",
            "property decorator", "@property", "getter setter Python",
            "classmethod", "staticmethod", "@classmethod", "@staticmethod",
            "inheritance Python", "super()", "MRO", "method resolution order",
            "multiple inheritance Python", "mixin", "ABC", "abstract method",

            # Python Decorators
            "decorator", "function decorator", "class decorator",
            "decorator with arguments", "functools.wraps",
            "@dataclass", "@functools.lru_cache", "@functools.cached_property",

            # Python Generators
            "generator", "yield", "yield from", "generator function",
            "generator iterator", "send method", "throw method",
            "itertools", "chain", "cycle", "repeat", "islice",

            # Python Async
            "asyncio", "async", "await", "coroutine Python",
            "asyncio.run", "asyncio.gather", "asyncio.create_task",
            "event loop Python", "aiohttp", "httpx",
            "async context manager", "async iterator", "async generator",

            # Python Type Hints
            "type hints", "typing module", "Type", "Optional", "Union",
            "List", "Dict", "Set", "Tuple", "Callable", "Any",
            "TypeVar", "Generic", "Protocol", "Literal",
            "mypy", "pyright", "type checking",

            # Python Packages
            "numpy", "pandas", "matplotlib", "seaborn",
            "requests", "urllib", "beautifulsoup", "selenium",
            "flask", "django", "fastapi", "starlette",
            "sqlalchemy", "pydantic", "marshmallow",
            "pytest", "unittest", "mock", "fixtures",
            "logging Python", "argparse", "click",

            # Python Internals
            "GIL", "Global Interpreter Lock", "bytecode", "dis module",
            "garbage collection Python", "reference counting",
            "slots", "__slots__", "memory optimization",
            "metaclass", "type metaclass", "__class__",
            "descriptor protocol", "__get__", "__set__", "__delete__",
        ])

        # ===== JAVASCRIPT =====
        concepts.extend([
            # JavaScript Fundamentals
            "JavaScript", "ECMAScript", "ES6", "ES2015", "ES2020", "ES2022",
            "Node.js", "npm", "yarn", "pnpm", "package.json",
            "V8 engine", "SpiderMonkey", "JavaScriptCore",

            # JavaScript Data Types
            "JavaScript types", "undefined", "null", "boolean JS",
            "number JS", "bigint", "string JS", "symbol", "object JS",
            "array JS", "Array methods", "map", "filter", "reduce",
            "forEach", "find", "findIndex", "some", "every",
            "spread operator", "rest parameters", "destructuring",

            # JavaScript Functions
            "function JS", "arrow function", "function expression",
            "IIFE", "closure JS", "higher-order function JS",
            "callback JS", "callback hell", "promise",
            "async await JS", "Promise.all", "Promise.race",
            "generator function JS", "yield JS",

            # JavaScript OOP
            "JavaScript class", "constructor JS", "extends JS", "super JS",
            "prototype", "prototype chain", "__proto__", "Object.create",
            "this keyword", "bind", "call", "apply",
            "getter setter JS", "static method JS", "private field",

            # JavaScript DOM
            "DOM", "Document Object Model", "document", "window",
            "querySelector", "querySelectorAll", "getElementById",
            "createElement", "appendChild", "removeChild", "innerHTML",
            "addEventListener", "event listener", "event bubbling", "event capturing",
            "event delegation", "preventDefault", "stopPropagation",

            # JavaScript Modern
            "let", "const", "var", "hoisting", "temporal dead zone",
            "template literal", "tagged template", "optional chaining",
            "nullish coalescing", "logical assignment", "object shorthand",
            "computed property", "Map JS", "Set JS", "WeakMap", "WeakSet",
            "Proxy", "Reflect", "Symbol.iterator", "for of", "for in",

            # JavaScript Async
            "event loop JS", "call stack", "callback queue", "microtask queue",
            "setTimeout", "setInterval", "requestAnimationFrame",
            "fetch API", "XMLHttpRequest", "axios",
            "Web Workers", "Service Workers", "SharedArrayBuffer",

            # JavaScript Modules
            "ES modules", "import", "export", "default export", "named export",
            "CommonJS", "require", "module.exports",
            "dynamic import", "import()", "tree shaking",

            # TypeScript
            "TypeScript", "TS", "tsc", "tsconfig.json",
            "TypeScript types", "interface TS", "type alias",
            "union type", "intersection type", "literal type",
            "generic TS", "type inference", "type guard",
            "keyof", "typeof TS", "mapped types", "conditional types",
            "utility types", "Partial", "Required", "Pick", "Omit",
            "enum TS", "const assertion", "as const",
            "declaration file", ".d.ts", "DefinitelyTyped", "@types",
        ])

        # ===== JAVA =====
        concepts.extend([
            # Java Fundamentals
            "Java", "JDK", "JRE", "JVM", "Java bytecode",
            "javac", "java command", "jar file", "classpath",
            "Java 8", "Java 11", "Java 17", "Java 21", "LTS Java",
            "main method Java", "public static void main",

            # Java Data Types
            "primitive types Java", "int Java", "long Java", "double Java",
            "float Java", "char Java", "boolean Java", "byte", "short",
            "wrapper classes", "Integer", "Long", "Double", "Boolean",
            "autoboxing", "unboxing", "String Java", "StringBuilder", "StringBuffer",
            "array Java", "ArrayList", "LinkedList Java",

            # Java OOP
            "Java class", "Java object", "constructor Java", "method Java",
            "access modifiers", "public Java", "private Java", "protected Java",
            "inheritance Java", "extends", "implements", "super Java",
            "interface Java", "abstract class Java", "abstract method",
            "polymorphism Java", "method overriding", "method overloading",
            "@Override", "final class", "final method", "final variable",
            "static Java", "static method", "static variable", "static block",
            "inner class", "nested class", "anonymous class", "local class",

            # Java Collections
            "Java Collections Framework", "Collection interface", "List interface",
            "Set interface", "Map interface", "Queue interface",
            "ArrayList Java", "LinkedList Java", "HashSet", "TreeSet",
            "HashMap", "TreeMap", "LinkedHashMap", "Hashtable",
            "PriorityQueue Java", "Deque", "ArrayDeque",
            "Collections utility", "Arrays utility", "Comparator", "Comparable",
            "Iterator Java", "ListIterator", "enhanced for loop",

            # Java Generics
            "generics Java", "generic class", "generic method",
            "type parameter", "bounded type", "wildcard", "? extends", "? super",
            "type erasure", "raw type", "generic interface",

            # Java Exceptions
            "exception handling Java", "try catch Java", "finally Java",
            "throw Java", "throws Java", "checked exception", "unchecked exception",
            "RuntimeException", "Exception class", "Error class",
            "try-with-resources", "AutoCloseable", "custom exception",

            # Java Streams
            "Stream API", "stream Java", "IntStream", "LongStream",
            "stream operations", "intermediate operation", "terminal operation",
            "map stream", "filter stream", "reduce stream", "collect",
            "Collectors", "groupingBy", "partitioningBy", "joining",
            "parallel stream", "sequential stream",

            # Java Concurrency
            "multithreading Java", "Thread class", "Runnable interface",
            "Callable", "Future", "ExecutorService", "ThreadPoolExecutor",
            "synchronized", "volatile Java", "Lock interface", "ReentrantLock",
            "Semaphore Java", "CountDownLatch", "CyclicBarrier",
            "ConcurrentHashMap", "BlockingQueue", "CompletableFuture",
            "atomic classes", "AtomicInteger", "AtomicReference",

            # Java IO/NIO
            "Java IO", "InputStream", "OutputStream", "Reader", "Writer",
            "BufferedReader", "BufferedWriter", "FileInputStream", "FileOutputStream",
            "Java NIO", "ByteBuffer", "Channel", "Selector",
            "Path", "Files", "FileSystem",

            # Java Frameworks
            "Spring Framework", "Spring Boot", "Spring MVC",
            "dependency injection Spring", "@Autowired", "@Component",
            "@Service", "@Repository", "@Controller", "@RestController",
            "Spring Data JPA", "JPA", "Hibernate", "entity Java",
            "@Entity", "@Table", "@Column", "@Id", "@GeneratedValue",
            "Maven", "Gradle", "pom.xml", "build.gradle",
        ])

        # ===== PHP =====
        concepts.extend([
            # PHP Fundamentals
            "PHP", "PHP 8", "PHP 8.1", "PHP 8.2", "PHP 8.3",
            "php.ini", "PHP CLI", "PHP-FPM", "Composer",
            "composer.json", "vendor", "autoload", "PSR standards",

            # PHP Syntax
            "PHP variables", "$variable", "PHP echo", "PHP print",
            "PHP array", "associative array", "array functions",
            "array_map", "array_filter", "array_reduce", "array_merge",
            "PHP string functions", "explode", "implode", "str_replace",
            "PHP control structures", "foreach PHP", "match expression",

            # PHP OOP
            "PHP class", "PHP object", "__construct", "__destruct",
            "PHP visibility", "public PHP", "private PHP", "protected PHP",
            "PHP inheritance", "extends PHP", "parent::", "self::",
            "PHP interface", "implements PHP", "PHP trait", "use trait",
            "abstract class PHP", "final class PHP", "static PHP",
            "magic methods PHP", "__get", "__set", "__call", "__toString",

            # PHP 8 Features
            "named arguments", "attributes PHP", "#[Attribute]",
            "constructor promotion", "union types PHP", "nullable type",
            "match expression", "nullsafe operator", "?->",
            "enums PHP", "enum PHP 8.1", "readonly properties",
            "fibers PHP", "intersection types",

            # PHP Web
            "$_GET", "$_POST", "$_REQUEST", "$_SESSION", "$_COOKIE",
            "PHP sessions", "session_start", "PHP cookies",
            "header function", "redirect PHP", "HTTP response codes",
            "file upload PHP", "$_FILES", "move_uploaded_file",

            # PHP Database
            "PDO", "PHP Data Objects", "prepared statement",
            "mysqli", "MySQL PHP", "fetch modes", "PDO::FETCH_ASSOC",
            "database connection PHP", "query execution",

            # PHP Frameworks
            "Laravel", "Symfony", "CodeIgniter", "CakePHP",
            "Laravel Eloquent", "Laravel Blade", "Laravel migrations",
            "Laravel artisan", "Laravel routes", "Laravel controllers",
            "Symfony components", "Doctrine ORM",
            "MVC PHP", "routing PHP", "middleware PHP",

            # PHP Testing
            "PHPUnit", "testing PHP", "unit tests PHP",
            "assertions", "mock objects PHP", "test doubles",
            "Pest PHP", "code coverage PHP",
        ])

        # ===== C# Y .NET =====
        concepts.extend([
            # C# Fundamentals
            "C#", "C Sharp", ".NET", ".NET Core", ".NET 6", ".NET 7", ".NET 8",
            "CLR", "Common Language Runtime", "IL", "Intermediate Language",
            "dotnet CLI", "dotnet new", "dotnet run", "dotnet build",
            "NuGet", "NuGet packages", ".csproj", "solution file",

            # C# Data Types
            "C# value types", "C# reference types", "int C#", "string C#",
            "bool C#", "double C#", "decimal", "char C#",
            "nullable types C#", "Nullable<T>", "null C#", "??", "?..",
            "var C#", "dynamic C#", "object C#",
            "array C#", "List C#", "Dictionary C#",

            # C# OOP
            "C# class", "C# struct", "record C#", "record struct",
            "constructor C#", "destructor C#", "finalizer",
            "properties C#", "get set", "auto-property", "init accessor",
            "C# inheritance", "sealed class", "partial class",
            "interface C#", "abstract class C#", "virtual method",
            "override C#", "new modifier", "base keyword",
            "access modifiers C#", "public C#", "private C#", "internal", "protected internal",

            # C# Generics
            "generics C#", "generic class C#", "generic method C#",
            "type constraints", "where T", "new() constraint",
            "covariance", "contravariance", "in out keywords",

            # C# LINQ
            "LINQ", "Language Integrated Query",
            "LINQ to Objects", "LINQ to SQL", "LINQ to XML",
            "Where LINQ", "Select LINQ", "OrderBy", "GroupBy LINQ",
            "Join LINQ", "Aggregate", "Any", "All", "First", "Single",
            "query syntax", "method syntax", "deferred execution",
            "IEnumerable", "IQueryable", "AsQueryable",

            # C# Async
            "async await C#", "Task", "Task<T>", "async method",
            "await keyword", "Task.Run", "Task.WhenAll", "Task.WhenAny",
            "ConfigureAwait", "CancellationToken", "async streams",
            "ValueTask", "IAsyncEnumerable",

            # C# Modern Features
            "pattern matching C#", "switch expression", "is pattern",
            "tuple C#", "ValueTuple", "deconstruction C#",
            "local functions", "static local functions",
            "top-level statements", "global using", "file-scoped namespace",
            "init-only setters", "with expression", "positional records",
            "required members", "primary constructors",

            # ASP.NET
            "ASP.NET", "ASP.NET Core", "ASP.NET MVC", "Razor Pages",
            "Web API", "minimal API", "controllers", "actions",
            "model binding", "model validation", "data annotations",
            "routing ASP.NET", "attribute routing", "middleware ASP.NET",
            "dependency injection ASP.NET", "IServiceCollection",
            "Entity Framework", "EF Core", "DbContext", "DbSet",
            "migrations EF", "Code First", "Database First",

            # Blazor
            "Blazor", "Blazor Server", "Blazor WebAssembly", "Blazor components",
            "@code block", "EventCallback", "component parameters",
        ])

        # ===== REACT Y REACT NATIVE =====
        concepts.extend([
            # React Fundamentals
            "React", "React.js", "React 18", "create-react-app", "Vite React",
            "JSX", "JSX syntax", "JSX expressions", "JSX fragments",
            "React component", "functional component", "class component",
            "React element", "ReactDOM", "render method",

            # React Components
            "props", "props drilling", "children prop", "default props",
            "state", "useState", "setState", "state management",
            "component lifecycle", "mounting", "updating", "unmounting",
            "componentDidMount", "componentDidUpdate", "componentWillUnmount",

            # React Hooks
            "React Hooks", "useState hook", "useEffect hook", "useContext hook",
            "useReducer hook", "useCallback hook", "useMemo hook",
            "useRef hook", "useLayoutEffect", "useImperativeHandle",
            "custom hooks", "hook rules", "hooks dependencies",

            # React State Management
            "Context API", "React Context", "useContext", "Provider", "Consumer",
            "Redux", "Redux Toolkit", "createSlice", "configureStore",
            "useSelector", "useDispatch", "Redux middleware", "Redux Thunk",
            "Zustand", "Recoil", "Jotai", "MobX",

            # React Router
            "React Router", "react-router-dom", "BrowserRouter", "Routes", "Route",
            "Link", "NavLink", "useNavigate", "useParams", "useLocation",
            "nested routes", "route parameters", "query parameters",
            "protected routes", "route guards",

            # React Patterns
            "higher-order component", "HOC", "render props",
            "compound components", "controlled component", "uncontrolled component",
            "presentational component", "container component",
            "React.memo", "React.lazy", "Suspense", "code splitting React",

            # React Performance
            "React performance", "memoization React", "useMemo", "useCallback",
            "React.memo", "shouldComponentUpdate", "PureComponent",
            "virtual DOM", "reconciliation", "React Fiber",
            "React DevTools", "profiler", "React Strict Mode",

            # React Testing
            "React Testing Library", "Jest React", "@testing-library/react",
            "render", "screen", "fireEvent", "userEvent", "waitFor",
            "snapshot testing", "component testing",

            # React Native
            "React Native", "React Native CLI", "Expo", "expo-cli",
            "metro bundler", "react-native-cli",
            "View", "Text", "Image", "ScrollView", "FlatList",
            "TouchableOpacity", "Pressable", "StyleSheet",
            "React Native navigation", "React Navigation",
            "Stack Navigator", "Tab Navigator", "Drawer Navigator",
            "React Native APIs", "Linking", "AsyncStorage",
            "React Native modules", "native modules", "native bridge",
            "Platform API", "Platform.OS", "Platform.select",
            "React Native styling", "flexbox React Native",
            "React Native animations", "Animated API", "Reanimated",
        ])

        # ===== DART Y FLUTTER =====
        concepts.extend([
            # Dart Fundamentals
            "Dart", "Dart language", "Dart SDK", "dart command",
            "pub", "pubspec.yaml", "dart pub get", "dart analyze",
            "Dart 3", "Dart null safety", "sound null safety",

            # Dart Data Types
            "Dart types", "int Dart", "double Dart", "String Dart", "bool Dart",
            "List Dart", "Map Dart", "Set Dart", "Iterable Dart",
            "var Dart", "final Dart", "const Dart", "late Dart",
            "nullable Dart", "non-nullable", "?", "!", "??", "?..",
            "dynamic Dart", "Object Dart", "Type Dart",

            # Dart OOP
            "Dart class", "constructor Dart", "named constructor",
            "factory constructor", "const constructor", "initializer list",
            "Dart inheritance", "extends Dart", "implements Dart", "with Dart",
            "mixin Dart", "abstract class Dart", "interface Dart",
            "getter setter Dart", "get", "set", "static Dart",
            "extension methods", "extension Dart",

            # Dart Functions
            "Dart functions", "arrow function Dart", "=>",
            "optional parameters", "named parameters", "positional parameters",
            "default parameter values", "required keyword",
            "Function type", "typedef Dart", "closure Dart",

            # Dart Async
            "async Dart", "await Dart", "Future", "Future.then",
            "Future.catchError", "Future.whenComplete",
            "Stream Dart", "StreamController", "async*", "yield Dart",
            "StreamBuilder", "await for", "Stream.listen",
            "Completer", "Future.value", "Future.error",

            # Dart Collections
            "Dart collections", "List methods", "add", "addAll", "remove",
            "where", "map Dart", "reduce Dart", "fold", "any", "every",
            "spread operator Dart", "...", "collection if", "collection for",

            # Flutter Fundamentals
            "Flutter", "Flutter SDK", "flutter create", "flutter run",
            "flutter doctor", "flutter build", "flutter pub get",
            "Flutter widget", "StatelessWidget", "StatefulWidget",
            "build method", "BuildContext", "State class",
            "widget tree", "element tree", "render tree",

            # Flutter Widgets
            "Container", "Row", "Column", "Stack", "Positioned",
            "Padding", "Center", "Align", "SizedBox", "Expanded", "Flexible",
            "ListView", "GridView", "SingleChildScrollView",
            "Scaffold", "AppBar", "BottomNavigationBar", "Drawer",
            "Text widget", "Image widget", "Icon widget", "Button widgets",
            "ElevatedButton", "TextButton", "IconButton", "FloatingActionButton",
            "TextField", "TextFormField", "Form widget", "FormField",
            "Card", "ListTile", "CircleAvatar", "Chip",

            # Flutter Layout
            "Flutter layout", "Flex", "mainAxisAlignment", "crossAxisAlignment",
            "MainAxisSize", "BoxConstraints", "LayoutBuilder",
            "MediaQuery", "responsive Flutter", "ConstrainedBox",

            # Flutter Navigation
            "Navigator", "Navigator.push", "Navigator.pop",
            "MaterialPageRoute", "named routes", "onGenerateRoute",
            "GoRouter", "auto_route", "Navigator 2.0",

            # Flutter State Management
            "setState", "InheritedWidget", "Provider Flutter",
            "ChangeNotifier", "ChangeNotifierProvider", "Consumer",
            "Riverpod", "StateNotifier", "StateNotifierProvider",
            "BLoC pattern", "flutter_bloc", "Cubit", "BlocProvider",
            "GetX", "GetxController", "Obx",

            # Flutter Packages
            "flutter packages", "http package", "dio", "shared_preferences",
            "sqflite", "hive", "firebase_core", "cloud_firestore",
            "get_it", "injectable", "freezed", "json_serializable",
        ])

        return concepts

    def _build_categories(self) -> Dict[str, List[str]]:
        """Organiza conceptos por categorías para exploración temática"""
        return {
            # === TECNOLOGÍA ===
            "programacion_basica": [
                "variable", "función", "loop", "condicional", "array", "objeto", "clase",
                "herencia", "polimorfismo", "recursión", "excepción"
            ],
            "estructuras_datos": [
                "lista", "diccionario", "árbol binario", "grafo", "heap", "hash table",
                "pila", "cola", "lista enlazada"
            ],
            "algoritmos": [
                "bubble sort", "quick sort", "binary search", "dijkstra", "programación dinámica",
                "Big O", "backtracking", "divide y vencerás"
            ],
            "bases_datos": [
                "SQL", "NoSQL", "índice", "transacción", "normalización", "ORM",
                "JOIN", "clave primaria", "ACID"
            ],
            "web": [
                "HTML", "CSS", "JavaScript", "REST", "API", "React", "Node.js",
                "HTTP", "JSON", "frontend", "backend"
            ],
            "cloud_devops": [
                "Docker", "Kubernetes", "AWS", "CI/CD", "Git", "Terraform",
                "microservicio", "serverless", "contenedor"
            ],
            "seguridad": [
                "autenticación", "JWT", "encriptación", "OWASP", "firewall",
                "SSL", "hash", "OAuth", "cifrado"
            ],
            "ia_ml": [
                "machine learning", "neural network", "deep learning", "NLP", "PyTorch",
                "CNN", "transformer", "GPT", "backpropagation"
            ],

            # === MCP Y AGENTES ===
            "mcp_tools": [
                "MCP", "Model Context Protocol", "MCP server", "MCP client", "MCP tool",
                "tool execution", "tool definition", "tool schema", "JSON Schema",
                "MCP transport", "stdio transport", "tool handler", "FastMCP",
                "function calling", "tool use", "tool calling"
            ],
            "llm_agentes": [
                "LLM", "large language model", "prompt engineering", "context window",
                "AI agent", "autonomous agent", "ReAct pattern", "agent loop",
                "RAG", "embeddings", "vector store", "semantic search",
                "LangChain", "Ollama", "fine-tuning"
            ],
            "patrones_diseno": [
                "singleton pattern", "factory pattern", "observer pattern", "strategy pattern",
                "decorator pattern", "adapter pattern", "dependency injection",
                "repository pattern", "SOLID principles", "clean architecture"
            ],
            "programacion_avanzada": [
                "async/await", "coroutine", "thread", "concurrency", "parallelism",
                "mutex", "semaphore", "event loop", "callback", "promise",
                "profiling", "optimization", "caching", "lazy loading"
            ],
            "api_design": [
                "REST API", "GraphQL", "gRPC", "OpenAPI", "API versioning",
                "rate limiting", "pagination", "authentication", "Bearer token"
            ],

            # === MATEMÁTICAS POR NIVEL ===
            # BÁSICA - Aritmética y conceptos fundamentales
            "matematicas_basica": [
                "suma", "resta", "multiplicación", "división", "fracción", "decimal",
                "porcentaje", "número natural", "número entero", "número par", "número impar",
                "múltiplo", "divisor", "máximo común divisor", "mínimo común múltiplo",
                "potencia", "raíz cuadrada", "orden de operaciones", "redondeo",
                "número primo", "número compuesto", "factorización", "proporción", "razón",
                "regla de tres", "conversión de unidades", "sistema métrico",
                "operaciones con fracciones", "fracciones equivalentes", "simplificar fracciones",
                "suma de fracciones", "resta de fracciones", "multiplicación de fracciones",
                "división de fracciones", "fracción mixta", "número decimal periódico"
            ],
            # MEDIA - Álgebra y Geometría básica
            "matematicas_media": [
                "álgebra", "ecuación lineal", "variable", "constante", "coeficiente",
                "despejar variable", "sistema de ecuaciones", "método de sustitución",
                "método de eliminación", "inecuación", "valor absoluto",
                "polinomio", "monomio", "binomio", "suma de polinomios", "resta de polinomios",
                "multiplicación de polinomios", "factor común", "productos notables",
                "ecuación cuadrática", "fórmula cuadrática", "discriminante", "raíces",
                "geometría plana", "triángulo", "cuadrilátero", "polígono", "círculo",
                "perímetro", "área", "teorema de Pitágoras", "ángulo", "ángulo recto",
                "ángulos complementarios", "ángulos suplementarios", "triángulos semejantes",
                "funciones", "dominio", "rango", "función lineal", "pendiente", "ordenada al origen",
                "gráfica de función", "función cuadrática", "parábola", "vértice"
            ],
            # AVANZADA - Cálculo, Trigonometría, Álgebra Lineal
            "matematicas_avanzada": [
                "trigonometría", "seno", "coseno", "tangente", "identidades trigonométricas",
                "ley de senos", "ley de cosenos", "radianes", "círculo unitario",
                "funciones trigonométricas inversas", "ecuaciones trigonométricas",
                "logaritmo", "logaritmo natural", "propiedades de logaritmos", "ecuación exponencial",
                "ecuación logarítmica", "función exponencial", "número e", "crecimiento exponencial",
                "sucesión", "serie", "progresión aritmética", "progresión geométrica",
                "límite", "continuidad", "derivada", "regla de la cadena", "regla del producto",
                "regla del cociente", "derivadas de funciones trigonométricas",
                "aplicaciones de la derivada", "máximos y mínimos", "optimización",
                "integral", "integral definida", "integral indefinida", "antiderivada",
                "integración por sustitución", "integración por partes",
                "matriz", "vector", "operaciones con matrices", "determinante", "matriz inversa",
                "sistemas de ecuaciones lineales", "método de Gauss", "espacio vectorial",
                "números complejos", "forma polar", "operaciones con complejos"
            ],
            # EXPERTA - Matemáticas universitarias y teoría
            "matematicas_experta": [
                "cálculo multivariable", "derivada parcial", "gradiente", "hessiano", "jacobiano",
                "integral doble", "integral triple", "coordenadas polares", "coordenadas cilíndricas",
                "coordenadas esféricas", "teorema de Green", "teorema de Stokes", "teorema de la divergencia",
                "ecuación diferencial ordinaria", "ecuación diferencial parcial", "EDO de primer orden",
                "EDO de segundo orden", "ecuación homogénea", "ecuación no homogénea",
                "serie de Taylor", "serie de Maclaurin", "serie de Fourier", "transformada de Laplace",
                "transformada de Fourier", "convergencia de series", "radio de convergencia",
                "álgebra lineal avanzada", "autovalor", "autovector", "diagonalización",
                "descomposición SVD", "espacios de Hilbert", "transformación lineal",
                "teoría de grupos", "anillo", "campo", "homomorfismo", "isomorfismo",
                "topología", "espacio métrico", "espacio normado", "completitud",
                "análisis real", "medida de Lebesgue", "teorema de la convergencia dominada",
                "probabilidad avanzada", "variable aleatoria", "función de distribución",
                "esperanza matemática", "varianza", "covarianza", "teorema del límite central",
                "distribución normal", "distribución de Poisson", "distribución binomial",
                "cadenas de Markov", "proceso estocástico", "teorema de Bayes avanzado",
                "optimización convexa", "programación lineal", "método simplex",
                "teoría de números", "congruencia", "aritmética modular", "teorema de Fermat",
                "función phi de Euler", "criptografía matemática", "RSA"
            ],
            # Categorías originales mantenidas para compatibilidad
            "matematicas_basicas": [
                "suma", "resta", "multiplicación", "división", "fracción", "porcentaje",
                "número primo", "factorial", "potencia", "raíz cuadrada"
            ],
            "algebra": [
                "ecuación", "polinomio", "función", "variable", "sistema de ecuaciones",
                "logaritmo", "ecuación cuadrática", "matriz", "vector"
            ],
            "geometria": [
                "triángulo", "círculo", "área", "perímetro", "volumen",
                "teorema de Pitágoras", "ángulo", "coordenadas cartesianas"
            ],
            "calculo": [
                "derivada", "integral", "límite", "serie de Taylor",
                "ecuación diferencial", "gradiente"
            ],
            "estadistica": [
                "probabilidad", "media", "varianza", "distribución normal",
                "correlación", "regresión", "hipótesis"
            ],

            # === FÍSICA ===
            "fisica_mecanica": [
                "fuerza", "masa", "velocidad", "aceleración", "leyes de Newton",
                "energía cinética", "energía potencial", "momento"
            ],
            "fisica_electromagnetismo": [
                "electricidad", "magnetismo", "corriente eléctrica", "voltaje",
                "ley de Ohm", "campo magnético", "ley de Faraday"
            ],
            "fisica_moderna": [
                "relatividad", "mecánica cuántica", "E=mc²", "efecto fotoeléctrico",
                "principio de incertidumbre", "Big Bang"
            ],

            # === QUÍMICA ===
            "quimica": [
                "átomo", "molécula", "elemento", "tabla periódica", "enlace químico",
                "reacción química", "ácido", "base", "pH", "mol"
            ],

            # === BIOLOGÍA ===
            "biologia": [
                "célula", "ADN", "gen", "evolución", "fotosíntesis",
                "mitosis", "ecosistema", "bacteria", "virus"
            ],
            "anatomia": [
                "corazón", "cerebro", "pulmón", "sistema nervioso",
                "sistema circulatorio", "neurona", "hormona"
            ],

            # === HUMANIDADES ===
            "literatura": [
                "novela", "poesía", "metáfora", "narrador", "protagonista",
                "romanticismo", "realismo mágico", "Shakespeare", "Cervantes"
            ],
            "lengua": [
                "gramática", "sintaxis", "verbo", "sustantivo", "oración",
                "ortografía", "semántica", "morfología"
            ],
            "historia": [
                "Edad Media", "Revolución Industrial", "Segunda Guerra Mundial",
                "Roma antigua", "Grecia antigua", "Renacimiento"
            ],
            "geografia": [
                "continente", "océano", "clima", "población", "río",
                "montaña", "tectónica de placas", "ecosistema"
            ],
            "filosofia": [
                "ética", "lógica", "metafísica", "Sócrates", "Platón",
                "verdad", "conocimiento", "existencialismo"
            ],
            "logica": [
                "silogismo", "premisa", "conclusión", "falacia",
                "razonamiento deductivo", "razonamiento inductivo", "paradoja"
            ],

            # === CIENCIAS SOCIALES ===
            "psicologia": [
                "mente", "conducta", "memoria", "aprendizaje", "Freud",
                "cognición", "emoción", "personalidad"
            ],
            "economia": [
                "oferta", "demanda", "PIB", "inflación", "mercado",
                "capitalismo", "banco central", "comercio"
            ],

            # === ARTE ===
            "arte": [
                "pintura", "escultura", "Picasso", "Leonardo da Vinci",
                "impresionismo", "renacimiento", "color", "composición"
            ],
            "musica": [
                "melodía", "armonía", "ritmo", "Bach", "Mozart",
                "orquesta", "sinfonía", "nota musical"
            ],

            # === LENGUAJES DE PROGRAMACIÓN ===
            "c_language": [
                "lenguaje C", "C programming", "gcc", "clang", "C99", "C11",
                "pointer", "puntero", "malloc", "free", "memory allocation",
                "array in C", "struct", "union", "typedef", "enum",
                "preprocessor", "macro", "#define", "#include", "header file",
                "makefile", "cmake", "linker", "static library", "dynamic library",
                "segmentation fault", "buffer overflow", "memory leak"
            ],
            "cpp_language": [
                "C++", "C++ programming", "g++", "C++11", "C++14", "C++17", "C++20",
                "class C++", "constructor", "destructor", "RAII",
                "template", "template metaprogramming", "STL", "vector", "map", "set",
                "smart pointer", "unique_ptr", "shared_ptr", "weak_ptr",
                "virtual function", "polymorphism C++", "inheritance C++", "abstract class",
                "namespace", "exception handling C++", "operator overloading",
                "move semantics", "rvalue reference", "lambda C++", "auto keyword",
                "constexpr", "static_assert", "range-based for", "structured binding"
            ],
            "python_advanced": [
                "Python", "Python programming", "CPython", "PyPy", "pip", "virtualenv",
                "class Python", "__init__", "__str__", "__repr__", "dunder method",
                "decorator", "@property", "@staticmethod", "@classmethod",
                "generator", "yield", "generator expression", "iterator protocol",
                "async Python", "asyncio", "await", "coroutine Python",
                "type hints", "typing module", "Union", "Optional", "TypeVar",
                "metaclass", "__new__", "__call__", "descriptor",
                "context manager", "__enter__", "__exit__", "with statement",
                "GIL", "multiprocessing", "threading Python", "concurrent.futures"
            ],
            "javascript": [
                "JavaScript", "ES6", "ES2020", "ES2021", "ECMAScript",
                "let", "const", "var", "hoisting", "scope",
                "arrow function", "template literal", "destructuring", "spread operator",
                "Promise", "async/await JavaScript", "callback", "fetch API",
                "DOM", "document", "querySelector", "addEventListener",
                "object JavaScript", "prototype", "class JavaScript", "extends",
                "module JavaScript", "import", "export", "CommonJS", "ES modules",
                "Node.js", "npm", "package.json", "node_modules"
            ],
            "typescript": [
                "TypeScript", "tsc", "tsconfig", "type annotation",
                "interface TypeScript", "type alias", "union type", "intersection type",
                "generic TypeScript", "type inference", "type guard", "discriminated union",
                "enum TypeScript", "tuple", "readonly", "partial", "required",
                "utility types", "mapped types", "conditional types", "infer keyword"
            ],
            "java": [
                "Java", "JVM", "JDK", "JRE", "javac", "bytecode",
                "class Java", "interface Java", "abstract class Java", "enum Java",
                "inheritance Java", "polymorphism Java", "encapsulation", "abstraction",
                "Collection", "List Java", "ArrayList", "HashMap", "HashSet", "TreeMap",
                "generics Java", "wildcard", "bounded type", "type erasure",
                "exception Java", "try-catch", "finally", "throws", "checked exception",
                "Stream API", "lambda Java", "functional interface", "method reference",
                "thread Java", "synchronized", "volatile", "ExecutorService", "CompletableFuture",
                "Spring Boot", "Spring Framework", "dependency injection Java", "annotation"
            ],
            "php": [
                "PHP", "PHP 8", "composer", "autoload", "PSR",
                "class PHP", "interface PHP", "trait", "abstract class PHP",
                "namespace PHP", "use statement", "constructor promotion",
                "type declaration", "union types PHP", "named arguments", "attributes",
                "array PHP", "associative array", "array functions",
                "PDO", "mysqli", "prepared statement", "SQL injection prevention",
                "session", "cookie", "superglobals", "$_GET", "$_POST", "$_SESSION",
                "Laravel", "Symfony", "Blade template", "Eloquent ORM", "Artisan"
            ],
            "csharp_dotnet": [
                "C#", "C sharp", ".NET", ".NET Core", "ASP.NET", "dotnet CLI",
                "class C#", "struct C#", "interface C#", "record",
                "property", "auto-property", "indexer", "delegate", "event",
                "LINQ", "lambda C#", "expression tree", "IEnumerable", "IQueryable",
                "async C#", "Task", "async/await C#", "ValueTask",
                "nullable reference", "null-coalescing", "pattern matching C#", "switch expression",
                "ASP.NET Core", "middleware", "dependency injection C#", "Entity Framework",
                "Blazor", "Razor Pages", "SignalR", "gRPC .NET"
            ],
            "react": [
                "React", "React.js", "JSX", "component", "props", "state",
                "useState", "useEffect", "useContext", "useReducer", "useMemo", "useCallback",
                "custom hook", "hook rules", "React lifecycle",
                "functional component", "class component", "render", "virtual DOM",
                "Redux", "Redux Toolkit", "useSelector", "useDispatch", "reducer",
                "React Router", "BrowserRouter", "Route", "Link", "useNavigate",
                "Context API", "Provider", "Consumer", "useContext",
                "React memo", "React.lazy", "Suspense", "code splitting"
            ],
            "react_native": [
                "React Native", "mobile development", "cross-platform",
                "View", "Text", "ScrollView", "FlatList", "TouchableOpacity",
                "StyleSheet", "Flexbox React Native", "responsive design mobile",
                "React Navigation", "Stack Navigator", "Tab Navigator", "Drawer Navigator",
                "AsyncStorage", "Expo", "react-native-cli", "Metro bundler",
                "native module", "bridge", "Hermes engine", "debugging React Native"
            ],
            "dart_flutter": [
                "Dart", "Dart programming", "dart pub", "pubspec.yaml",
                "class Dart", "mixin", "extension", "late", "final Dart", "const Dart",
                "Future", "async Dart", "await Dart", "Stream", "StreamBuilder",
                "null safety", "required keyword", "named parameters", "optional parameters",
                "Flutter", "widget", "StatelessWidget", "StatefulWidget", "setState",
                "BuildContext", "MaterialApp", "Scaffold", "AppBar",
                "Container", "Row", "Column", "Stack", "ListView", "GridView",
                "Navigator", "routes", "push", "pop", "named routes",
                "Provider Flutter", "Riverpod", "BLoC pattern", "Cubit", "GetX",
                "pubspec dependencies", "flutter pub get", "hot reload", "hot restart"
            ],
        }

    def _load_activity_log(self) -> Dict:
        """Carga registro de actividad"""
        log_file = self.data_dir / "activity_log.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)

        return {
            "questions_asked": [],
            "total_questions": 0,
            "last_question_time": None,
            "daily_count": {},
            "hourly_count": {}
        }

    def _save_activity_log(self):
        """Guarda registro de actividad"""
        log_file = self.data_dir / "activity_log.json"

        with open(log_file, 'w') as f:
            json.dump(self.activity_log, f, indent=2)

    def _can_ask_question(self) -> tuple[bool, str]:
        """
        Verifica si puede hacer más preguntas según límites de seguridad

        Returns:
            (puede_preguntar, razón_si_no)
        """
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        current_day = now.strftime("%Y-%m-%d")

        # Verificar límite por hora
        hourly_count = self.activity_log.get("hourly_count", {}).get(current_hour, 0)
        if hourly_count >= self.max_questions_per_hour:
            return False, f"Límite por hora alcanzado ({self.max_questions_per_hour} preguntas/hora)"

        # Verificar límite por día
        daily_count = self.activity_log.get("daily_count", {}).get(current_day, 0)
        if daily_count >= self.max_questions_per_day:
            return False, f"Límite diario alcanzado ({self.max_questions_per_day} preguntas/día)"

        # Verificar tiempo mínimo entre preguntas
        last_time = self.activity_log.get("last_question_time")
        if last_time:
            last_dt = datetime.fromisoformat(last_time)
            elapsed = (now - last_dt).total_seconds()
            if elapsed < self.min_seconds_between_questions:
                return False, f"Debe esperar {self.min_seconds_between_questions - elapsed:.0f}s más"

        return True, "OK"

    def _record_question(self, question: str, answer: str):
        """Registra una pregunta realizada"""
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        current_day = now.strftime("%Y-%m-%d")

        # Actualizar contadores
        if "hourly_count" not in self.activity_log:
            self.activity_log["hourly_count"] = {}
        if "daily_count" not in self.activity_log:
            self.activity_log["daily_count"] = {}

        self.activity_log["hourly_count"][current_hour] = \
            self.activity_log["hourly_count"].get(current_hour, 0) + 1
        self.activity_log["daily_count"][current_day] = \
            self.activity_log["daily_count"].get(current_day, 0) + 1

        # Registrar pregunta
        self.activity_log["questions_asked"].append({
            "timestamp": now.isoformat(),
            "question": question,
            "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer
        })

        self.activity_log["total_questions"] += 1
        self.activity_log["last_question_time"] = now.isoformat()

        # Limpiar contadores antiguos (más de 24 horas)
        self._cleanup_old_counters()

        self._save_activity_log()

    def _cleanup_old_counters(self):
        """Limpia contadores de hace más de 24 horas"""
        now = datetime.now()
        cutoff_hour = (now - timedelta(hours=24)).strftime("%Y-%m-%d-%H")
        cutoff_day = (now - timedelta(days=7)).strftime("%Y-%m-%d")

        # Limpiar horas antiguas
        old_hours = [h for h in self.activity_log.get("hourly_count", {}).keys() if h < cutoff_hour]
        for hour in old_hours:
            del self.activity_log["hourly_count"][hour]

        # Limpiar días antiguos
        old_days = [d for d in self.activity_log.get("daily_count", {}).keys() if d < cutoff_day]
        for day in old_days:
            del self.activity_log["daily_count"][day]

    def generate_question(
        self,
        cognitive_age: int = 0,
        topic: Optional[str] = None,
        category: Optional[str] = None
    ) -> Optional[str]:
        """
        Genera una pregunta apropiada para la edad cognitiva actual

        Args:
            cognitive_age: Edad cognitiva de THAU (0-15)
            topic: Tema específico (opcional)
            category: Categoría temática (opcional)

        Returns:
            Pregunta generada o None si no se puede generar
        """
        # Mapear edad a bracket de plantillas
        age_brackets = [0, 1, 2, 3, 6, 11, 15]
        age_bracket = 0
        for bracket in age_brackets:
            if cognitive_age >= bracket:
                age_bracket = bracket

        templates = self.topic_templates.get(age_bracket, self.topic_templates[0])

        # Seleccionar concepto
        if topic:
            selected_topic = topic
        elif category and category in self.knowledge_categories:
            selected_topic = random.choice(self.knowledge_categories[category])
        else:
            selected_topic = random.choice(self.exploration_concepts)

        # Generar pregunta usando plantilla
        template = random.choice(templates)

        # Si la plantilla necesita dos conceptos
        if "{concept1}" in template and "{concept2}" in template:
            concept1 = selected_topic
            # Elegir segundo concepto relacionado o aleatorio
            if category and category in self.knowledge_categories:
                available = [c for c in self.knowledge_categories[category] if c != concept1]
                if available:
                    concept2 = random.choice(available)
                else:
                    concept2 = random.choice([c for c in self.exploration_concepts if c != concept1])
            else:
                concept2 = random.choice([c for c in self.exploration_concepts if c != concept1])
            question = template.format(concept1=concept1, concept2=concept2)
        else:
            question = template.format(concept=selected_topic)

        return question

    def answer_question(
        self,
        question: str,
        timeout: int = 60,
        use_model: str = "auto"
    ) -> Optional[str]:
        """
        Responde una pregunta usando múltiples fuentes de conocimiento

        Args:
            question: Pregunta a responder
            timeout: Tiempo máximo de espera en segundos
            use_model: "ollama", "deepseek", "llama", "mistral", "phi3", "gpt_oss", "auto", o "random"

        Returns:
            Respuesta generada o None si falla
        """
        # Mapeo de modelos disponibles
        model_methods = {
            "ollama": self._ask_ollama,
            "deepseek": self._ask_deepseek,
            "llama": self._ask_llama,
            "mistral": self._ask_mistral,
            "phi3": self._ask_phi3,
            "gpt_oss": self._ask_gpt_oss,
            "coder": self._ask_coder,
            "gemini": self._ask_gemini,
        }

        if use_model == "auto":
            # Orden de preferencia: empezar con el más rápido
            models_to_try = ["coder", "phi3", "deepseek", "llama", "mistral"]
        elif use_model == "random":
            # Elegir aleatoriamente para diversidad de respuestas
            models_to_try = random.sample(list(model_methods.keys()), len(model_methods))
        elif use_model == "best":
            # Usar los modelos más capaces primero
            models_to_try = ["gpt_oss", "deepseek", "llama", "mistral", "phi3", "ollama"]
        else:
            models_to_try = [use_model] if use_model in model_methods else ["ollama"]

        for model_type in models_to_try:
            try:
                method = model_methods.get(model_type)
                if method:
                    answer = method(question, timeout)
                    if answer:
                        return answer

            except Exception as e:
                print(f"⚠️  Error con {model_type}: {e}")
                continue

        return None

    def _ask_ollama(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a Ollama local"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"Responde de forma clara y concisa en español:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            print(f"❌ Ollama error: {e}")

        return None

    def _ask_deepseek(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a DeepSeek R1 - excelente para razonamiento"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "deepseek-r1:8b",
                    "prompt": f"Responde de forma clara y detallada en español:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 600
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            pass

        return None

    def _ask_llama(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a Llama 3.1 8B"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": f"Responde de forma clara y detallada en español:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 600
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            pass

        return None

    def _ask_mistral(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a Mistral"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "mistral:latest",
                    "prompt": f"Responde de forma técnica y precisa en español:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "num_predict": 600
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            pass

        return None

    def _ask_phi3(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a Phi-3 - modelo pequeño pero capaz"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "phi3:latest",
                    "prompt": f"Responde de forma clara en español:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            pass

        return None

    def _ask_gpt_oss(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a GPT-OSS 20B - modelo grande y capaz"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "gpt-oss:20b",
                    "prompt": f"Responde de forma experta y detallada en español:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 800
                    }
                },
                timeout=timeout * 2  # Más tiempo para modelo grande
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            pass

        return None

    def _ask_coder(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a Qwen2.5-Coder - especializado en código y programación"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": f"Responde de forma técnica y detallada en español. Si la pregunta es sobre programación, incluye ejemplos de código cuando sea relevante:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 600
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            pass

        return None

    def _ask_gemini(self, question: str, timeout: int) -> Optional[str]:
        """Consulta a Google Gemini CLI - modelo potente y rápido"""
        try:
            # Formato del prompt para Gemini
            prompt = f"Responde de forma técnica y detallada en español. Si la pregunta es sobre programación, incluye ejemplos de código cuando sea relevante:\n\n{question}"

            # Ejecutar gemini CLI
            result = subprocess.run(
                ["gemini", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            # gemini CLI no está instalado
            pass
        except Exception as e:
            pass

        return None

    def self_question_cycle(self, cognitive_age: int = 0) -> Optional[Dict]:
        """
        Ejecuta un ciclo completo de auto-cuestionamiento

        Returns:
            Diccionario con pregunta, respuesta y metadatos o None si no se puede ejecutar
        """
        # Verificar límites de seguridad
        can_ask, reason = self._can_ask_question()
        if not can_ask:
            print(f"⚠️  No se puede hacer pregunta: {reason}")
            return None

        print(f"\n💭 THAU se está haciendo una pregunta (Edad: {cognitive_age} años)...")

        # Generar pregunta
        question = self.generate_question(cognitive_age)
        if not question:
            print("❌ No se pudo generar pregunta")
            return None

        print(f"❓ Pregunta: {question}")

        # Responder pregunta
        answer = self.answer_question(question)
        if not answer:
            print("❌ No se pudo generar respuesta")
            return None

        print(f"✅ Respuesta generada ({len(answer)} caracteres)")

        # Registrar actividad
        self._record_question(question, answer)

        # Evaluar calidad de respuesta (heurística estricta)
        confidence = self._evaluate_response_quality(question, answer)

        result = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "cognitive_age": cognitive_age,
            "self_generated": True
        }

        # Guardar para análisis
        self._save_question_answer(result)

        return result

    def _evaluate_response_quality(self, question: str, answer: str) -> float:
        """
        Evalúa la calidad de una respuesta de forma estricta

        Returns:
            Confianza entre 0.0 y 1.0
        """
        if not answer or not question:
            return 0.0

        answer_lower = answer.lower().strip()
        question_lower = question.lower()

        # === DESCALIFICADORES INMEDIATOS (retorna 0.0) ===

        # Respuesta empieza con pregunta o puntuación extraña
        if answer.strip().startswith(("¿", "?", ",", ".", ";", "-")):
            return 0.0

        # Respuesta muy corta
        if len(answer) < 100:
            return 0.0

        # Respuesta en inglés cuando debería ser español
        english_starts = ["certainly", "here's", "here is", "i would", "i can",
                         "to help", "how can i", "sure!", "of course"]
        for eng in english_starts:
            if answer_lower.startswith(eng):
                return 0.0

        # Respuesta dice que no puede/sabe
        negative_patterns = [
            "no tengo acceso", "no puedo responder", "no puido",
            "no tenga acceso", "no puedo proporcionar", "necesito más contexto",
            "necesitaría más información", "no dispongo de"
        ]
        for pattern in negative_patterns:
            if pattern in answer_lower:
                return 0.0

        # Demasiadas preguntas en la respuesta
        if answer[:300].count("?") >= 2:
            return 0.0

        # Respuesta truncada (termina abruptamente)
        if answer.strip().endswith(("...", "..", " el", " la", " los", " las", " un", " una", " y", " o", " que", " de")):
            return 0.0

        # Contenido irrelevante (sitio web sin contexto)
        if "sitio web" in answer_lower and "web" not in question_lower:
            return 0.0

        # === FACTORES DE CALIDAD POSITIVOS ===
        confidence = 0.5  # Base

        # Longitud apropiada (100-1500 caracteres)
        if 100 <= len(answer) <= 1500:
            confidence += 0.15

        # Tiene estructura (explicación coherente)
        structure_words = ["es", "permite", "sirve", "utiliza", "consiste", "significa",
                         "ejemplo", "código", "función", "método", "clase"]
        if sum(1 for w in structure_words if w in answer_lower) >= 2:
            confidence += 0.15

        # Suficientes palabras
        if len(answer.split()) >= 20:
            confidence += 0.1

        # Contiene código (bueno para programación)
        if "```" in answer or "def " in answer or "function " in answer:
            confidence += 0.1

        return min(confidence, 1.0)

    def _is_valid_qa_pair(self, question: str, answer: str, category: str = "") -> bool:
        """
        Valida si un par pregunta-respuesta es de calidad suficiente para guardar

        Returns:
            True si debe guardarse, False si no
        """
        confidence = self._evaluate_response_quality(question, answer)
        return confidence >= 0.7

    def _save_question_answer(self, qa_data: Dict):
        """Guarda pregunta y respuesta solo si pasa validación de calidad"""
        question = qa_data.get("question", "")
        answer = qa_data.get("answer", "")
        category = qa_data.get("category", "")

        # Validar calidad antes de guardar
        if not self._is_valid_qa_pair(question, answer, category):
            print(f"   ⚠️  Descartado por baja calidad")
            return False

        qa_file = self.data_dir / f"qa_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(qa_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_data, ensure_ascii=False) + '\n')

        return True

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema de auto-cuestionamiento"""
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        current_day = now.strftime("%Y-%m-%d")

        return {
            "total_questions": self.activity_log.get("total_questions", 0),
            "questions_this_hour": self.activity_log.get("hourly_count", {}).get(current_hour, 0),
            "questions_today": self.activity_log.get("daily_count", {}).get(current_day, 0),
            "max_per_hour": self.max_questions_per_hour,
            "max_per_day": self.max_questions_per_day,
            "last_question_time": self.activity_log.get("last_question_time"),
            "can_ask_now": self._can_ask_question()[0]
        }


    def intensive_learning_session(
        self,
        cognitive_age: int = 0,
        num_questions: int = 50,
        category: Optional[str] = None,
        delay_between: float = 2.0
    ) -> Dict:
        """
        Sesión intensiva de aprendizaje - genera muchas preguntas rápidamente

        Args:
            cognitive_age: Edad cognitiva actual
            num_questions: Número de preguntas a generar
            category: Categoría específica (opcional)
            delay_between: Segundos entre preguntas

        Returns:
            Estadísticas de la sesión
        """
        print(f"\n🚀 Iniciando sesión intensiva de aprendizaje")
        print(f"   Preguntas objetivo: {num_questions}")
        print(f"   Edad cognitiva: {cognitive_age}")
        if category:
            print(f"   Categoría: {category}")
        print(f"{'='*60}\n")

        results = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "high_confidence": 0,
            "low_confidence": 0,
            "topics_covered": set(),
            "start_time": datetime.now().isoformat()
        }

        for i in range(num_questions):
            print(f"\n[{i+1}/{num_questions}] ", end="")

            # Generar y responder pregunta
            question = self.generate_question(cognitive_age, category=category)
            if not question:
                print("❌ No se pudo generar pregunta")
                results["failed"] += 1
                continue

            print(f"❓ {question[:60]}...")
            results["total_attempted"] += 1

            # Obtener respuesta
            answer = self.answer_question(question, timeout=60, use_model="auto")

            if answer:
                # Evaluar confianza
                confidence = self._evaluate_response_quality(question, answer)

                # Registrar
                self._record_question(question, answer)

                qa_data = {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "answer": answer,
                    "confidence": confidence,
                    "cognitive_age": cognitive_age,
                    "category": category,
                    "self_generated": True,
                    "intensive_session": True
                }
                self._save_question_answer(qa_data)

                results["successful"] += 1
                if confidence >= 0.7:
                    results["high_confidence"] += 1
                    print(f"✅ Alta confianza ({confidence:.2f})")
                else:
                    results["low_confidence"] += 1
                    print(f"⚠️  Baja confianza ({confidence:.2f})")

                # Extraer tópico
                words = question.split()
                for word in words:
                    clean = word.strip("¿?.,;:").lower()
                    if clean in self.exploration_concepts:
                        results["topics_covered"].add(clean)

            else:
                print("❌ Sin respuesta")
                results["failed"] += 1

            # Pequeña pausa
            if i < num_questions - 1:
                time.sleep(delay_between)

        results["end_time"] = datetime.now().isoformat()
        results["topics_covered"] = list(results["topics_covered"])

        # Resumen
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DE SESIÓN INTENSIVA")
        print(f"{'='*60}")
        print(f"   Intentadas: {results['total_attempted']}")
        print(f"   Exitosas: {results['successful']}")
        print(f"   Fallidas: {results['failed']}")
        print(f"   Alta confianza: {results['high_confidence']}")
        print(f"   Baja confianza: {results['low_confidence']}")
        print(f"   Tópicos cubiertos: {len(results['topics_covered'])}")

        return results

    def explore_category(self, category: str, cognitive_age: int = 0, depth: int = 20) -> Dict:
        """
        Explora profundamente una categoría específica

        Args:
            category: Nombre de la categoría a explorar
            cognitive_age: Edad cognitiva
            depth: Cantidad de preguntas

        Returns:
            Resultados de la exploración
        """
        if category not in self.knowledge_categories:
            print(f"❌ Categoría '{category}' no encontrada")
            print(f"   Disponibles: {list(self.knowledge_categories.keys())}")
            return {"error": "Categoría no encontrada"}

        print(f"\n🔬 Explorando categoría: {category}")
        return self.intensive_learning_session(
            cognitive_age=cognitive_age,
            num_questions=depth,
            category=category,
            delay_between=1.5
        )

    def get_available_categories(self) -> List[str]:
        """Retorna lista de categorías disponibles"""
        return list(self.knowledge_categories.keys())

    def get_knowledge_stats(self) -> Dict:
        """Obtiene estadísticas detalladas del conocimiento"""
        return {
            "total_concepts": len(self.exploration_concepts),
            "categories": len(self.knowledge_categories),
            "category_sizes": {k: len(v) for k, v in self.knowledge_categories.items()},
            "template_levels": list(self.topic_templates.keys()),
            "limits": {
                "questions_per_hour": self.max_questions_per_hour,
                "questions_per_day": self.max_questions_per_day,
                "min_seconds_between": self.min_seconds_between_questions
            }
        }


# Testing
if __name__ == "__main__":
    print("🧠 Probando Sistema de Auto-Cuestionamiento EXPANDIDO\n")

    system = SelfQuestioningSystem()

    # Mostrar estadísticas de conocimiento
    print("📚 Base de conocimiento:")
    stats = system.get_knowledge_stats()
    print(f"   Total conceptos: {stats['total_concepts']}")
    print(f"   Categorías: {stats['categories']}")
    print(f"   Niveles de plantillas: {stats['template_levels']}")
    print(f"\n   Límites:")
    print(f"      Preguntas/hora: {stats['limits']['questions_per_hour']}")
    print(f"      Preguntas/día: {stats['limits']['questions_per_day']}")
    print(f"      Espera mínima: {stats['limits']['min_seconds_between']}s")

    # Mostrar categorías
    print(f"\n📂 Categorías disponibles:")
    for cat in system.get_available_categories():
        size = stats['category_sizes'].get(cat, 0)
        print(f"   - {cat}: {size} conceptos")

    # Prueba rápida
    print(f"\n{'='*60}")
    print("🧪 Prueba rápida (3 preguntas)")
    print('='*60)

    for i in range(3):
        question = system.generate_question(cognitive_age=3)
        print(f"\n[{i+1}] {question}")

    # Estadísticas actuales
    print(f"\n{'='*60}")
    print("📈 Estadísticas del sistema")
    print('='*60)
    current_stats = system.get_stats()
    print(json.dumps(current_stats, indent=2, ensure_ascii=False))
