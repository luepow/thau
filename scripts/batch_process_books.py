"""Batch process programming books for THAU training."""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our modules
from pdf_extractor import PDFExtractor
from generate_python_qa import PythonQAGenerator, PYTHON_TOPICS

# Technology categorization
TECH_CATEGORIES = {
    "python": {
        "keywords": ["python", "django", "flask", "pandas", "numpy", "pytorch", "tensorflow"],
        "extensions": [],
    },
    "java": {
        "keywords": ["java", "spring", "hibernate", "maven", "gradle", "jvm", "jpos", "vaadin", "jmix"],
        "extensions": [],
    },
    "javascript": {
        "keywords": ["javascript", "nodejs", "node.js", "react", "angular", "vue", "typescript", "express"],
        "extensions": [],
    },
    "flutter_dart": {
        "keywords": ["flutter", "dart", "widget", "cupertino", "material design"],
        "extensions": [],
    },
    "sql_databases": {
        "keywords": ["sql", "mysql", "postgresql", "mongodb", "database", "query", "nosql", "redis"],
        "extensions": [],
    },
    "devops": {
        "keywords": ["docker", "kubernetes", "ci/cd", "jenkins", "aws", "azure", "cloud", "terraform"],
        "extensions": [],
    },
    "git": {
        "keywords": ["git", "github", "gitlab", "version control", "branching"],
        "extensions": [],
    },
    "rust_go": {
        "keywords": ["rust", "golang", "go lang", "cargo", "goroutine"],
        "extensions": [],
    },
    "web": {
        "keywords": ["html", "css", "web", "frontend", "responsive", "bootstrap"],
        "extensions": [],
    },
    "powershell": {
        "keywords": ["powershell", "cmdlet", "windows", "scripting"],
        "extensions": [],
    },
    "agile": {
        "keywords": ["scrum", "agile", "kanban", "sprint", "product owner"],
        "extensions": [],
    },
    "math": {
        "keywords": ["math", "mathematics", "algebra", "calculus", "statistics", "linear"],
        "extensions": [],
    },
    "algorithms": {
        "keywords": ["algorithm", "data structure", "sorting", "searching", "complexity"],
        "extensions": [],
    },
    "php": {
        "keywords": ["php", "laravel", "symfony", "wordpress", "composer"],
        "extensions": [],
    },
}

# Q&A Templates for different technologies
TECH_QA_TEMPLATES = {
    "java": [
        {"q": "¿Cómo se declara una clase en Java?", "a": "En Java, una clase se declara usando la palabra clave `class`:\n\n```java\npublic class MiClase {\n    // Atributos\n    private String nombre;\n    private int edad;\n    \n    // Constructor\n    public MiClase(String nombre, int edad) {\n        this.nombre = nombre;\n        this.edad = edad;\n    }\n    \n    // Métodos\n    public String getNombre() {\n        return nombre;\n    }\n    \n    public void setNombre(String nombre) {\n        this.nombre = nombre;\n    }\n}\n```"},
        {"q": "¿Cuáles son los tipos de datos primitivos en Java?", "a": "Java tiene 8 tipos primitivos:\n\n1. **byte**: 8 bits (-128 a 127)\n2. **short**: 16 bits (-32,768 a 32,767)\n3. **int**: 32 bits (-2^31 a 2^31-1)\n4. **long**: 64 bits (-2^63 a 2^63-1)\n5. **float**: 32 bits decimal\n6. **double**: 64 bits decimal\n7. **boolean**: true o false\n8. **char**: 16 bits Unicode\n\n```java\nint numero = 42;\ndouble precio = 19.99;\nboolean activo = true;\nchar letra = 'A';\nString texto = \"Hola\"; // No es primitivo, es objeto\n```"},
        {"q": "¿Cómo funcionan las interfaces en Java?", "a": "Las interfaces definen contratos que las clases deben implementar:\n\n```java\n// Definir interfaz\npublic interface Vehiculo {\n    void acelerar();\n    void frenar();\n    int getVelocidad();\n}\n\n// Implementar interfaz\npublic class Auto implements Vehiculo {\n    private int velocidad = 0;\n    \n    @Override\n    public void acelerar() {\n        velocidad += 10;\n    }\n    \n    @Override\n    public void frenar() {\n        velocidad -= 10;\n    }\n    \n    @Override\n    public int getVelocidad() {\n        return velocidad;\n    }\n}\n```\n\nDesde Java 8, las interfaces pueden tener métodos default y static."},
        {"q": "¿Qué son los Streams en Java?", "a": "Los Streams permiten procesar colecciones de forma funcional:\n\n```java\nimport java.util.List;\nimport java.util.stream.Collectors;\n\nList<String> nombres = List.of(\"Ana\", \"Juan\", \"Pedro\", \"María\");\n\n// Filtrar y transformar\nList<String> resultado = nombres.stream()\n    .filter(n -> n.length() > 3)\n    .map(String::toUpperCase)\n    .sorted()\n    .collect(Collectors.toList());\n// [JUAN, MARÍA, PEDRO]\n\n// Operaciones comunes\nnombres.stream().forEach(System.out::println);\nlong count = nombres.stream().count();\nboolean existe = nombres.stream().anyMatch(n -> n.startsWith(\"A\"));\n```"},
        {"q": "¿Cómo se maneja la herencia en Java?", "a": "Java usa `extends` para herencia de clases:\n\n```java\n// Clase padre\npublic class Animal {\n    protected String nombre;\n    \n    public Animal(String nombre) {\n        this.nombre = nombre;\n    }\n    \n    public void hacerSonido() {\n        System.out.println(\"Sonido genérico\");\n    }\n}\n\n// Clase hija\npublic class Perro extends Animal {\n    public Perro(String nombre) {\n        super(nombre); // Llama constructor padre\n    }\n    \n    @Override\n    public void hacerSonido() {\n        System.out.println(\"Guau!\");\n    }\n    \n    public void buscarPelota() {\n        System.out.println(nombre + \" busca la pelota\");\n    }\n}\n```\n\nJava solo permite herencia simple (una clase padre)."},
    ],
    "flutter_dart": [
        {"q": "¿Cómo se crea un Widget en Flutter?", "a": "En Flutter hay dos tipos de widgets:\n\n```dart\n// StatelessWidget - inmutable\nclass MiBoton extends StatelessWidget {\n  final String texto;\n  final VoidCallback onPressed;\n  \n  const MiBoton({Key? key, required this.texto, required this.onPressed}) : super(key: key);\n  \n  @override\n  Widget build(BuildContext context) {\n    return ElevatedButton(\n      onPressed: onPressed,\n      child: Text(texto),\n    );\n  }\n}\n\n// StatefulWidget - con estado mutable\nclass Contador extends StatefulWidget {\n  @override\n  _ContadorState createState() => _ContadorState();\n}\n\nclass _ContadorState extends State<Contador> {\n  int _count = 0;\n  \n  @override\n  Widget build(BuildContext context) {\n    return Column(\n      children: [\n        Text('Contador: $_count'),\n        ElevatedButton(\n          onPressed: () => setState(() => _count++),\n          child: Text('Incrementar'),\n        ),\n      ],\n    );\n  }\n}\n```"},
        {"q": "¿Cuáles son los layouts principales en Flutter?", "a": "Los layouts principales en Flutter son:\n\n```dart\n// Column - vertical\nColumn(\n  mainAxisAlignment: MainAxisAlignment.center,\n  crossAxisAlignment: CrossAxisAlignment.start,\n  children: [\n    Text('Item 1'),\n    Text('Item 2'),\n  ],\n)\n\n// Row - horizontal\nRow(\n  mainAxisAlignment: MainAxisAlignment.spaceBetween,\n  children: [\n    Icon(Icons.star),\n    Text('Rating'),\n  ],\n)\n\n// Stack - superpuestos\nStack(\n  children: [\n    Image.network('url'),\n    Positioned(\n      bottom: 10,\n      right: 10,\n      child: Text('Overlay'),\n    ),\n  ],\n)\n\n// ListView - scroll\nListView.builder(\n  itemCount: items.length,\n  itemBuilder: (context, index) => ListTile(\n    title: Text(items[index]),\n  ),\n)\n```"},
        {"q": "¿Cómo se maneja el estado en Flutter?", "a": "Flutter tiene varias opciones para manejo de estado:\n\n```dart\n// 1. setState (local)\nsetState(() {\n  _counter++;\n});\n\n// 2. Provider\nclass CounterProvider extends ChangeNotifier {\n  int _count = 0;\n  int get count => _count;\n  \n  void increment() {\n    _count++;\n    notifyListeners();\n  }\n}\n\n// Usar Provider\nChangeNotifierProvider(\n  create: (_) => CounterProvider(),\n  child: Consumer<CounterProvider>(\n    builder: (context, counter, child) {\n      return Text('${counter.count}');\n    },\n  ),\n)\n\n// 3. BLoC\nclass CounterBloc extends Bloc<CounterEvent, int> {\n  CounterBloc() : super(0) {\n    on<Increment>((event, emit) => emit(state + 1));\n  }\n}\n```"},
        {"q": "¿Cómo se navega entre pantallas en Flutter?", "a": "La navegación en Flutter usa Navigator:\n\n```dart\n// Navegación básica\nNavigator.push(\n  context,\n  MaterialPageRoute(builder: (context) => SegundaPantalla()),\n);\n\n// Volver atrás\nNavigator.pop(context);\n\n// Con parámetros\nNavigator.push(\n  context,\n  MaterialPageRoute(\n    builder: (context) => DetallePantalla(id: itemId),\n  ),\n);\n\n// Retornar datos\nfinal result = await Navigator.push(\n  context,\n  MaterialPageRoute(builder: (context) => SeleccionPantalla()),\n);\n\n// Named routes\n// En MaterialApp:\nroutes: {\n  '/': (context) => HomePage(),\n  '/detalle': (context) => DetallePage(),\n}\n\n// Navegar\nNavigator.pushNamed(context, '/detalle');\n\n// Con argumentos\nNavigator.pushNamed(\n  context,\n  '/detalle',\n  arguments: {'id': 123},\n);\n```"},
    ],
    "sql_databases": [
        {"q": "¿Cuáles son los comandos básicos de SQL?", "a": "Los comandos SQL se dividen en categorías:\n\n```sql\n-- DDL (Data Definition Language)\nCREATE TABLE usuarios (\n    id INT PRIMARY KEY AUTO_INCREMENT,\n    nombre VARCHAR(100) NOT NULL,\n    email VARCHAR(255) UNIQUE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);\n\nALTER TABLE usuarios ADD COLUMN edad INT;\nDROP TABLE usuarios;\n\n-- DML (Data Manipulation Language)\nINSERT INTO usuarios (nombre, email) VALUES ('Juan', 'juan@mail.com');\nUPDATE usuarios SET nombre = 'Juan Pablo' WHERE id = 1;\nDELETE FROM usuarios WHERE id = 1;\n\n-- DQL (Data Query Language)\nSELECT * FROM usuarios;\nSELECT nombre, email FROM usuarios WHERE edad > 18;\nSELECT COUNT(*) FROM usuarios;\n```"},
        {"q": "¿Cómo funcionan los JOINs en SQL?", "a": "Los JOINs combinan datos de múltiples tablas:\n\n```sql\n-- INNER JOIN - solo coincidencias\nSELECT u.nombre, o.total\nFROM usuarios u\nINNER JOIN ordenes o ON u.id = o.usuario_id;\n\n-- LEFT JOIN - todos de la izquierda\nSELECT u.nombre, o.total\nFROM usuarios u\nLEFT JOIN ordenes o ON u.id = o.usuario_id;\n\n-- RIGHT JOIN - todos de la derecha\nSELECT u.nombre, o.total\nFROM usuarios u\nRIGHT JOIN ordenes o ON u.id = o.usuario_id;\n\n-- FULL OUTER JOIN - todos\nSELECT u.nombre, o.total\nFROM usuarios u\nFULL OUTER JOIN ordenes o ON u.id = o.usuario_id;\n\n-- Múltiples JOINs\nSELECT u.nombre, o.id, p.nombre as producto\nFROM usuarios u\nJOIN ordenes o ON u.id = o.usuario_id\nJOIN productos p ON o.producto_id = p.id;\n```"},
        {"q": "¿Cómo se usan las funciones de agregación en SQL?", "a": "Las funciones de agregación resumen datos:\n\n```sql\n-- Funciones básicas\nSELECT \n    COUNT(*) as total_usuarios,\n    AVG(edad) as edad_promedio,\n    MAX(edad) as edad_maxima,\n    MIN(edad) as edad_minima,\n    SUM(saldo) as saldo_total\nFROM usuarios;\n\n-- GROUP BY\nSELECT \n    ciudad,\n    COUNT(*) as cantidad,\n    AVG(edad) as edad_promedio\nFROM usuarios\nGROUP BY ciudad;\n\n-- HAVING (filtrar grupos)\nSELECT \n    ciudad,\n    COUNT(*) as cantidad\nFROM usuarios\nGROUP BY ciudad\nHAVING COUNT(*) > 10;\n\n-- ORDER BY\nSELECT nombre, edad\nFROM usuarios\nORDER BY edad DESC, nombre ASC;\n\n-- LIMIT/OFFSET\nSELECT * FROM usuarios\nORDER BY created_at DESC\nLIMIT 10 OFFSET 20;\n```"},
    ],
    "docker": [
        {"q": "¿Cómo se crea un Dockerfile?", "a": "Un Dockerfile define cómo construir una imagen:\n\n```dockerfile\n# Imagen base\nFROM python:3.11-slim\n\n# Metadatos\nLABEL maintainer=\"tu@email.com\"\n\n# Variables de entorno\nENV PYTHONDONTWRITEBYTECODE=1\nENV PYTHONUNBUFFERED=1\n\n# Directorio de trabajo\nWORKDIR /app\n\n# Copiar requirements primero (cache de layers)\nCOPY requirements.txt .\n\n# Instalar dependencias\nRUN pip install --no-cache-dir -r requirements.txt\n\n# Copiar código\nCOPY . .\n\n# Puerto expuesto\nEXPOSE 8000\n\n# Comando de ejecución\nCMD [\"python\", \"app.py\"]\n\n# O con ENTRYPOINT\nENTRYPOINT [\"python\"]\nCMD [\"app.py\"]\n```"},
        {"q": "¿Cuáles son los comandos básicos de Docker?", "a": "Comandos esenciales de Docker:\n\n```bash\n# Imágenes\ndocker build -t mi-app:1.0 .\ndocker images\ndocker pull nginx:latest\ndocker push usuario/mi-app:1.0\ndocker rmi imagen:tag\n\n# Contenedores\ndocker run -d -p 8080:80 --name web nginx\ndocker ps              # contenedores activos\ndocker ps -a           # todos los contenedores\ndocker stop web\ndocker start web\ndocker restart web\ndocker rm web\n\n# Logs y acceso\ndocker logs web\ndocker logs -f web     # follow\ndocker exec -it web bash\n\n# Volúmenes\ndocker volume create datos\ndocker run -v datos:/app/data mi-app\ndocker run -v $(pwd):/app mi-app  # bind mount\n\n# Redes\ndocker network create mi-red\ndocker run --network mi-red mi-app\n\n# Limpieza\ndocker system prune\ndocker image prune\n```"},
        {"q": "¿Cómo se usa Docker Compose?", "a": "Docker Compose orquesta múltiples contenedores:\n\n```yaml\n# docker-compose.yml\nversion: '3.8'\n\nservices:\n  web:\n    build: .\n    ports:\n      - \"8000:8000\"\n    environment:\n      - DATABASE_URL=postgres://db:5432/app\n    depends_on:\n      - db\n      - redis\n    volumes:\n      - .:/app\n    networks:\n      - app-network\n\n  db:\n    image: postgres:15\n    environment:\n      POSTGRES_DB: app\n      POSTGRES_PASSWORD: secret\n    volumes:\n      - postgres_data:/var/lib/postgresql/data\n    networks:\n      - app-network\n\n  redis:\n    image: redis:alpine\n    networks:\n      - app-network\n\nvolumes:\n  postgres_data:\n\nnetworks:\n  app-network:\n    driver: bridge\n```\n\n```bash\n# Comandos\ndocker-compose up -d\ndocker-compose down\ndocker-compose logs -f\ndocker-compose ps\ndocker-compose exec web bash\n```"},
    ],
    "git": [
        {"q": "¿Cuáles son los comandos básicos de Git?", "a": "Comandos esenciales de Git:\n\n```bash\n# Configuración inicial\ngit config --global user.name \"Tu Nombre\"\ngit config --global user.email \"tu@email.com\"\n\n# Iniciar repositorio\ngit init\ngit clone https://github.com/user/repo.git\n\n# Estado y cambios\ngit status\ngit diff\ngit diff --staged\n\n# Agregar y confirmar\ngit add archivo.txt\ngit add .\ngit commit -m \"Mensaje descriptivo\"\ngit commit -am \"Add y commit juntos\"\n\n# Historial\ngit log\ngit log --oneline\ngit log --graph\n\n# Ramas\ngit branch\ngit branch nueva-rama\ngit checkout nueva-rama\ngit checkout -b nueva-rama  # crear y cambiar\ngit merge otra-rama\ngit branch -d rama-a-eliminar\n\n# Remoto\ngit remote add origin https://github.com/user/repo.git\ngit push -u origin main\ngit pull\ngit fetch\n```"},
        {"q": "¿Cómo se resuelven conflictos en Git?", "a": "Proceso para resolver conflictos:\n\n```bash\n# 1. Al hacer merge o pull, Git marca conflictos\ngit merge feature-branch\n# CONFLICT (content): Merge conflict in archivo.txt\n\n# 2. Abrir archivo y buscar marcadores\n<<<<<<< HEAD\nCódigo de tu rama actual\n=======\nCódigo de la rama que estás mergeando\n>>>>>>> feature-branch\n\n# 3. Editar manualmente, elegir qué código mantener\n# Eliminar los marcadores <<<<, ====, >>>>\n\n# 4. Marcar como resuelto\ngit add archivo.txt\n\n# 5. Completar el merge\ngit commit -m \"Resolver conflicto en archivo.txt\"\n\n# Alternativas\ngit merge --abort    # Cancelar merge\ngit checkout --ours archivo.txt   # Mantener nuestra versión\ngit checkout --theirs archivo.txt # Mantener su versión\n\n# Herramientas visuales\ngit mergetool\n```"},
    ],
    "spring": [
        {"q": "¿Cómo se crea un REST Controller en Spring Boot?", "a": "Un REST Controller en Spring Boot:\n\n```java\nimport org.springframework.web.bind.annotation.*;\nimport org.springframework.http.ResponseEntity;\nimport java.util.List;\n\n@RestController\n@RequestMapping(\"/api/usuarios\")\npublic class UsuarioController {\n\n    private final UsuarioService usuarioService;\n\n    public UsuarioController(UsuarioService usuarioService) {\n        this.usuarioService = usuarioService;\n    }\n\n    @GetMapping\n    public List<Usuario> listar() {\n        return usuarioService.findAll();\n    }\n\n    @GetMapping(\"/{id}\")\n    public ResponseEntity<Usuario> obtener(@PathVariable Long id) {\n        return usuarioService.findById(id)\n            .map(ResponseEntity::ok)\n            .orElse(ResponseEntity.notFound().build());\n    }\n\n    @PostMapping\n    public ResponseEntity<Usuario> crear(@RequestBody @Valid Usuario usuario) {\n        Usuario nuevo = usuarioService.save(usuario);\n        return ResponseEntity.status(HttpStatus.CREATED).body(nuevo);\n    }\n\n    @PutMapping(\"/{id}\")\n    public ResponseEntity<Usuario> actualizar(\n            @PathVariable Long id,\n            @RequestBody @Valid Usuario usuario) {\n        return usuarioService.update(id, usuario)\n            .map(ResponseEntity::ok)\n            .orElse(ResponseEntity.notFound().build());\n    }\n\n    @DeleteMapping(\"/{id}\")\n    public ResponseEntity<Void> eliminar(@PathVariable Long id) {\n        usuarioService.delete(id);\n        return ResponseEntity.noContent().build();\n    }\n}\n```"},
        {"q": "¿Cómo se configura Spring Data JPA?", "a": "Configuración de Spring Data JPA:\n\n```java\n// Entity\n@Entity\n@Table(name = \"usuarios\")\npublic class Usuario {\n    @Id\n    @GeneratedValue(strategy = GenerationType.IDENTITY)\n    private Long id;\n\n    @Column(nullable = false)\n    private String nombre;\n\n    @Column(unique = true)\n    private String email;\n\n    @OneToMany(mappedBy = \"usuario\", cascade = CascadeType.ALL)\n    private List<Orden> ordenes;\n\n    // getters y setters\n}\n\n// Repository\npublic interface UsuarioRepository extends JpaRepository<Usuario, Long> {\n    Optional<Usuario> findByEmail(String email);\n    List<Usuario> findByNombreContaining(String nombre);\n    \n    @Query(\"SELECT u FROM Usuario u WHERE u.edad > :edad\")\n    List<Usuario> findMayoresQue(@Param(\"edad\") int edad);\n}\n\n// application.yml\nspring:\n  datasource:\n    url: jdbc:postgresql://localhost:5432/mydb\n    username: user\n    password: pass\n  jpa:\n    hibernate:\n      ddl-auto: update\n    show-sql: true\n```"},
    ],
}


class BatchBookProcessor:
    """Process multiple books and generate training datasets."""

    def __init__(
        self,
        books_dir: str,
        output_dir: str = "./data/extracted",
        datasets_dir: str = "./data/datasets",
    ):
        self.books_dir = Path(books_dir)
        self.output_dir = Path(output_dir)
        self.datasets_dir = Path(datasets_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = PDFExtractor(output_dir=str(self.output_dir))
        self.processed_books = []
        self.failed_books = []

    def categorize_book(self, filename: str) -> str:
        """Determine the category of a book based on filename."""
        filename_lower = filename.lower()

        for category, info in TECH_CATEGORIES.items():
            for keyword in info["keywords"]:
                if keyword in filename_lower:
                    return category

        return "general"

    def find_all_books(self) -> Dict[str, List[Path]]:
        """Find all PDF books and categorize them."""
        categorized = {cat: [] for cat in TECH_CATEGORIES}
        categorized["general"] = []

        # Find PDFs in main directory and subdirectories
        for pdf_path in self.books_dir.rglob("*.pdf"):
            # Skip very small files (likely not real books)
            if pdf_path.stat().st_size < 100000:  # < 100KB
                continue

            category = self.categorize_book(pdf_path.name)
            categorized[category].append(pdf_path)

        # Log summary
        for cat, books in categorized.items():
            if books:
                logger.info(f"{cat}: {len(books)} books")

        return categorized

    def process_single_book(self, pdf_path: Path) -> Tuple[bool, str, Dict]:
        """Process a single PDF book."""
        try:
            logger.info(f"Processing: {pdf_path.name}")
            result = self.extractor.process_pdf(str(pdf_path))
            return True, str(pdf_path), result
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return False, str(pdf_path), {"error": str(e)}

    def process_all_books(self, max_per_category: int = 10) -> Dict:
        """Process all books in parallel."""
        categorized = self.find_all_books()
        results = {"processed": [], "failed": [], "by_category": {}}

        all_books = []
        for category, books in categorized.items():
            # Limit books per category
            selected = books[:max_per_category]
            for book in selected:
                all_books.append((category, book))

        logger.info(f"Processing {len(all_books)} books total...")

        # Process with thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_single_book, book): (cat, book)
                for cat, book in all_books
            }

            for future in as_completed(futures):
                cat, book = futures[future]
                success, path, result = future.result()

                if success:
                    results["processed"].append({"category": cat, "path": path, **result})
                    if cat not in results["by_category"]:
                        results["by_category"][cat] = []
                    results["by_category"][cat].append(path)
                else:
                    results["failed"].append({"category": cat, "path": path, **result})

        return results

    def generate_qa_for_category(self, category: str, text_files: List[str]) -> List[Dict]:
        """Generate Q&A pairs for a category."""
        qa_pairs = []

        # Add template Q&A for known categories
        if category in TECH_QA_TEMPLATES:
            for template in TECH_QA_TEMPLATES[category]:
                qa_pairs.append({
                    "instruction": template["q"],
                    "input": "",
                    "output": template["a"],
                    "category": category,
                })

        # Extract Q&A from book content
        for text_file in text_files:
            text_path = self.output_dir / f"{Path(text_file).stem}_clean.txt"
            if text_path.exists():
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Generate Q&A from content
                    extracted_qa = self._extract_qa_from_content(content, category)
                    qa_pairs.extend(extracted_qa)
                except Exception as e:
                    logger.warning(f"Could not process {text_path}: {e}")

        return qa_pairs

    def _extract_qa_from_content(self, content: str, category: str) -> List[Dict]:
        """Extract Q&A pairs from book content."""
        qa_pairs = []

        # Find paragraphs that look like explanations
        paragraphs = content.split('\n\n')

        keywords = TECH_CATEGORIES.get(category, {}).get("keywords", [])

        for para in paragraphs:
            if len(para) < 100 or len(para) > 1000:
                continue

            # Check if paragraph contains relevant keywords
            para_lower = para.lower()
            matches = sum(1 for kw in keywords if kw in para_lower)

            if matches >= 1:
                # Generate a question based on content
                first_sentence = para.split('.')[0].strip()
                if len(first_sentence) > 20:
                    qa_pairs.append({
                        "instruction": f"Explica: {first_sentence[:100]}",
                        "input": "",
                        "output": para.strip(),
                        "category": category,
                        "source": "book_extraction",
                    })

        return qa_pairs[:20]  # Limit per book

    def generate_all_datasets(self, process_results: Dict) -> Dict[str, Path]:
        """Generate datasets for all categories."""
        dataset_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d")

        for category, book_paths in process_results["by_category"].items():
            if not book_paths:
                continue

            logger.info(f"Generating dataset for {category}...")
            qa_pairs = self.generate_qa_for_category(category, book_paths)

            if qa_pairs:
                output_path = self.datasets_dir / f"{category}_training_{timestamp}.jsonl"
                with open(output_path, 'w', encoding='utf-8') as f:
                    for qa in qa_pairs:
                        f.write(json.dumps(qa, ensure_ascii=False) + '\n')

                dataset_paths[category] = output_path
                logger.info(f"  Saved {len(qa_pairs)} Q&A pairs to {output_path}")

        return dataset_paths

    def create_combined_dataset(self, dataset_paths: Dict[str, Path]) -> Path:
        """Combine all category datasets into one."""
        combined = []

        for category, path in dataset_paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        combined.append(json.loads(line))

        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = self.datasets_dir / f"programming_combined_{timestamp}.jsonl"

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in combined:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')

        logger.info(f"Combined dataset: {len(combined)} Q&A pairs -> {output_path}")
        return output_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Batch process programming books")
    parser.add_argument(
        "--books-dir",
        default="/Users/lperez/Library/CloudStorage/Dropbox/Books",
        help="Directory containing PDF books"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/extracted",
        help="Directory for extracted text"
    )
    parser.add_argument(
        "--datasets-dir",
        default="./data/datasets",
        help="Directory for generated datasets"
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=5,
        help="Maximum books to process per category"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories to process (default: all)"
    )
    args = parser.parse_args()

    processor = BatchBookProcessor(
        books_dir=args.books_dir,
        output_dir=args.output_dir,
        datasets_dir=args.datasets_dir,
    )

    # Process books
    print("=" * 60)
    print("BATCH BOOK PROCESSOR FOR THAU TRAINING")
    print("=" * 60)

    results = processor.process_all_books(max_per_category=args.max_per_category)

    print(f"\n✓ Processed: {len(results['processed'])} books")
    print(f"✗ Failed: {len(results['failed'])} books")

    # Generate datasets
    print("\nGenerating training datasets...")
    dataset_paths = processor.generate_all_datasets(results)

    # Create combined dataset
    if dataset_paths:
        combined_path = processor.create_combined_dataset(dataset_paths)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\nDatasets generated:")
        for cat, path in dataset_paths.items():
            print(f"  - {cat}: {path}")
        print(f"\nCombined dataset: {combined_path}")
        print(f"\nTo train THAU, run:")
        print(f"  python scripts/train_language_specialization.py \\")
        print(f"    --language programming \\")
        print(f"    --datasets {combined_path}")


if __name__ == "__main__":
    main()
