"""
API FastAPI para THAU Integrado
Combina desarrollo cognitivo, auto-aprendizaje, memoria vectorizada,
multilingüismo y protocolo MCP
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uvicorn
import asyncio

from thau_trainer.integrated_trainer import IntegratedTHAUTrainer
from thau_trainer.mcp_server import MCPServer


# Pydantic models para requests/responses
class InteractionRequest(BaseModel):
    question: str = Field(..., description="Pregunta del usuario")
    answer: str = Field(..., description="Respuesta de THAU")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confianza en la respuesta")

class MemoryQueryRequest(BaseModel):
    query: str = Field(..., description="Consulta para buscar en memoria")
    k: int = Field(3, ge=1, le=20, description="Número de resultados")

class LearnWordRequest(BaseModel):
    word: str
    language: str = Field(..., description="Código de idioma (es, en, fr, etc.)")
    definition: str
    pos: Optional[str] = None
    examples: Optional[List[str]] = None

class MCPToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class StatusResponse(BaseModel):
    cognitive_age: int
    stage_name: str
    progress_pct: float
    total_interactions: int
    memory_vectors: int
    auto_learning_active: bool
    languages: List[str]


# Crear FastAPI app
app = FastAPI(
    title="THAU API",
    description="API para el sistema de entrenamiento autónomo THAU",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del trainer
thau_trainer: Optional[IntegratedTHAUTrainer] = None
mcp_server: Optional[MCPServer] = None


@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar"""
    global thau_trainer, mcp_server

    print("\n" + "="*70)
    print("🚀 Iniciando THAU API Integrada")
    print("="*70 + "\n")

    # Inicializar trainer integrado
    thau_trainer = IntegratedTHAUTrainer(auto_train_enabled=True)

    # Inicializar servidor MCP
    mcp_server = MCPServer()

    # Inicializar idiomas básicos
    thau_trainer.language_manager.initialize_spanish_basics()
    thau_trainer.language_manager.add_language("en")
    thau_trainer.language_manager.initialize_english_basics()

    # Iniciar loop de auto-mejora (cada 6 horas)
    thau_trainer.start_auto_improvement_loop(interval_hours=6)

    print("\n✅ THAU API lista\n")
    print(f"📍 Documentación: http://localhost:8000/docs")
    print(f"📍 MCP Tools: {len(mcp_server.tools)} herramientas disponibles")
    print(f"📍 Edad cognitiva: {thau_trainer.cognitive_manager.current_age} años")
    print()


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al apagar"""
    global thau_trainer

    if thau_trainer:
        thau_trainer.stop_auto_improvement_loop()
        thau_trainer.save_all()

    print("\n👋 THAU API detenida")


# ============== ENDPOINTS ==============

@app.get("/", summary="Información del servidor")
async def root():
    """Información básica del servidor"""
    return {
        "name": "THAU API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Desarrollo cognitivo (edades 0-15+)",
            "Auto-generación de datasets",
            "Memoria vectorizada",
            "Aprendizaje multilingüe",
            "Protocolo MCP"
        ]
    }


@app.get("/status", response_model=StatusResponse, summary="Estado completo del sistema")
async def get_status():
    """Obtiene estado completo de THAU"""
    status = thau_trainer.get_full_status()

    return StatusResponse(
        cognitive_age=status["cognitive"]["current_age"],
        stage_name=status["cognitive"]["stage_name"],
        progress_pct=status["cognitive"]["progress"]["progress_pct"],
        total_interactions=status["general"]["total_interactions"],
        memory_vectors=status["memory"]["active_vectors"],
        auto_learning_active=status["auto_improvement_running"],
        languages=thau_trainer.language_manager.active_languages
    )


@app.post("/interact", summary="Procesar interacción")
async def process_interaction(request: InteractionRequest):
    """
    Procesa una interacción completa:
    - Detecta brechas de conocimiento
    - Almacena en memoria vectorizada
    - Añade a cola de entrenamiento
    """
    result = thau_trainer.process_interaction(
        question=request.question,
        answer=request.answer,
        confidence=request.confidence
    )

    return result


@app.post("/memory/recall", summary="Recuperar de memoria")
async def recall_memory(request: MemoryQueryRequest):
    """Busca información en la memoria vectorizada"""
    results = thau_trainer.recall_from_memory(
        query=request.query,
        k=request.k
    )

    return {
        "query": request.query,
        "results": results,
        "total_found": len(results)
    }


@app.post("/train", summary="Entrenar ahora")
async def train_now(force: bool = False):
    """Ejecuta entrenamiento inmediato"""
    result = thau_trainer.train_now(force=force)

    return result


@app.post("/auto-improve", summary="Auto-mejorar conocimiento")
async def auto_improve(min_gaps: int = 5):
    """Genera datasets para cubrir brechas de conocimiento"""
    result = thau_trainer.auto_improve_knowledge(min_gaps=min_gaps)

    return result


@app.get("/cognitive/status", summary="Estado cognitivo")
async def cognitive_status():
    """Estado del desarrollo cognitivo"""
    return thau_trainer.cognitive_manager.get_status()


@app.post("/cognitive/advance", summary="Avanzar edad cognitiva")
async def advance_age():
    """
    Verifica y avanza a la siguiente edad si cumple criterios
    """
    can_advance = thau_trainer.cognitive_manager.check_advancement()

    if not can_advance:
        status = thau_trainer.cognitive_manager.get_status()
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No se puede avanzar. Criterios no cumplidos.",
                "current_age": status["current_age"],
                "progress": status["progress"]
            }
        )

    advanced = thau_trainer.check_and_advance_age()

    return {
        "advanced": advanced,
        "new_age": thau_trainer.cognitive_manager.current_age,
        "stage_name": thau_trainer.cognitive_manager.stage.name
    }


# ============== REASONING ENDPOINTS ==============

@app.post("/reasoning/chain-of-thought", summary="Razonamiento paso a paso")
async def chain_of_thought_reasoning(request: Dict):
    """Responde usando razonamiento Chain-of-Thought"""
    question = request.get("question")
    max_steps = request.get("max_steps", 5)

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    result = thau_trainer.answer_with_reasoning(question, max_steps)
    return result


@app.post("/reasoning/explore", summary="Explorar múltiples respuestas")
async def explore_reasoning(request: Dict):
    """Explora múltiples caminos de razonamiento"""
    question = request.get("question")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    result = thau_trainer.explore_answers(question)
    return result


@app.post("/reasoning/plan", summary="Planificar tarea compleja")
async def plan_task(request: Dict):
    """Crea un plan para lograr un objetivo"""
    goal = request.get("goal")
    context = request.get("context")

    if not goal:
        raise HTTPException(status_code=400, detail="Goal is required")

    result = thau_trainer.plan_complex_task(goal, context)
    return result


@app.post("/reasoning/improve", summary="Mejorar respuesta")
async def improve_response(request: Dict):
    """Mejora una respuesta usando auto-reflexión"""
    question = request.get("question")
    initial_response = request.get("response")
    criteria = request.get("criteria")

    if not question or not initial_response:
        raise HTTPException(status_code=400, detail="Question and response are required")

    result = thau_trainer.improve_response(question, initial_response, criteria)
    return result


@app.get("/reasoning/stats", summary="Estadísticas de razonamiento")
async def reasoning_stats():
    """Obtiene estadísticas de los sistemas de razonamiento"""
    status = thau_trainer.get_full_status()
    return status.get("reasoning", {})


@app.post("/language/learn-word", summary="Aprender palabra")
async def learn_word(request: LearnWordRequest):
    """Aprende una nueva palabra"""
    entry = thau_trainer.language_manager.learn_word(
        word=request.word,
        language=request.language,
        definition=request.definition,
        pos=request.pos,
        examples=request.examples
    )

    return entry


@app.get("/language/progress/{language}", summary="Progreso de idioma")
async def language_progress(language: str):
    """Progreso de aprendizaje de un idioma"""
    progress = thau_trainer.language_manager.get_learning_progress(language)

    return progress


@app.post("/language/add", summary="Añadir idioma")
async def add_language(language_code: str):
    """Comienza a aprender un nuevo idioma"""
    success = thau_trainer.language_manager.add_language(language_code)

    if not success:
        raise HTTPException(status_code=400, detail=f"Idioma '{language_code}' no soportado")

    return {
        "language": language_code,
        "status": "added",
        "active_languages": thau_trainer.language_manager.active_languages
    }


# ============== MCP ENDPOINTS ==============

@app.get("/mcp/tools", summary="Listar herramientas MCP")
async def mcp_list_tools():
    """Lista todas las herramientas MCP disponibles"""
    response = await mcp_server.handle_tools_list()
    return response


@app.post("/mcp/call", summary="Llamar herramienta MCP")
async def mcp_call_tool(request: MCPToolCallRequest):
    """Ejecuta una herramienta MCP"""
    response = await mcp_server.handle_tools_call(
        name=request.tool_name,
        arguments=request.arguments
    )

    return response


@app.get("/mcp/resources", summary="Listar recursos MCP")
async def mcp_list_resources():
    """Lista todos los recursos MCP disponibles"""
    response = await mcp_server.handle_resources_list()
    return response


# ============== STATS & MONITORING ==============

@app.get("/stats/memory", summary="Estadísticas de memoria")
async def memory_stats():
    """Estadísticas de la memoria vectorizada"""
    return thau_trainer.vector_memory.get_stats()


@app.get("/stats/self-learning", summary="Estadísticas de auto-aprendizaje")
async def self_learning_stats():
    """Estadísticas del sistema de auto-aprendizaje"""
    return thau_trainer.self_learning.get_stats()


@app.get("/stats/self-questioning", summary="Estadísticas de auto-cuestionamiento")
async def self_questioning_stats():
    """Estadísticas del sistema de auto-cuestionamiento autónomo"""
    return thau_trainer.self_questioning.get_stats()


@app.get("/stats/full", summary="Estadísticas completas del sistema")
async def full_stats():
    """Estadísticas completas de THAU incluyendo modelo propio"""
    return thau_trainer.get_full_status()


@app.get("/stats/datasets", summary="Datasets generados")
async def datasets_stats():
    """Información sobre datasets auto-generados"""
    from pathlib import Path

    gen_dir = Path("./data/datasets/auto_generated")

    if not gen_dir.exists():
        return {"total_datasets": 0, "datasets": []}

    datasets = []
    for file in gen_dir.glob("*.jsonl"):
        # Contar líneas
        with open(file, 'r') as f:
            num_examples = sum(1 for _ in f)

        datasets.append({
            "filename": file.name,
            "examples": num_examples,
            "created": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
        })

    return {
        "total_datasets": len(datasets),
        "datasets": sorted(datasets, key=lambda x: x["created"], reverse=True)
    }


# ============== HEALTH CHECK ==============

@app.get("/health", summary="Health check")
async def health_check():
    """Verifica que todos los componentes estén funcionando"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "cognitive_manager": thau_trainer.cognitive_manager is not None,
            "vector_memory": thau_trainer.vector_memory is not None,
            "language_manager": thau_trainer.language_manager is not None,
            "mcp_server": mcp_server is not None
        }
    }


# ============== DASHBOARD COMPLETO ==============

@app.get("/dashboard", summary="Dashboard completo de THAU")
async def get_dashboard():
    """
    Endpoint completo con toda la información de THAU:
    - Estado cognitivo y progreso
    - Memoria vectorizada
    - Auto-aprendizaje
    - Idiomas
    - Herramientas MCP
    - Estadísticas generales
    """
    try:
        # Información cognitiva
        cognitive_status = thau_trainer.cognitive_manager.get_status()

        # Información de memoria
        memory_info = thau_trainer.vector_memory.get_stats()

        # Información de auto-aprendizaje
        self_learning_stats = thau_trainer.self_learning.get_stats()

        # Información de idiomas
        active_languages = thau_trainer.language_manager.active_languages

        # Estadísticas generales
        stats = thau_trainer.stats

        # Información MCP (simplificado)
        mcp_tools_info = {
            "web_search": "Búsqueda web para encontrar información",
            "execute_python": "Ejecutar código Python seguro",
            "recall_memory": "Buscar en memoria vectorizada",
            "learn_word": "Aprender vocabulario",
            "generate_dataset": "Crear datasets automáticamente"
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "status": "healthy",
                "uptime_info": "Running",
                "version": "1.0.0"
            },
            "cognitive": {
                "age": cognitive_status["current_age"],
                "stage_name": cognitive_status["stage_name"],
                "stage_description": cognitive_status["description"],
                "progress_percentage": cognitive_status["progress"]["progress_pct"],
                "examples_in_current_age": cognitive_status["progress"]["examples_at_age"],
                "examples_needed": cognitive_status["progress"]["examples_needed"],
                "can_advance": cognitive_status["progress"]["can_advance"],
                "total_age_progressions": stats.get("age_progressions", 0),
                "capabilities": cognitive_status.get("capabilities", [])
            },
            "memory": {
                "total_vectors": memory_info["total_vectors"],
                "active_vectors": memory_info["active_vectors"],
                "deleted_vectors": memory_info.get("deleted_vectors", 0),
                "dimension": memory_info["dimension"],
                "index_type": memory_info["index_type"],
                "memory_size_mb": memory_info["memory_size_mb"],
                "capacity": "Unlimited (Flat index)"
            },
            "auto_learning": {
                "gaps_detected": self_learning_stats["total_gaps_detected"],
                "datasets_generated": self_learning_stats["total_datasets_generated"],
                "examples_generated": self_learning_stats["total_examples_generated"],
                "last_generation": self_learning_stats.get("last_generation", "Never"),
                "auto_improve_active": True
            },
            "languages": {
                "total_active": len(active_languages),
                "languages": active_languages
            },
            "mcp": {
                "total_tools": len(mcp_tools_info),
                "tools": [{"name": name, "description": desc} for name, desc in mcp_tools_info.items()]
            },
            "statistics": {
                "total_interactions": stats.get("total_interactions", 0),
                "total_trainings": stats.get("total_trainings", 0),
                "age_progressions": stats.get("age_progressions", 0)
            },
            "model_info": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "base_model": "qwen2.5-coder:1.5b-base (via Ollama)",
                "model_parameters": "1.5B",
                "model_size_description": "1.5 mil millones de parámetros",
                "device": "MPS (Apple Silicon)"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")


@app.get("/dashboard/html", summary="Dashboard HTML visual")
async def get_dashboard_html():
    """Dashboard visual en HTML para ver en el navegador"""
    from fastapi.responses import HTMLResponse

    # Obtener datos del dashboard
    dashboard_data = await get_dashboard()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>THAU Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                text-align: center;
            }}
            .header h1 {{
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .header .subtitle {{
                color: #666;
                font-size: 1.1em;
            }}
            .timestamp {{
                color: #999;
                font-size: 0.9em;
                margin-top: 10px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .card h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .stat {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }}
            .stat:last-child {{
                border-bottom: none;
            }}
            .stat-label {{
                color: #666;
                font-weight: 500;
            }}
            .stat-value {{
                color: #333;
                font-weight: bold;
            }}
            .progress-bar {{
                width: 100%;
                height: 30px;
                background: #eee;
                border-radius: 15px;
                overflow: hidden;
                margin: 10px 0;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                transition: width 0.3s ease;
            }}
            .badge {{
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: bold;
                margin: 5px 5px 5px 0;
            }}
            .badge-success {{
                background: #10b981;
                color: white;
            }}
            .badge-info {{
                background: #3b82f6;
                color: white;
            }}
            .badge-warning {{
                background: #f59e0b;
                color: white;
            }}
            .tool-list {{
                list-style: none;
            }}
            .tool-item {{
                padding: 10px;
                margin: 5px 0;
                background: #f9fafb;
                border-radius: 5px;
                border-left: 3px solid #667eea;
            }}
            .tool-name {{
                font-weight: bold;
                color: #667eea;
            }}
            .tool-desc {{
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }}
            .refresh-btn {{
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                font-size: 1em;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }}
            .refresh-btn:hover {{
                background: #764ba2;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .status-healthy {{
                color: #10b981;
                font-weight: bold;
                font-size: 1.2em;
            }}
            .interact-form {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .form-group {{
                margin-bottom: 20px;
            }}
            .form-group label {{
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #667eea;
            }}
            .form-group input, .form-group textarea {{
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 1em;
                font-family: inherit;
                transition: border-color 0.3s;
            }}
            .form-group input:focus, .form-group textarea:focus {{
                outline: none;
                border-color: #667eea;
            }}
            .form-group textarea {{
                min-height: 100px;
                resize: vertical;
            }}
            .submit-btn {{
                background: #667eea;
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 5px;
                font-size: 1.1em;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                width: 100%;
            }}
            .submit-btn:hover {{
                background: #764ba2;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .submit-btn:disabled {{
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }}
            .response-message {{
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }}
            .response-message.success {{
                background: #d1fae5;
                color: #065f46;
                border: 1px solid #10b981;
            }}
            .response-message.error {{
                background: #fee2e2;
                color: #991b1b;
                border: 1px solid #ef4444;
            }}
            .reasoning-tabs {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }}
            .tab-btn {{
                padding: 10px 20px;
                border: 2px solid #667eea;
                background: white;
                color: #667eea;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.3s;
                font-weight: bold;
            }}
            .tab-btn.active {{
                background: #667eea;
                color: white;
            }}
            .tab-btn:hover {{
                background: #5a67d8;
                color: white;
            }}
            .reasoning-result {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
                border-left: 4px solid #667eea;
            }}
            .reasoning-step {{
                background: white;
                padding: 12px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 3px solid #10b981;
            }}
            .reasoning-step strong {{
                color: #667eea;
            }}
        </style>
        <script>
            function refreshDashboard() {{
                location.reload();
            }}

            // Auto-refresh cada 30 segundos
            setInterval(refreshDashboard, 30000);

            async function teachTHAU(event) {{
                event.preventDefault();

                const submitBtn = document.getElementById('submitBtn');
                const responseMsg = document.getElementById('responseMsg');
                const question = document.getElementById('question').value;
                const answer = document.getElementById('answer').value;
                const confidence = parseFloat(document.getElementById('confidence').value);

                // Deshabilitar botón
                submitBtn.disabled = true;
                submitBtn.textContent = '⏳ Enseñando a THAU...';
                responseMsg.style.display = 'none';

                try {{
                    const response = await fetch('/interact', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            question: question,
                            answer: answer,
                            confidence: confidence
                        }})
                    }});

                    const data = await response.json();

                    if (response.ok) {{
                        responseMsg.className = 'response-message success';
                        responseMsg.innerHTML = `
                            ✅ <strong>¡THAU ha aprendido!</strong><br>
                            📊 Vectores en memoria: ${{data.vectors_stored || 'N/A'}}<br>
                            🎯 Brecha detectada: ${{data.gap_detected ? 'Sí' : 'No'}}<br>
                            📈 Total interacciones: ${{data.total_interactions || 'N/A'}}
                        `;
                        responseMsg.style.display = 'block';

                        // Limpiar formulario
                        document.getElementById('interactForm').reset();

                        // Actualizar dashboard en 2 segundos
                        setTimeout(() => {{
                            location.reload();
                        }}, 2000);
                    }} else {{
                        throw new Error(data.detail || 'Error desconocido');
                    }}
                }} catch (error) {{
                    responseMsg.className = 'response-message error';
                    responseMsg.innerHTML = `❌ <strong>Error:</strong> ${{error.message}}`;
                    responseMsg.style.display = 'block';
                }} finally {{
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🎓 Enseñar a THAU';
                }}
            }}

            // Funciones de razonamiento
            function switchReasoningTab(tabName) {{
                // Remover active de todos los tabs
                document.querySelectorAll('.tab-btn').forEach(btn => {{
                    btn.classList.remove('active');
                }});

                // Activar el tab clickeado
                event.target.classList.add('active');

                // Mostrar el formulario correspondiente
                document.querySelectorAll('.reasoning-form').forEach(form => {{
                    form.style.display = 'none';
                }});
                document.getElementById(tabName + '-form').style.display = 'block';
            }}

            async function runChainOfThought() {{
                const question = document.getElementById('cot-question').value;
                const resultDiv = document.getElementById('cot-result');
                const btn = event.target;

                if (!question) return;

                btn.disabled = true;
                btn.textContent = '🧠 Razonando...';
                resultDiv.innerHTML = '<p>Procesando...</p>';

                try {{
                    const response = await fetch('/reasoning/chain-of-thought', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{question}})
                    }});

                    const data = await response.json();
                    let html = '<div class="reasoning-result">';
                    html += `<h4>🧠 Razonamiento Paso a Paso</h4>`;
                    html += `<p><strong>Pregunta:</strong> ${{data.question}}</p>`;

                    if (data.reasoning_steps) {{
                        data.reasoning_steps.forEach((step, i) => {{
                            html += `<div class="reasoning-step">
                                <strong>Paso ${{step.step}}:</strong> ${{step.thought}}
                            </div>`;
                        }});
                    }}

                    html += `<p><strong>Respuesta Final:</strong> ${{data.final_answer}}</p>`;
                    html += `<p><strong>Confianza:</strong> ${{(data.confidence * 100).toFixed(0)}}%</p>`;
                    html += '</div>';

                    resultDiv.innerHTML = html;
                }} catch (error) {{
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${{error.message}}</p>`;
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '🧠 Razonar Paso a Paso';
                }}
            }}

            async function runTreeOfThoughts() {{
                const question = document.getElementById('tot-question').value;
                const resultDiv = document.getElementById('tot-result');
                const btn = event.target;

                if (!question) return;

                btn.disabled = true;
                btn.textContent = '🌲 Explorando...';
                resultDiv.innerHTML = '<p>Explorando múltiples caminos...</p>';

                try {{
                    const response = await fetch('/reasoning/explore', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{question}})
                    }});

                    const data = await response.json();
                    let html = '<div class="reasoning-result">';
                    html += `<h4>🌲 Exploración de Pensamientos</h4>`;
                    html += `<p><strong>Pregunta:</strong> ${{data.question}}</p>`;
                    html += `<p><strong>Nodos explorados:</strong> ${{data.total_nodes}}</p>`;

                    if (data.best_path) {{
                        html += '<h5>Mejor camino encontrado:</h5>';
                        data.best_path.forEach((thought, i) => {{
                            html += `<div class="reasoning-step">${{thought}}</div>`;
                        }});
                    }}

                    html += `<p><strong>Puntuación:</strong> ${{data.best_score.toFixed(2)}}</p>`;
                    html += '</div>';

                    resultDiv.innerHTML = html;
                }} catch (error) {{
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${{error.message}}</p>`;
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '🌲 Explorar Opciones';
                }}
            }}

            async function runPlanner() {{
                const goal = document.getElementById('plan-goal').value;
                const resultDiv = document.getElementById('plan-result');
                const btn = event.target;

                if (!goal) return;

                btn.disabled = true;
                btn.textContent = '📋 Planificando...';
                resultDiv.innerHTML = '<p>Creando plan...</p>';

                try {{
                    const response = await fetch('/reasoning/plan', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{goal}})
                    }});

                    const data = await response.json();
                    let html = '<div class="reasoning-result">';
                    html += `<h4>📋 Plan de Acción</h4>`;
                    html += `<p><strong>Objetivo:</strong> ${{data.goal}}</p>`;
                    html += `<p><strong>Total de tareas:</strong> ${{data.total_tasks}}</p>`;
                    html += `<p><strong>Duración estimada:</strong> ${{data.estimated_duration}}</p>`;

                    if (data.tasks) {{
                        html += '<h5>Tareas:</h5>';
                        data.tasks.forEach(task => {{
                            const deps = task.dependencies.length > 0 ? ` (depende de: ${{task.dependencies.join(', ')}})` : '';
                            html += `<div class="reasoning-step">
                                <strong>Tarea ${{task.id}}:</strong> ${{task.description}}${{deps}}<br>
                                <small>Prioridad: ${{task.priority}} | Esfuerzo: ${{task.estimated_effort}}</small>
                            </div>`;
                        }});
                    }}

                    html += '</div>';

                    resultDiv.innerHTML = html;
                }} catch (error) {{
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${{error.message}}</p>`;
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '📋 Crear Plan';
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 THAU Dashboard</h1>
                <div class="subtitle">Trainable Helpful AI Unit - Sistema de Aprendizaje Autónomo</div>
                <div class="timestamp">📅 {dashboard_data['timestamp']}</div>
                <div class="timestamp status-healthy">✅ Sistema: {dashboard_data['system']['status'].upper()}</div>
                <button class="refresh-btn" onclick="refreshDashboard()">🔄 Actualizar</button>
            </div>

            <div class="grid">
                <!-- Desarrollo Cognitivo -->
                <div class="card">
                    <h2>🧠 Desarrollo Cognitivo</h2>
                    <div class="stat">
                        <span class="stat-label">Edad Cognitiva:</span>
                        <span class="stat-value">{dashboard_data['cognitive']['age']} años</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Etapa:</span>
                        <span class="stat-value">{dashboard_data['cognitive']['stage_name']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Descripción:</span>
                        <span class="stat-value">{dashboard_data['cognitive']['stage_description']}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {dashboard_data['cognitive']['progress_percentage']}%">
                            {dashboard_data['cognitive']['progress_percentage']:.1f}%
                        </div>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Ejemplos Actuales:</span>
                        <span class="stat-value">{dashboard_data['cognitive']['examples_in_current_age']}/{dashboard_data['cognitive']['examples_needed']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Puede Avanzar:</span>
                        <span class="stat-value">{'✅ Sí' if dashboard_data['cognitive']['can_advance'] else '❌ No'}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Progresiones de Edad:</span>
                        <span class="stat-value">{dashboard_data['cognitive']['total_age_progressions']}</span>
                    </div>
                </div>

                <!-- Memoria Vectorizada -->
                <div class="card">
                    <h2>💾 Memoria Vectorizada</h2>
                    <div class="stat">
                        <span class="stat-label">Vectores Totales:</span>
                        <span class="stat-value">{dashboard_data['memory']['total_vectors']:,}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Vectores Activos:</span>
                        <span class="stat-value">{dashboard_data['memory']['active_vectors']:,}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Vectores Eliminados:</span>
                        <span class="stat-value">{dashboard_data['memory']['deleted_vectors']:,}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Dimensión:</span>
                        <span class="stat-value">{dashboard_data['memory']['dimension']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Tipo de Índice:</span>
                        <span class="stat-value">{dashboard_data['memory']['index_type']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Tamaño en Memoria:</span>
                        <span class="stat-value">{dashboard_data['memory']['memory_size_mb']:.2f} MB</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Capacidad:</span>
                        <span class="stat-value">{dashboard_data['memory']['capacity']}</span>
                    </div>
                </div>

                <!-- Auto-Aprendizaje -->
                <div class="card">
                    <h2>📚 Auto-Aprendizaje</h2>
                    <div class="stat">
                        <span class="stat-label">Estado:</span>
                        <span class="stat-value">{'✅ Activo' if dashboard_data['auto_learning']['auto_improve_active'] else '❌ Inactivo'}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Brechas Detectadas:</span>
                        <span class="stat-value">{dashboard_data['auto_learning']['gaps_detected']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Datasets Generados:</span>
                        <span class="stat-value">{dashboard_data['auto_learning']['datasets_generated']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Ejemplos Generados:</span>
                        <span class="stat-value">{dashboard_data['auto_learning']['examples_generated']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Última Generación:</span>
                        <span class="stat-value">{dashboard_data['auto_learning']['last_generation']}</span>
                    </div>
                </div>

                <!-- Estadísticas Generales -->
                <div class="card">
                    <h2>📈 Estadísticas Generales</h2>
                    <div class="stat">
                        <span class="stat-label">Interacciones Totales:</span>
                        <span class="stat-value">{dashboard_data['statistics']['total_interactions']:,}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Entrenamientos Realizados:</span>
                        <span class="stat-value">{dashboard_data['statistics']['total_trainings']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Progresiones de Edad:</span>
                        <span class="stat-value">{dashboard_data['statistics']['age_progressions']}</span>
                    </div>
                </div>

                <!-- Idiomas -->
                <div class="card">
                    <h2>🌍 Idiomas</h2>
                    <div class="stat">
                        <span class="stat-label">Idiomas Activos:</span>
                        <span class="stat-value">{dashboard_data['languages']['total_active']}</span>
                    </div>
                    <div style="margin-top: 10px;">
                        {''.join([f'<span class="badge badge-info">{lang.upper()}</span>' for lang in dashboard_data['languages']['languages']])}
                    </div>
                </div>

                <!-- Modelo e Infraestructura -->
                <div class="card">
                    <h2>⚙️ Modelo e Infraestructura</h2>
                    <div class="stat">
                        <span class="stat-label">Modelo de Embeddings:</span>
                        <span class="stat-value">{dashboard_data['model_info']['embedding_model']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Dimensión Embeddings:</span>
                        <span class="stat-value">{dashboard_data['model_info']['embedding_dimension']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Modelo Base:</span>
                        <span class="stat-value">{dashboard_data['model_info']['base_model']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Dispositivo:</span>
                        <span class="stat-value">{dashboard_data['model_info']['device']}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Versión THAU:</span>
                        <span class="stat-value">{dashboard_data['system']['version']}</span>
                    </div>
                </div>

                <!-- Herramientas MCP -->
                <div class="card">
                    <h2>🔧 Herramientas MCP</h2>
                    <div class="stat">
                        <span class="stat-label">Total de Herramientas:</span>
                        <span class="stat-value">{dashboard_data['mcp']['total_tools']}</span>
                    </div>
                    <ul class="tool-list">
                        {''.join([
                            f'''<li class="tool-item">
                                <div class="tool-name">🔹 {tool['name']}</div>
                                <div class="tool-desc">{tool['description']}</div>
                            </li>'''
                            for tool in dashboard_data['mcp']['tools']
                        ])}
                    </ul>
                </div>
            </div>

            <!-- Formulario de Interacción -->
            <div class="interact-form">
                <h2 style="color: #667eea; margin-bottom: 20px;">💬 Interactuar con THAU</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    Enseña algo nuevo a THAU. Escribe una pregunta y su respuesta, y THAU lo almacenará en su memoria para aprender.
                </p>
                <form id="interactForm" onsubmit="teachTHAU(event)">
                    <div class="form-group">
                        <label for="question">❓ Pregunta:</label>
                        <input
                            type="text"
                            id="question"
                            name="question"
                            placeholder="Ejemplo: ¿Qué es Python?"
                            required
                        />
                    </div>
                    <div class="form-group">
                        <label for="answer">💡 Respuesta:</label>
                        <textarea
                            id="answer"
                            name="answer"
                            placeholder="Ejemplo: Python es un lenguaje de programación de alto nivel, interpretado y de propósito general..."
                            required
                        ></textarea>
                    </div>
                    <div class="form-group">
                        <label for="confidence">🎯 Confianza (0.0 - 1.0):</label>
                        <input
                            type="number"
                            id="confidence"
                            name="confidence"
                            min="0"
                            max="1"
                            step="0.1"
                            value="0.9"
                            required
                        />
                        <small style="color: #999; display: block; margin-top: 5px;">
                            0.0 = poca confianza, 1.0 = total confianza en la respuesta
                        </small>
                    </div>
                    <button type="submit" id="submitBtn" class="submit-btn">
                        🎓 Enseñar a THAU
                    </button>
                    <div id="responseMsg" class="response-message"></div>
                </form>
            </div>

            <!-- Sistema de Razonamiento Avanzado -->
            <div class="interact-form" style="margin-top: 30px;">
                <h2 style="color: #667eea; margin-bottom: 20px;">🧠 Razonamiento Avanzado</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    Prueba las capacidades de razonamiento de THAU: paso a paso, exploración de opciones, y planificación de tareas.
                </p>

                <!-- Tabs de Razonamiento -->
                <div class="reasoning-tabs">
                    <button class="tab-btn active" onclick="switchReasoningTab('cot')">🧠 Chain of Thought</button>
                    <button class="tab-btn" onclick="switchReasoningTab('tot')">🌲 Tree of Thoughts</button>
                    <button class="tab-btn" onclick="switchReasoningTab('plan')">📋 Planner</button>
                </div>

                <!-- Chain of Thought Form -->
                <div id="cot-form" class="reasoning-form">
                    <div class="form-group">
                        <label for="cot-question">❓ Pregunta para razonar:</label>
                        <input
                            type="text"
                            id="cot-question"
                            placeholder="Ejemplo: ¿Cómo puedo mejorar mi productividad?"
                        />
                    </div>
                    <button class="submit-btn" onclick="runChainOfThought()">
                        🧠 Razonar Paso a Paso
                    </button>
                    <div id="cot-result"></div>
                </div>

                <!-- Tree of Thoughts Form -->
                <div id="tot-form" class="reasoning-form" style="display: none;">
                    <div class="form-group">
                        <label for="tot-question">❓ Pregunta para explorar:</label>
                        <input
                            type="text"
                            id="tot-question"
                            placeholder="Ejemplo: ¿Cuál es la mejor estrategia para aprender programación?"
                        />
                    </div>
                    <button class="submit-btn" onclick="runTreeOfThoughts()">
                        🌲 Explorar Opciones
                    </button>
                    <div id="tot-result"></div>
                </div>

                <!-- Planner Form -->
                <div id="plan-form" class="reasoning-form" style="display: none;">
                    <div class="form-group">
                        <label for="plan-goal">🎯 Objetivo a planificar:</label>
                        <input
                            type="text"
                            id="plan-goal"
                            placeholder="Ejemplo: Crear una aplicación web para gestión de tareas"
                        />
                    </div>
                    <button class="submit-btn" onclick="runPlanner()">
                        📋 Crear Plan
                    </button>
                    <div id="plan-result"></div>
                </div>
            </div>

            <div class="header" style="margin-top: 20px;">
                <h3>📍 Endpoints Útiles</h3>
                <div style="margin-top: 15px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;">
                    <div class="stat">
                        <span class="stat-label">API Docs:</span>
                        <a href="/docs" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">/docs</a>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Dashboard JSON:</span>
                        <a href="/dashboard" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">/dashboard</a>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Status:</span>
                        <a href="/status" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">/status</a>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Health:</span>
                        <a href="/health" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">/health</a>
                    </div>
                </div>
                <div style="margin-top: 20px; color: #999; font-size: 0.9em;">
                    ⏰ Auto-refresh cada 30 segundos
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "thau_api_integrated:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Desactivar en producción
        log_level="info"
    )
