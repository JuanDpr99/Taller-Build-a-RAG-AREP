# Build-RAG: Sistema de Generación Aumentada por Recuperación

Un sistema RAG basado en Python que combina LangChain con modelos de lenguaje de OpenAI para recuperar y generar respuestas contextuales usando documentación obtenida de la web.

## Arquitectura del Proyecto

### Componentes

1. **Cargador de Documentos** - Obtiene y analiza contenido web usando BeautifulSoup
2. **Divisor de Texto** - Divide documentos en fragmentos manejables con solapamiento para preservar contexto
3. **Embeddings** - Convierte texto en representaciones vectoriales usando el modelo de embeddings de OpenAI
4. **Almacén Vectorial** - Almacenamiento en memoria para embeddings de documentos que permite búsqueda semántica
5. **Herramienta de Recuperación** - Herramienta personalizada que busca en el almacén vectorial contexto relevante
6. **Agente IA** - Agente de LangChain que orquesta consultas y llamadas de herramientas
7. **Middleware de Prompt Dinámico** - Inyecta contexto recuperado en las solicitudes del modelo

### Flujo de Arquitectura

```
Consulta del Usuario
    ↓
Agente IA con Herramientas
    ↓
Herramienta de Recuperación de Contexto
    ↓
Almacén Vectorial (Búsqueda de Similitud)
    ↓
Documentos Recuperados + Prompt Dinámico
    ↓
Modelo de Lenguaje OpenAI
    ↓
Respuesta Generada
```

## Instalación

### Requisitos Previos

- Python 3.13+
- Clave API de OpenAI

### Paso 1: Clonar y Navegar al Proyecto

```bash
cd Build-RAG
```

### Paso 2: Crear Entorno Virtual

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

Instala todos los paquetes requeridos ejecutando en orden:

```bash
pip install langchain langchain-text-splitters langchain-community bs4
pip install -U "langchain[openai]"
pip install -U "langchain-openai"
pip install -U "langchain-core"
```

### Paso 4: Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=sk-proj-tu_clave_aqui
```

## Instrucciones de Ejecución

Abre `Build-RAG.ipynb` en Jupyter Notebook o JupyterLab y ejecuta las celdas en orden:

### Celda 1-4: Instalar Dependencias

```python
pip install langchain langchain-text-splitters langchain-community bs4
pip install -U "langchain[openai]"
pip install -U "langchain-openai"
pip install -U "langchain-core"
```

**Propósito**: Instala todas las librerías necesarias para el sistema RAG.

### Celda 5: Inicializar Modelo

```python
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

model = init_chat_model("gpt-4.1")
```

**Propósito**: Configura el modelo de lenguaje GPT-4.1 de OpenAI.

### Celda 6: Configurar Embeddings

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

**Propósito**: Inicializa embeddings de OpenAI para convertir texto en vectores.

### Celda 7: Crear Almacén Vectorial

```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
```

**Propósito**: Crea un almacén vectorial en memoria para almacenar embeddings.

### Celda 8: Cargar Contenido Web

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
```

**Propósito**: Obtiene y analiza contenido web sobre agentes de IA del blog de Lilian Weng.

### Celda 9: Ver Contenido Cargado

```python
print(docs[0].page_content[:500])
```

**Propósito**: Muestra los primeros 500 caracteres del contenido cargado.

### Celda 10: Dividir Documentos

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")
```

**Propósito**: Divide el documento en fragmentos de 1000 caracteres con solapamiento de 200 caracteres para preservar contexto.

### Celda 11: Poblar Almacén Vectorial

```python
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
```

**Propósito**: Añade todos los fragmentos al almacén vectorial con embeddings calculados.

### Celda 12: Definir Herramienta de Recuperación

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

**Propósito**: Crea una herramienta personalizada que busca los 2 documentos más similares a una consulta.

### Celda 13: Crear Agente con Herramientas

```python
from langchain.agents import create_agent

tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)
```

**Propósito**: Inicializa un agente que puede usar la herramienta de recuperación.

### Celda 14: Ejecutar Consulta con Agente

```python
query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

**Propósito**: Ejecuta una consulta compleja que el agente responde buscando contexto relevante.

### Celda 15: Crear Agente con Middleware Dinámico

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

agent = create_agent(model, tools=[], middleware=[prompt_with_context])
```

**Propósito**: Crea un agente que inyecta dinámicamente contexto recuperado en cada solicitud.

### Celda 16: Ejecutar Consulta con Contexto Inyectado

```python
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

**Propósito**: Ejecuta una consulta simple con contexto automáticamente inyectado por el middleware.

## Características Principales

- **Carga de Contenido Web**: Obtiene y analiza automáticamente contenido HTML
- **Búsqueda Semántica**: Utiliza embeddings de OpenAI para recuperación inteligente de documentos
- **Arquitectura Basada en Agentes**: Aprovecha agentes de LangChain para manejo flexible de consultas
- **Inyección Dinámica de Contexto**: Enriquece automáticamente prompts con documentos recuperados relevantes
- **Respuestas en Tiempo Real**: Retorna respuestas del agente en streaming

## Opciones de Configuración

### Divisor de Texto
- `chunk_size`: 1000 (caracteres por fragmento)
- `chunk_overlap`: 200 (solapamiento de caracteres entre fragmentos)
- `add_start_index`: True (rastrear índice en documento original)

### Modelo de Embeddings
- `text-embedding-3-large` (modelo de embedding más reciente de OpenAI)

### Modelo LLM
- `gpt-4.1` (puedes cambiar a `gpt-4`, `gpt-3.5-turbo`, etc.)

### Búsqueda en Almacén Vectorial
- `k=2` (recupera los 2 documentos más similares)

## Solución de Problemas

### Problema: "Enter API key for OpenAI" (solicitud de clave API)

**Solución**: Asegúrate de que tu archivo `.env` contenga una `OPENAI_API_KEY` válida:

```env
OPENAI_API_KEY=sk-proj-tu_clave_aqui
```

### Problema: Error de conexión a la red (gaierror)

**Solución**: Verifica tu conexión a internet:

```powershell
ping google.com
```

Si estás detrás de un proxy, configura las variables de entorno:

```python
import os
os.environ['HTTP_PROXY'] = 'http://tu-proxy:puerto'
os.environ['HTTPS_PROXY'] = 'http://tu-proxy:puerto'
```

### Problema: Resultados de recuperación vacíos o irrelevantes

**Solución**: Ajusta estos parámetros:
- Aumenta `chunk_overlap` para mejor contexto
- Aumenta `k` en `similarity_search()` para más resultados
- Usa consultas más específicas y descriptivas

### Problema: API de OpenAI rechaza solicitudes

**Solución**: Verifica que:
- Tu clave API sea válida y activa
- Tu cuenta OpenAI tenga créditos disponibles
- No hayas excedido los límites de velocidad (rate limits)

## Dependencias

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `langchain` | Latest | Framework de agentes y orquestación |
| `langchain-text-splitters` | Latest | Utilidades de división de documentos |
| `langchain-community` | Latest | Cargador web e integraciones comunitarias |
| `langchain-openai` | Latest | Integraciones de modelos y embeddings de OpenAI |
| `langchain-core` | Latest | Implementaciones de almacén vectorial |
| `bs4` | Latest | Análisis de HTML para contenido web |

## Mejoras Futuras

- [ ] Soporte para almacenes vectoriales persistentes (Pinecone, Chroma, Milvus)
- [ ] Soporte para múltiples fuentes de documentos
- [ ] Modelos de embedding personalizados y fine-tuning
- [ ] Sistema de caché para mejor rendimiento
- [ ] RAG multi-documento con razonamiento entre documentos
- [ ] Interfaz web con Streamlit o Gradio
- [ ] Evaluación de calidad de respuestas (RAGAS)
- [ ] Soporte para múltiples idiomas

## Licencia

Este proyecto está bajo licencia MIT.

## Autor

Build-RAG - Sistema de Generación Aumentada por Recuperación

## Contacto y Soporte

Para reportar problemas o sugerencias, abre un issue en el repositorio.
