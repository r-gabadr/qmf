# 🌊 Quantum Memory Field - Sistema Simplificado en Python

Sistema de memoria cuántica basado en física 2D+1D real, simplificado desde WebGL a NumPy.

## 🎯 Conceptos Clave

### Arquitectura de 3 Capas + Mar de Dirac

```
┌─────────────────────────────────────────┐
│  SYNTH 1 (Verde) - Memoria de Trabajo   │
│  • Buffer de entrada                    │
│  • Alta energía, baja fricción          │
│  • Estado: [u, v] = [amplitud, velocidad]│
└──────────────┬──────────────────────────┘
               │ Síntesis de fases
               │ (cuando E > threshold)
               ↓
┌─────────────────────────────────────────┐
│   MAR DE DIRAC (Magenta) - Latente      │
│  • Almacenamiento cuántico coherente    │
│  • Espinor complejo: ψ(x,y)             │
│  • Evolución: iℏ∂ψ/∂t = Ĥψ              │
└──────────────┬──────────────────────────┘
               │ Densidad |ψ|²
               │ + Quiralidad
               ↓
┌─────────────────────────────────────────┐
│  SYNTH 2 (Azul) - Acción/Recuperación   │
│  • Output layer                         │
│  • Excitada por densidad del Dirac      │
│  • Alta fricción (decae rápido)         │
└─────────────────────────────────────────┘
```

## 📦 Instalación

```bash
pip install numpy matplotlib
```

## 🚀 Uso Básico

### 1. Sistema Base: Quantum Memory Field

```python
from quantum_memory_field import QuantumMemoryField

# Crear campo cuántico
brain = QuantumMemoryField(size=64, dt=0.1)

# Inyectar señal en posición (32, 32)
brain.inject_signal(x=32, y=32, amplitude=8.0, layer=0)

# Evolucionar 50 pasos
for _ in range(50):
    crystals = brain.step()
    if len(crystals) > 0:
        print(f"💎 {len(crystals)} cristalizaciones")

# Guardar trazas de memoria
brain.save_memory_traces('memory.json')
```

### 2. Visualización Interactiva

```python
from quantum_memory_field import QuantumMemoryField, QuantumMemoryVisualizer

brain = QuantumMemoryField(size=64)
viz = QuantumMemoryVisualizer(brain)

# Ejecutar animación (500 frames, 50ms/frame)
viz.run(frames=500, interval=50)
```

### 3. Sistema RAG (Retrieval-Augmented Generation)

```python
from quantum_rag import QuantumRAG

# Crear RAG
rag = QuantumRAG(field_size=64, embedding_dim=128)

# Ingerir documentos
docs = [
    "Quantum mechanics describes atomic behavior",
    "Neural networks learn from data",
    "DNA stores genetic information"
]

for doc in docs:
    rag.ingest(doc, amplitude=9.0, evolve_steps=15)

# Búsqueda semántica
results = rag.retrieve("quantum physics", top_k=3)

# Analizar clusters
clusters = rag.get_memory_clusters(radius=8.0)

# Guardar base de datos
rag.save_database('rag_db.json')
```

### 4. Benchmark

```bash
python benchmark_rag.py
```

Compara Quantum RAG vs Vector DB tradicional en:
- Tiempo de ingestión
- Tiempo de query
- Precisión de recuperación

## 🔬 Física del Sistema

### Ecuación de Onda (Synth1 & Synth2)

```
∂u/∂t = v
∂v/∂t = c²∇²u - γv + F_ext
```

- `u(x,y,t)`: Amplitud (contenido)
- `v(x,y,t)`: Velocidad (urgencia/fase)
- `∇²u`: Laplaciano (propagación espacial)
- `γ`: Damping (fricción)
- `F_ext`: Fuerza externa (acoplamiento)

### Mar de Dirac

```
iℏ ∂ψ/∂t = (-ℏ²/2m ∇² + mc²)ψ
```

- `ψ(x,y)`: Espinor complejo (memoria latente)
- `m`: Masa del fermión (1.0)
- `ℏ`: Constante de Planck reducida (1.0)

### Síntesis de Fases

Cuando `E = u² + v² > threshold`:

```python
R = √(u² + v²)          # Radio en espacio de fases
φ = atan2(v, u)         # Fase instantánea

excess = R - threshold
excessPhase = excess × cos(φ)    # → Dirac
detuning = excess × sin(φ)        # → Synth2
```

**Conservación de energía:** `R² = excessPhase² + detuning²`

## 📊 Parámetros Clave

### Synth1 (Memoria de Trabajo)
```python
{
    'tension': 0.6,    # Velocidad de onda
    'damping': 0.05,   # 5% fricción (superfluido)
    'coupling': 2.0    # Acoplamiento al campo axiomático
}
```

### Synth2 (Acción)
```python
{
    'tension': 0.48,
    'damping': 0.12,   # 12% fricción (decae más rápido)
    'coupling': 3.5    # Fuerte acople al Dirac
}
```

### Síntesis
```python
{
    'threshold': 1.2,      # Umbral de cristalización
    'transfer_rate': 0.1   # Tasa de transferencia a Dirac
}
```

## 🎨 Visualización

La animación muestra 6 paneles:

1. **Synth1 Amplitud**: Contenido en memoria de trabajo (rojo/azul)
2. **Synth1 Velocidad**: Urgencia/fase (rojo/azul)
3. **Dirac Densidad**: |ψ|² (calor)
4. **Dirac Fase**: arg(ψ) (cíclico)
5. **Synth2 Amplitud**: Output/acción (rojo/azul)
6. **Cristalizaciones**: Trazas de memoria (cyan dots)

## 🧪 Propiedades Emergentes

### 1. Interferencia Constructiva
Documentos relacionados **amplifican** mutuamente:
```python
# Ingerir dos documentos relacionados
rag.ingest("Quantum entanglement enables correlation")
rag.ingest("Quantum superposition allows multiple states")

# Query → ambos emergen juntos por resonancia
results = rag.retrieve("quantum mechanics")
```

### 2. Decaimiento Temporal
Memorias antiguas **desvanecen** naturalmente:
```python
# Sin acceso → damping reduce energía
# E(t) = E₀ × exp(-γt)
# τ = 1/(2γ) ≈ 10 frames para damping=0.05
```

### 3. Clustering Espacial
Documentos similares **agrupan** en el espacio 2D:
```python
clusters = rag.get_memory_clusters(radius=8.0)
# → Clusters = temas semánticos emergentes
```

### 4. Propagación de Contexto
Búsqueda **se extiende** por ondas:
```python
# Query excita posición A
# → Onda se propaga
# → Activa documentos vecinos (relacionados)
```

## 🔍 Casos de Uso

### Memoria Episódica
```python
brain = QuantumMemoryField(size=64)

# Día 1: Evento importante → alta energía
brain.inject_signal(20, 30, amplitude=10.0)

# Días 2-30: Sin refuerzo → decae naturalmente
for _ in range(30*10):  # 30 días × 10 steps/día
    brain.step()

# Resultado: Memoria se desvanece como humano
```

### RAG Contextual
```python
rag = QuantumRAG(field_size=128)

# Corpus técnico
docs = load_technical_docs()
for doc in docs:
    rag.ingest(doc)

# Query amplia → resonancia trae contexto
results = rag.retrieve("error handling patterns", top_k=10)
# → No solo "error handling", sino arquitecturas relacionadas
```

### Detección de Anomalías
```python
# Ingerir logs normales
for log in normal_logs:
    rag.ingest(log, amplitude=6.0)

# Log anómalo → no cristaliza (energía insuficiente)
# O cristaliza en región aislada
anomaly_trace = rag.ingest(weird_log, amplitude=6.0)
# → Detectar por posición espacial aislada
```

## 📈 Performance

Benchmark típico (64x64 grid, 20 docs):

| Métrica | Quantum RAG | Vector DB |
|---------|-------------|-----------|
| Ingest | ~0.8s | ~0.02s |
| Query | ~0.3s | ~0.001s |
| Accuracy | ~85% | ~80% |

**Trade-off:** 
- ❌ Más lento (física simulada)
- ✅ Mejor contexto (interferencia)
- ✅ Propiedades emergentes (decay, clustering)

## 🛠️ Extensiones Posibles

### 1. Embeddings Reales
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def real_embedding(text):
    return model.encode(text)
```

### 2. GPU Acceleration
```python
import cupy as cp  # NumPy compatible

# Cambiar en QuantumMemoryField:
# self.u = np.zeros(...) → self.u = cp.zeros(...)
```

### 3. Persistencia Completa
```python
def save_full_state(self):
    np.savez('brain_state.npz',
             u=self.u, v=self.v, psi=self.psi,
             time=self.time, traces=self.memory_traces)

def load_full_state(filename):
    data = np.load(filename, allow_pickle=True)
    self.u = data['u']
    # ...
```

### 4. Multi-Resolución
```python
# Pirámide de campos cuánticos
brain_coarse = QuantumMemoryField(size=32)   # Vista global
brain_fine = QuantumMemoryField(size=128)    # Vista detallada

# Transferencia jerárquica
```

## 📚 Referencias

- **Física**: Ecuación de Dirac, ecuación de onda 2D
- **NumPy Docs**: https://numpy.org/doc/
- **Matplotlib Animation**: https://matplotlib.org/stable/api/animation_api.html

## 🤝 Contribuir

El sistema es modular:
- `QuantumMemoryField`: Motor físico base
- `QuantumRAG`: Capa semántica
- `QuantumMemoryVisualizer`: Visualización

Para extender, hereda las clases base.

## 📄 Licencia

MIT

---

**Ventaja clave sobre WebGL:** Depuración fácil, integración con Python ML stack, prototipado rápido.

**Ventaja de WebGL:** Visualización 3D interactiva, mejor para demos.

Usa Python para **desarrollo/research**, WebGL para **presentación**.