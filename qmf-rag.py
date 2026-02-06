#!/usr/bin/env python3
"""
Quantum RAG: Sistema de Retrieval-Augmented Generation usando física cuántica
Memoria semántica emergente del Mar de Dirac
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

# Importar el sistema base
from quantum_memory_field import QuantumMemoryField, MemoryTrace

@dataclass
class SemanticMemory:
    """Memoria semántica con embedding vectorial"""
    text: str
    embedding: np.ndarray  # Vector de características
    position: tuple  # Coordenadas en el campo cuántico
    energy: float
    phase: float
    timestamp: int
    
    def to_dict(self):
        return {
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'position': self.position,
            'energy': self.energy,
            'phase': self.phase,
            'timestamp': self.timestamp
        }


class QuantumRAG:
    """
    Sistema RAG basado en física cuántica
    
    Conceptos clave:
    - Cada documento/chunk → señal espacial en Synth1
    - Cristalización → almacenamiento en Dirac (memoria latente)
    - Retrieval → excitación del Dirac por similaridad → Synth2
    - Interferencia constructiva → documentos relacionados emergen juntos
    """
    
    def __init__(self, field_size=64, embedding_dim=128):
        """
        Args:
            field_size: Tamaño del campo cuántico (64x64 = 4096 slots)
            embedding_dim: Dimensión del espacio de embeddings
        """
        self.field = QuantumMemoryField(size=field_size)
        self.embedding_dim = embedding_dim
        
        # Base de datos de memoria semántica
        self.semantic_db: List[SemanticMemory] = []
        
        # Mapa de posiciones ocupadas
        self.occupied_positions = set()
        
    def _simple_embedding(self, text: str) -> np.ndarray:
        """
        Embedding simplificado (para demo - usar BERT/OpenAI en producción)
        
        Genera vector basado en:
        - Hash de palabras → proyección en espacio aleatorio
        - Normalización L2
        """
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding /= (np.linalg.norm(embedding) + 1e-8)
        return embedding
    
    def _embedding_to_position(self, embedding: np.ndarray) -> tuple:
        """
        Proyectar embedding de alta dimensión a coordenadas 2D
        
        Usa PCA simplificado: toma las 2 primeras componentes principales
        y mapea a la malla [0, field_size)
        """
        # Proyección simple: suma ponderada de componentes
        x_weights = np.cos(np.linspace(0, np.pi, self.embedding_dim))
        y_weights = np.sin(np.linspace(0, np.pi, self.embedding_dim))
        
        x_proj = np.dot(embedding, x_weights)
        y_proj = np.dot(embedding, y_weights)
        
        # Normalizar a [0, 1] y escalar al tamaño del campo
        x = int((x_proj + 1) / 2 * (self.field.N - 10) + 5)
        y = int((y_proj + 1) / 2 * (self.field.N - 10) + 5)
        
        # Evitar colisiones
        while (x, y) in self.occupied_positions:
            x = (x + 1) % self.field.N
            y = (y + 1) % self.field.N
        
        self.occupied_positions.add((x, y))
        return (x, y)
    
    def ingest(self, text: str, amplitude: float = 8.0, 
               evolve_steps: int = 20) -> SemanticMemory:
        """
        Ingerir un documento en la memoria cuántica
        
        Proceso:
        1. Generar embedding del texto
        2. Proyectar a posición 2D
        3. Inyectar señal en Synth1
        4. Evolucionar campo → cristalización en Dirac
        5. Guardar traza semántica
        
        Args:
            text: Contenido a memorizar
            amplitude: Energía de inyección (>6.0 para forzar cristalización)
            evolve_steps: Pasos de evolución después de inyección
        
        Returns:
            SemanticMemory creada
        """
        # 1. Embedding
        embedding = self._simple_embedding(text)
        
        # 2. Posición espacial
        x, y = self._embedding_to_position(embedding)
        
        # 3. Inyectar señal
        self.field.inject_signal(x, y, amplitude=amplitude, layer=0)
        
        # 4. Evolucionar hasta cristalización
        for _ in range(evolve_steps):
            crystals = self.field.step()
            
        # 5. Recuperar estado del Dirac en esa posición
        energy = np.abs(self.field.psi[x, y])**2
        phase = np.angle(self.field.psi[x, y])
        
        # 6. Crear memoria semántica
        memory = SemanticMemory(
            text=text,
            embedding=embedding,
            position=(x, y),
            energy=float(energy),
            phase=float(phase),
            timestamp=self.field.time
        )
        
        self.semantic_db.append(memory)
        
        print(f"📝 Ingested: '{text[:50]}...' → pos({x},{y}) E={energy:.3f}")
        
        return memory
    
    def retrieve(self, query: str, top_k: int = 5, 
                 amplify_steps: int = 30) -> List[SemanticMemory]:
        """
        Recuperar documentos relevantes usando resonancia cuántica
        
        Proceso:
        1. Embedding del query
        2. Excitar Dirac en posiciones similares
        3. Evolucionar → interferencia constructiva amplifica relacionados
        4. Leer densidad en Synth2 (acción)
        5. Ranking por energía emergente
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados
            amplify_steps: Pasos de evolución para amplificación
        
        Returns:
            Lista de SemanticMemory ordenadas por relevancia
        """
        if not self.semantic_db:
            return []
        
        # 1. Embedding del query
        query_embedding = self._simple_embedding(query)
        
        # 2. Calcular similaridades (producto punto)
        similarities = np.array([
            np.dot(query_embedding, mem.embedding) 
            for mem in self.semantic_db
        ])
        
        # 3. Excitar posiciones similares en el Dirac
        # Amplitud proporcional a similaridad
        for mem, sim in zip(self.semantic_db, similarities):
            if sim > 0.3:  # Umbral de relevancia
                x, y = mem.position
                # Inyectar directamente en Dirac (no en Synth1)
                self.field.psi[x, y] += sim * 3.0 * np.exp(1j * mem.phase)
        
        # 4. Evolucionar → resonancia
        for _ in range(amplify_steps):
            self.field.step()
        
        # 5. Leer energía emergente en Synth2
        retrieval_scores = []
        for mem in self.semantic_db:
            x, y = mem.position
            # Score = amplitud en Synth2 + densidad Dirac
            score = (np.abs(self.field.u[2, x, y]) + 
                    np.abs(self.field.psi[x, y])**2)
            retrieval_scores.append(score)
        
        # 6. Ranking
        top_indices = np.argsort(retrieval_scores)[-top_k:][::-1]
        results = [self.semantic_db[i] for i in top_indices]
        
        print(f"\n🔍 Query: '{query}'")
        for i, mem in enumerate(results, 1):
            score = retrieval_scores[top_indices[i-1]]
            print(f"  {i}. [{score:.3f}] {mem.text[:60]}...")
        
        return results
    
    def save_database(self, filename='quantum_rag_db.json'):
        """Guardar base de datos semántica"""
        db_dict = {
            'memories': [mem.to_dict() for mem in self.semantic_db],
            'field_state': {
                'time': self.field.time,
                'num_traces': len(self.field.memory_traces)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(db_dict, f, indent=2)
        
        print(f"💾 Database saved: {len(self.semantic_db)} memories → {filename}")
    
    def get_memory_clusters(self, radius: float = 5.0) -> List[List[SemanticMemory]]:
        """
        Detectar clusters de memoria por proximidad espacial
        
        Memorias cercanas espacialmente → semánticamente relacionadas
        (gracias a la proyección embedding→posición)
        
        Args:
            radius: Radio de clustering (en celdas)
        
        Returns:
            Lista de clusters (cada cluster = lista de memorias)
        """
        if not self.semantic_db:
            return []
        
        # Matriz de distancias
        positions = np.array([mem.position for mem in self.semantic_db])
        
        clusters = []
        visited = set()
        
        for i, mem in enumerate(self.semantic_db):
            if i in visited:
                continue
            
            # Encontrar vecinos dentro del radio
            distances = np.linalg.norm(positions - positions[i], axis=1)
            neighbors = np.where(distances < radius)[0]
            
            cluster = [self.semantic_db[j] for j in neighbors]
            clusters.append(cluster)
            
            visited.update(neighbors)
        
        # Ordenar clusters por tamaño
        clusters.sort(key=len, reverse=True)
        
        return clusters


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    print("🧠 Quantum RAG System - Demo\n")
    
    # Corpus de ejemplo (documentos técnicos)
    documents = [
        "Quantum computing uses qubits to perform parallel computations",
        "Machine learning models require large datasets for training",
        "Neural networks are inspired by biological brain structure",
        "The Dirac equation describes relativistic quantum mechanics",
        "GPT models use transformer architecture for language processing",
        "Quantum entanglement enables faster-than-light communication",  # Falso pero para demo
        "Backpropagation is the algorithm used to train deep networks",
        "The Schrödinger equation governs non-relativistic quantum systems",
        "Convolutional neural networks excel at image recognition",
        "Quantum superposition allows particles to exist in multiple states",
        "Reinforcement learning trains agents through reward signals",
        "The uncertainty principle limits simultaneous measurement precision",
        "Recurrent neural networks process sequential data effectively",
        "Wave-particle duality is a fundamental quantum phenomenon",
        "Transfer learning reuses pretrained models for new tasks"
    ]
    
    # 1. Crear sistema RAG
    rag = QuantumRAG(field_size=64, embedding_dim=128)
    
    # 2. Ingerir documentos
    print("📚 Ingesting documents...\n")
    for doc in documents:
        rag.ingest(doc, amplitude=9.0, evolve_steps=15)
    
    print(f"\n✅ {len(rag.semantic_db)} documents in memory\n")
    print("="*70)
    
    # 3. Queries de prueba
    queries = [
        "quantum mechanics and physics",
        "neural networks and deep learning",
        "how quantum computers work"
    ]
    
    for query in queries:
        print()
        results = rag.retrieve(query, top_k=3, amplify_steps=25)
        print()
    
    print("="*70)
    
    # 4. Análisis de clusters
    print("\n📊 Memory Clusters (spatial proximity):\n")
    clusters = rag.get_memory_clusters(radius=8.0)
    
    for i, cluster in enumerate(clusters[:3], 1):  # Top 3 clusters
        print(f"Cluster {i} ({len(cluster)} documents):")
        for mem in cluster:
            print(f"  - {mem.text[:60]}...")
        print()
    
    # 5. Guardar base de datos
    rag.save_database()
    
    print("\n🎉 Demo completed!")