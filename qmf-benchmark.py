#!/usr/bin/env python3
"""
Benchmark: Quantum RAG vs Traditional Vector DB
Comparación de rendimiento y propiedades emergentes
"""

import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass

# Mock de un vector DB tradicional (ej: FAISS, Pinecone)
class TraditionalVectorDB:
    """Baseline: Vector DB tradicional con similaridad coseno"""
    
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []
    
    def add(self, text: str):
        """Añadir documento"""
        # Embedding simplificado (mismo que Quantum RAG)
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding /= (np.linalg.norm(embedding) + 1e-8)
        
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Búsqueda por similaridad coseno"""
        # Query embedding
        np.random.seed(hash(query) % (2**32))
        query_emb = np.random.randn(self.embedding_dim)
        query_emb /= (np.linalg.norm(query_emb) + 1e-8)
        
        # Calcular similaridades
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_emb, emb)
            similarities.append(sim)
        
        # Top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.documents[i], similarities[i]) for i in top_indices]
        
        return results


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark"""
    method: str
    ingest_time: float
    query_time: float
    num_documents: int
    accuracy: float  # Proporción de documentos relevantes recuperados
    
    def __str__(self):
        return (f"{self.method:20s} | "
                f"Ingest: {self.ingest_time:6.3f}s | "
                f"Query: {self.query_time:6.3f}s | "
                f"Acc: {self.accuracy:.2%}")


def evaluate_retrieval(results: List[str], ground_truth: List[str]) -> float:
    """
    Calcular precisión de retrieval
    
    Args:
        results: Documentos recuperados
        ground_truth: Documentos realmente relevantes
    
    Returns:
        Precisión (proporción de overlap)
    """
    results_set = set(results)
    truth_set = set(ground_truth)
    
    if not truth_set:
        return 0.0
    
    overlap = len(results_set & truth_set)
    return overlap / len(truth_set)


def run_benchmark(corpus: List[str], queries: List[Tuple[str, List[str]]],
                  field_size: int = 64) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Ejecutar benchmark completo
    
    Args:
        corpus: Lista de documentos
        queries: Lista de (query, ground_truth_docs)
        field_size: Tamaño del campo cuántico
    
    Returns:
        (resultado_quantum, resultado_traditional)
    """
    from quantum_rag import QuantumRAG
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {len(corpus)} documentos, {len(queries)} queries")
    print(f"{'='*70}\n")
    
    # ========== QUANTUM RAG ==========
    print("🌊 Testing Quantum RAG...")
    
    qrag = QuantumRAG(field_size=field_size, embedding_dim=128)
    
    # Ingest
    start = time.time()
    for doc in corpus:
        qrag.ingest(doc, amplitude=8.0, evolve_steps=10)
    ingest_time_q = time.time() - start
    
    # Query
    query_times_q = []
    accuracies_q = []
    
    for query, ground_truth in queries:
        start = time.time()
        results = qrag.retrieve(query, top_k=5, amplify_steps=20)
        query_time = time.time() - start
        query_times_q.append(query_time)
        
        result_texts = [r.text for r in results]
        acc = evaluate_retrieval(result_texts, ground_truth)
        accuracies_q.append(acc)
    
    result_q = BenchmarkResult(
        method="Quantum RAG",
        ingest_time=ingest_time_q,
        query_time=np.mean(query_times_q),
        num_documents=len(corpus),
        accuracy=np.mean(accuracies_q)
    )
    
    # ========== TRADITIONAL VECTOR DB ==========
    print("\n📊 Testing Traditional Vector DB...")
    
    vdb = TraditionalVectorDB(embedding_dim=128)
    
    # Ingest
    start = time.time()
    for doc in corpus:
        vdb.add(doc)
    ingest_time_t = time.time() - start
    
    # Query
    query_times_t = []
    accuracies_t = []
    
    for query, ground_truth in queries:
        start = time.time()
        results = vdb.search(query, top_k=5)
        query_time = time.time() - start
        query_times_t.append(query_time)
        
        result_texts = [r[0] for r in results]
        acc = evaluate_retrieval(result_texts, ground_truth)
        accuracies_t.append(acc)
    
    result_t = BenchmarkResult(
        method="Traditional VectorDB",
        ingest_time=ingest_time_t,
        query_time=np.mean(query_times_t),
        num_documents=len(corpus),
        accuracy=np.mean(accuracies_t)
    )
    
    return result_q, result_t


# ==================== DATASET DE PRUEBA ====================

CORPUS = [
    # Cluster 1: Quantum Physics
    "Quantum mechanics describes behavior at atomic scales",
    "The Schrödinger equation is fundamental to quantum theory",
    "Quantum superposition allows particles in multiple states",
    "Heisenberg uncertainty principle limits measurement precision",
    "Quantum entanglement shows non-local correlations",
    
    # Cluster 2: Machine Learning
    "Neural networks learn patterns from data",
    "Backpropagation trains deep learning models",
    "Convolutional networks excel at image recognition",
    "Recurrent networks process sequential information",
    "Transformers use attention mechanisms for NLP",
    
    # Cluster 3: Biology
    "DNA stores genetic information in cells",
    "Mitochondria generate energy in eukaryotes",
    "Photosynthesis converts light into chemical energy",
    "Protein folding determines biological function",
    "Evolution drives adaptation through natural selection",
    
    # Cluster 4: Astronomy
    "Black holes warp spacetime extremely",
    "The Big Bang started universe expansion",
    "Dark matter affects galactic rotation curves",
    "Neutron stars are incredibly dense objects",
    "Exoplanets orbit stars outside our solar system",
]

QUERIES = [
    # Query 1: Quantum
    (
        "quantum physics principles",
        [
            "Quantum mechanics describes behavior at atomic scales",
            "The Schrödinger equation is fundamental to quantum theory",
            "Quantum superposition allows particles in multiple states",
        ]
    ),
    
    # Query 2: Neural Networks
    (
        "deep learning and neural networks",
        [
            "Neural networks learn patterns from data",
            "Backpropagation trains deep learning models",
            "Convolutional networks excel at image recognition",
        ]
    ),
    
    # Query 3: Space
    (
        "astrophysics and cosmic phenomena",
        [
            "Black holes warp spacetime extremely",
            "The Big Bang started universe expansion",
            "Dark matter affects galactic rotation curves",
        ]
    ),
]


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n🔬 QUANTUM RAG vs TRADITIONAL VECTOR DB BENCHMARK")
    
    # Ejecutar benchmark
    result_quantum, result_traditional = run_benchmark(CORPUS, QUERIES)
    
    # Mostrar resultados
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}\n")
    print(result_quantum)
    print(result_traditional)
    
    # Análisis comparativo
    print(f"\n{'='*70}")
    print("ANALYSIS:")
    print(f"{'='*70}\n")
    
    speedup_ingest = result_traditional.ingest_time / result_quantum.ingest_time
    speedup_query = result_traditional.query_time / result_quantum.query_time
    
    print(f"Ingest Speedup:  {speedup_ingest:.2f}x {'⚡' if speedup_ingest > 1 else '🐌'}")
    print(f"Query Speedup:   {speedup_query:.2f}x {'⚡' if speedup_query > 1 else '🐌'}")
    
    acc_diff = result_quantum.accuracy - result_traditional.accuracy
    print(f"Accuracy Delta:  {acc_diff:+.1%} {'✅' if acc_diff > 0 else '⚠️'}")
    
    print("\n🔍 UNIQUE PROPERTIES OF QUANTUM RAG:")
    print("  • Interference: Related docs amplify each other")
    print("  • Temporal decay: Old memories naturally fade")
    print("  • Spatial clustering: Similar concepts group physically")
    print("  • Emergent retrieval: Context spreads through wave propagation")
    print("  • Uncertainty: Probabilistic rather than deterministic")
    
    print("\n💡 TRADE-OFFS:")
    if speedup_ingest < 1:
        print(f"  ⚠️  Ingest slower ({abs(speedup_ingest-1)*100:.0f}% overhead)")
        print("      → Physics simulation adds computational cost")
    
    if speedup_query < 1:
        print(f"  ⚠️  Query slower ({abs(speedup_query-1)*100:.0f}% overhead)")
        print("      → Resonance amplification requires evolution steps")
    else:
        print(f"  ✅ Query faster ({(speedup_query-1)*100:.0f}% speedup)")
    
    if acc_diff > 0:
        print(f"  ✅ Better accuracy (+{acc_diff*100:.1f}%)")
        print("      → Wave interference helps related docs")
    elif acc_diff < 0:
        print(f"  ⚠️  Lower accuracy ({acc_diff*100:.1f}%)")
        print("      → Might need parameter tuning")
    
    print("\n🎯 RECOMMENDATION:")
    if acc_diff > 0.1 or (speedup_query > 0.8 and acc_diff > 0):
        print("   → Quantum RAG shows promise for semantic-heavy tasks")
    else:
        print("   → Traditional VectorDB sufficient for simple similarity")
    
    print()