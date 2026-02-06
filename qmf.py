#!/usr/bin/env python3
"""
Quantum Memory Field: Mar de Dirac simplificado para memoria semántica emergente
Basado en física 2D+1D real con síntesis de fases
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json

@dataclass
class MemoryTrace:
    """Un cristal de memoria que emerge del sistema"""
    position: Tuple[int, int]
    energy: float
    phase: float
    timestamp: int
    content: str = ""  # Metadata semántica opcional

class QuantumMemoryField:
    def __init__(self, size=64, dt=0.1):
        """
        Sistema de memoria cuántica con 3 capas + Mar de Dirac
        
        Capas:
        0 - Synth1 (Memoria de Trabajo): Buffer de entrada, alta energía
        1 - Dirac (Latente/RAG): Almacenamiento cuántico coherente
        2 - Synth2 (Acción): Recuperación y output
        
        Args:
            size: Resolución de la malla espacial (64x64 = 4096 células)
            dt: Paso temporal (0.1 por defecto, estable hasta 0.2)
        """
        self.N = size
        self.dt = dt
        self.time = 0
        
        # Estado de las capas bosónicas (Synth 1 & 2)
        self.u = np.zeros((3, size, size))  # Amplitud (contenido)
        self.v = np.zeros((3, size, size))  # Velocidad (urgencia/fase)
        
        # Mar de Dirac (capa fermiónica central)
        self.psi = np.zeros((size, size), dtype=np.complex128)  # Espinor complejo
        
        # Parámetros físicos
        self.params = {
            'synth1': {'tension': 0.6, 'damping': 0.05, 'coupling': 2.0},
            'synth2': {'tension': 0.48, 'damping': 0.12, 'coupling': 3.5},
            'dirac': {'mass': 1.0, 'hbar': 1.0},
            'synthesis': {'threshold': 1.2, 'transfer_rate': 0.1}
        }
        
        # Memoria de trazas cristalizadas
        self.memory_traces: List[MemoryTrace] = []
        
    def laplacian(self, field):
        """
        Operador Laplaciano 2D con condiciones de frontera periódicas
        ∇²f ≈ (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{i,j})
        """
        return (-4 * field + 
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))
    
    def inject_signal(self, x, y, amplitude=5.0, velocity=0.0, layer=0):
        """
        Inyectar señal en una capa específica
        
        Args:
            x, y: Coordenadas espaciales
            amplitude: Energía inicial (default: 5.0)
            velocity: Velocidad inicial (default: 0.0, inyecta solo momento)
            layer: Capa objetivo (0=Synth1, 2=Synth2)
        """
        if layer == 1:
            # Para Dirac, inyectamos directamente en el espinor
            self.psi[x, y] += amplitude * np.exp(1j * velocity)
        else:
            # Para capas bosónicas, inyectamos velocidad (crea onda propagante)
            self.v[layer, x, y] += amplitude
            self.u[layer, x, y] += velocity
    
    def synthesize_phases(self):
        """
        Síntesis de fases: Detector de umbrales + transferencia de energía
        
        Cuando E = u² + v² > threshold en Synth1:
        1. Descompone en (excessPhase, detuning)
        2. Transfiere excessPhase → Dirac (cristalización)
        3. Conserva energía total
        
        Returns:
            Matriz booleana indicando células que cristalizaron
        """
        # Energía local en Synth1 (espacio de fases)
        energy_L0 = self.u[0]**2 + self.v[0]**2
        
        # Detectar excesos de energía
        threshold = self.params['synthesis']['threshold']
        mask = energy_L0 > threshold
        
        if not np.any(mask):
            return mask
        
        # Calcular exceso
        radius = np.sqrt(energy_L0)
        excess = np.where(mask, radius - threshold, 0)
        
        # Fase instantánea en espacio de fases
        phase = np.arctan2(self.v[0], self.u[0])
        
        # Descomposición (conserva energía)
        excess_phase = excess * np.cos(phase) * 0.8
        detuning = (threshold + excess * np.sin(phase)) * 0.6
        
        # Transferencia a Dirac (acoplamiento real→complejo)
        transfer_rate = self.params['synthesis']['transfer_rate']
        self.psi += (excess_phase + 1j * detuning) * transfer_rate
        
        # Drenar energía de Synth1 (evita acumulación infinita)
        drain_factor = 0.7
        self.u[0] = np.where(mask, self.u[0] * drain_factor, self.u[0])
        self.v[0] = np.where(mask, self.v[0] * drain_factor, self.v[0])
        
        return mask
    
    def evolve_dirac(self):
        """
        Evolución del Mar de Dirac (ecuación de Schrödinger 2D)
        
        iℏ ∂ψ/∂t = (-ℏ²/2m ∇² + mc²)ψ
        
        En unidades naturales (ℏ=c=1):
        ∂ψ/∂t = -i(∇²/2m + m)ψ
        """
        mass = self.params['dirac']['mass']
        
        # Hamiltoniano: Ĥ = -∇²/2m + m
        lap_psi = self.laplacian(self.psi)
        H_psi = -lap_psi / (2 * mass) + mass * self.psi
        
        # Evolución unitaria: ψ(t+dt) = ψ(t) - i·Ĥ·ψ·dt
        self.psi += -1j * H_psi * self.dt
        
        # Normalización suave (evita explosiones numéricas)
        norm = np.abs(self.psi)
        max_norm = 10.0
        self.psi = np.where(norm > max_norm, 
                           self.psi * (max_norm / (norm + 1e-6)),
                           self.psi)
    
    def step(self, input_signal_pos: Optional[Tuple[int, int]] = None,
             input_amplitude: float = 5.0):
        """
        Un paso de evolución temporal completo
        
        Flujo:
        1. INPUT → Synth1 (si hay señal)
        2. Dinámica de ondas en Synth1 & Synth2
        3. Síntesis de fases: Synth1 → Dirac
        4. Evolución cuántica del Dirac
        5. Retroalimentación: Dirac → Synth2
        6. Detección de cristalizaciones
        
        Args:
            input_signal_pos: (x, y) coordenadas de input, o None
            input_amplitude: Energía de la señal (default: 5.0)
        
        Returns:
            Lista de coordenadas donde ocurrió cristalización
        """
        # 1. INYECCIÓN DE SEÑAL
        if input_signal_pos is not None:
            x, y = input_signal_pos
            self.inject_signal(x, y, amplitude=input_amplitude, layer=0)
        
        # 2. DINÁMICA DE ONDAS (Synth1 & Synth2)
        for layer_idx, layer_name in [(0, 'synth1'), (2, 'synth2')]:
            p = self.params[layer_name]
            
            # Laplaciano espacial
            lap_u = self.laplacian(self.u[layer_idx])
            
            # Ecuación de onda: dv/dt = c²∇²u - γv + F_ext
            acceleration = (p['tension'] * lap_u - 
                          p['damping'] * self.v[layer_idx])
            
            # Integración semi-implícita
            self.v[layer_idx] += acceleration * self.dt
            self.u[layer_idx] += self.v[layer_idx] * self.dt
        
        # 3. SÍNTESIS DE FASES (Synth1 → Dirac)
        crystallization_mask = self.synthesize_phases()
        
        # 4. EVOLUCIÓN DEL MAR DE DIRAC
        self.evolve_dirac()
        
        # 5. RETROALIMENTACIÓN (Dirac → Synth2)
        # La densidad del espinor "empuja" la capa de acción
        dirac_density = np.abs(self.psi)**2
        coupling = self.params['synth2']['coupling']
        self.v[2] += dirac_density * coupling * self.dt
        
        # 6. ALMACENAR TRAZAS CRISTALIZADAS
        crystals = np.argwhere(crystallization_mask)
        for x, y in crystals:
            energy = self.u[0, x, y]**2 + self.v[0, x, y]**2
            phase = np.angle(self.psi[x, y])
            
            trace = MemoryTrace(
                position=(int(x), int(y)),
                energy=float(energy),
                phase=float(phase),
                timestamp=self.time
            )
            self.memory_traces.append(trace)
        
        self.time += 1
        return crystals
    
    def get_state_snapshot(self):
        """Snapshot completo del estado para visualización"""
        return {
            'synth1_amplitude': self.u[0].copy(),
            'synth1_velocity': self.v[0].copy(),
            'dirac_real': np.real(self.psi).copy(),
            'dirac_imag': np.imag(self.psi).copy(),
            'dirac_density': np.abs(self.psi)**2,
            'dirac_phase': np.angle(self.psi),
            'synth2_amplitude': self.u[2].copy(),
            'synth2_velocity': self.v[2].copy(),
            'time': self.time
        }
    
    def save_memory_traces(self, filename='memory_traces.json'):
        """Guardar trazas de memoria a JSON"""
        traces_dict = [
            {
                'position': t.position,
                'energy': t.energy,
                'phase': t.phase,
                'timestamp': t.timestamp,
                'content': t.content
            }
            for t in self.memory_traces
        ]
        
        with open(filename, 'w') as f:
            json.dump(traces_dict, f, indent=2)
        
        print(f"💾 Guardadas {len(traces_dict)} trazas en {filename}")


class QuantumMemoryVisualizer:
    """Visualizador interactivo en tiempo real"""
    
    def __init__(self, field: QuantumMemoryField):
        self.field = field
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Quantum Memory Field - Sistema 2D+1D', fontsize=16)
        
        # Configurar subplots
        titles = [
            'Synth1: Amplitud (Trabajo)',
            'Synth1: Velocidad (Urgencia)',
            'Dirac: Densidad |ψ|²',
            'Dirac: Fase arg(ψ)',
            'Synth2: Amplitud (Acción)',
            'Memoria: Cristalizaciones'
        ]
        
        for ax, title in zip(self.axes.flat, titles):
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.images = []
        self.memory_scatter = None
        
    def init_animation(self):
        """Inicializar imágenes para animación"""
        N = self.field.N
        
        # Colormaps
        cmaps = ['RdBu_r', 'seismic', 'hot', 'twilight', 'RdBu_r', 'viridis']
        
        for ax, cmap in zip(self.axes.flat[:-1], cmaps):
            img = ax.imshow(np.zeros((N, N)), cmap=cmap, 
                           interpolation='bilinear', animated=True)
            self.images.append(img)
            plt.colorbar(img, ax=ax, fraction=0.046)
        
        # Scatter para memoria
        self.memory_scatter = self.axes.flat[-1].scatter(
            [], [], c='cyan', s=20, alpha=0.6
        )
        self.axes.flat[-1].set_xlim(0, N)
        self.axes.flat[-1].set_ylim(0, N)
        
        return self.images + [self.memory_scatter]
    
    def update(self, frame):
        """Actualizar visualización (llamado cada frame)"""
        # Inyectar señal aleatoria ocasionalmente
        if np.random.random() < 0.1:  # 10% de probabilidad
            x, y = np.random.randint(10, self.field.N-10, size=2)
            self.field.step(input_signal_pos=(x, y), input_amplitude=6.0)
        else:
            self.field.step()
        
        # Obtener snapshot
        state = self.field.get_state_snapshot()
        
        # Actualizar imágenes
        data = [
            state['synth1_amplitude'],
            state['synth1_velocity'],
            state['dirac_density'],
            state['dirac_phase'],
            state['synth2_amplitude']
        ]
        
        for img, d in zip(self.images, data):
            img.set_data(d)
            img.set_clim(d.min(), d.max())
        
        # Actualizar memoria
        if self.field.memory_traces:
            positions = np.array([t.position for t in self.field.memory_traces[-100:]])
            if len(positions) > 0:
                self.memory_scatter.set_offsets(positions[:, ::-1])  # Invertir x,y
        
        self.fig.suptitle(
            f'Quantum Memory Field - t={self.field.time} | '
            f'Trazas: {len(self.field.memory_traces)}',
            fontsize=16
        )
        
        return self.images + [self.memory_scatter]
    
    def run(self, frames=500, interval=50):
        """Ejecutar animación"""
        anim = FuncAnimation(
            self.fig, self.update, init_func=self.init_animation,
            frames=frames, interval=interval, blit=True
        )
        plt.tight_layout()
        plt.show()
        return anim


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    print("🌊 Inicializando Quantum Memory Field...")
    
    # Crear sistema
    brain = QuantumMemoryField(size=64, dt=0.1)
    
    # Opción 1: Simulación batch (sin visualización)
    print("\n📊 Simulación batch (100 pasos)...")
    for step in range(100):
        # Inyectar señales periódicas
        if step % 10 == 0:
            x, y = np.random.randint(10, 54, size=2)
            crystals = brain.step(input_signal_pos=(x, y), input_amplitude=8.0)
            if len(crystals) > 0:
                print(f"  [t={step}] 💎 {len(crystals)} cristalizaciones")
        else:
            brain.step()
    
    print(f"\n✅ Total de trazas almacenadas: {len(brain.memory_traces)}")
    brain.save_memory_traces()
    
    # Opción 2: Visualización interactiva (descomentar para usar)
    # print("\n🎨 Iniciando visualización...")
    # viz = QuantumMemoryVisualizer(brain)
    # viz.run(frames=500, interval=50)