# Practicas-GPUs

## Práctica 1: Asistente de Mantenimiento de Carril (`Practica 1/lane-assist`)

Implementación y paralelización en **CUDA** de un sistema de detección de líneas de carril (*Lane Assist*) sobre imágenes en formato PNG, comparando la ejecución secuencial en **CPU** frente a la aceleración en **GPU**.

### Descripción y Pipeline de Procesamiento

El programa recibe una imagen RGB en formato PNG, la transforma a escala de grises mediante luminancia ponderada (`0.2989·R + 0.5870·G + 0.1140·B`) y ejecuta las siguientes etapas para detectar las marcas viales y dibujarlas en rojo sobre la imagen de salida (`out.png`):

1. **Detector de bordes de Canny:**
   * **Reducción de ruido (`NR`):** Filtro gaussiano $5 \times 5$.
   * **Cálculo de gradientes (`Gx`, `Gy`, `G`, `phi`):** Operadores direccionales $5 \times 5$, magnitud del gradiente y cuantización de ángulo en $0^\circ, 45^\circ, 90^\circ$ y $135^\circ$.
   * **Supresión de no máximos (`pedge`):** Adelgazamiento de bordes según la dirección del gradiente.
   * **Umbralización por histéresis:** Clasificación de bordes fuertes y débiles conectados en vecindad $3 \times 3$ (`level = 1000.0f`).
2. **Transformada de Hough:**
   * Precomputación de tablas trigonométricas ($\sin\theta, \cos\theta$) para $\theta \in [0^\circ, 179^\circ]$.
   * Votación en el espacio de parámetros $(\rho, \theta)$. En la versión GPU, las actualizaciones concurrentes sobre el acumulador se resuelven mediante operaciones atómicas (`atomicAdd`).
3. **Extracción y trazado de líneas:**
   * Búsqueda de máximos locales en ventanas $9 \times 9$ del acumulador que superen un umbral dinámico ($\max(\text{width}, \text{height}) / 6$) y conversión a coordenadas cartesianas para dibujar los carriles detectados.


## Práctica 2 — Proyecto 1: Simulación Gravitatoria N-Cuerpos con SYCL (`Practica_Sycor`)

Simulación gravitatoria de $N$ cuerpos en 3D basada en la estructura de datos *Structure of Arrays* (**SoA**) y acelerada mediante **SYCL** sobre GPU[cite: 12, 16]. El código toma como base ejemplos del *Code Optimization Workshop* (Fabio Baruffa)[cite: 11, 12, 13, 14, 15, 16, 17].

### Descripción y Funcionamiento

La simulación inicializa un conjunto de partículas (por defecto `16000` partículas, `10` pasos temporales y `dt = 0.1`) con posiciones, velocidades y masas aleatorias (`mt19937` con semilla `42`) y aceleraciones iniciales en cero[cite: 12]. En cada paso temporal[cite: 12]:

1. **Cálculo de aceleraciones (`get_acceleration_SoA`):** Selecciona un dispositivo GPU mediante `sycl::gpu_selector_v`, crea buffers SYCL sobre los arreglos SoA y ejecuta un kernel en paralelo por grupos de trabajo (`parallel_for_work_group` / `parallel_for_work_item` en 2D con bloque `16x16`) calculando la interacción gravitatoria con suavizado (`softeningSquared = 1e-3f`, $G = 6.67259 \times 10^{-11}$)[cite: 12].
2. **Actualización de partículas y energía (`updateParticles` / `updateParticles_SoA`):** Actualiza velocidades y posiciones, reinicia las aceleraciones y acumula la energía cinética del sistema[cite: 12].
3. **Métricas de rendimiento:** Mide los tiempos de ejecución con `CPUTime` (`gettimeofday`) e informa paso a paso y al final el tiempo total y el rendimiento promedio en **GFlops**[cite: 11, 12].

## Práctica 2 — Proyecto 2: Eliminación de Ruido Sal y Pimienta con SYCL (`Sal&Piper`)

Implementación paralela en **SYCL** de un filtro de mediana adaptativo por umbral para eliminar ruido impulsivo (*sal y pimienta*) en imágenes BMP de 24 bits sin compresión[cite: 18, 22], con soporte de ejecución secuencial en host (`h`), paralela en CPU (`c`) y paralela en GPU (`g`).

### Descripción y Funcionamiento

El programa lee una imagen BMP de 24 bits[cite: 22], reserva memoria compartida (*Unified Shared Memory*, USM) con `malloc_shared`[cite: 22], la convierte a escala de grises (`0.2989·R + 0.5870·G + 0.1140·B`)[cite: 22] y aplica el filtrado espacial[cite: 18, 22]:

1. **Extracción de vecindad y ordenación:** Para cada píxel, recorre una ventana local de tamaño $3 \times 3$ (`window_size = 3`) y ordena los 9 valores mediante ordenamiento burbuja (*bubble sort*) para obtener la mediana[cite: 18, 22].
2. **Filtrado adaptativo por umbral:** Evalúa la diferencia relativa entre el píxel original y la mediana ($\frac{\vert{}\text{mediana} - I(i,j)\vert{}}{\text{mediana}} \le 0.1$)[cite: 18, 22]. Si está dentro del umbral, conserva el valor original para preservar detalles; en caso contrario, lo sustituye por la mediana[cite: 18, 22].
3. **Paralelización con SYCL (`remove_noise_SYCL`):** Distribuye el procesamiento de la imagen en grupos de trabajo 2D (`parallel_for_work_group` / `parallel_for_work_item`) sobre el dispositivo seleccionado en la cola `sycl::queue` (`cpu_selector_v` o `gpu_selector_v`)[cite: 18, 22].
4. **Salida y métricas:** Guarda la imagen filtrada en formato BMP[cite: 22] e imprime en consola el dispositivo empleado y el tiempo de ejecución[cite: 22].


### Estructura de Archivos

```text
Practica 1/lane-assist/
├── main.cpp          # Punto de entrada, gestión de argumentos, E/S y medición de tiempos
├── routinesCPU.c     # Implementación secuencial en C (Canny, Hough, escala de grises y dibujo)
├── routinesCPU.h     # Cabeceras de las rutinas de CPU
├── routinesGPU.cu    # Kernels CUDA para Canny y Transformada de Hough + gestión de memoria en GPU
├── png_io.c          # Lectura y escritura de imágenes PNG usando libpng
├── png_io.h          # Cabeceras de E/S de imágenes PNG
├── Makefile          # Compilación con g++, gcc y nvcc (enlazado con libpng y cudart)
└── Makefile2         # Variante de compilación delegando fases de compilación y enlazado a nvcc
Practica 2
├── Practica_Sycor/
    ├── main_2.cpp        # Punto de entrada y lectura de argumentos (N partículas, nsteps)
    ├── GSimulation.hpp   # Declaración de la clase GSimulation y parámetros de simulación
    ├── GSimulation.cpp   # Inicialización, kernels SYCL (get_acceleration_SoA, updateParticles_SoA) y bucle temporal
    ├── Particle.hpp      # Definición de las estructuras Particle y ParticleSoA
    ├── types.hpp         # Definición del tipo real_type (float)
    ├── cpu_time.hpp      # Utilidad CPUTime para medición de tiempos en microsegundos/segundos
    └── Makefile          # Compilación con icpx y soporte SYCL para objetivo nvptx64-nvidia-cuda
└── Sal&Piper/
    ├── main_3.cpp          # Lectura/escritura BMP, conversión RGB->BW, versión secuencial CPU y cola SYCL
    ├── kernels.hpp         # Cabecera de la función remove_noise_SYCL
    ├── kernels.cpp         # Kernel SYCL jerárquico (work-groups y work-items) para el filtro de mediana
    ├── Makefile            # Compilación con icpx y soporte SYCL para objetivo nvptx64-nvidia-cuda
    ├── LennaSALTPEPPER.bmp # Imagen de entrada de ejemplo con ruido sal y pimienta
    └── output.bmp          # Imagen resultante tras aplicar el filtro
