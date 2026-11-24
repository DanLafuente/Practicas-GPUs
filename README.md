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
