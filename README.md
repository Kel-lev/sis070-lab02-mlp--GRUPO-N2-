# SIS070 - Laboratorio 04: El Perceptrón Multicapa (MLP) y Backpropagation desde Cero

## Datos generales

- **Estudiantes:** Leva Ayte, Kelma Ivonne / Delgado Ccorihuaman, Hamlet Nayeli
- **Asignatura:** Inteligencia Artificial
- **Tema:** Perceptrón Multicapa (MLP) y Backpropagation
- **Docente:** Lazo Mamani, Juan Carlos

## Descripción

En este laboratorio se implementó desde cero un Perceptrón Multicapa (MLP) utilizando únicamente Python y NumPy, sin emplear librerías especializadas de aprendizaje automático.

El objetivo principal fue comprender de manera práctica cómo funciona una red neuronal multicapa, implementando manualmente los procesos de:

- Propagación hacia adelante (forward propagation).
- Retropropagación del error (backpropagation).
- Cálculo de gradientes mediante la regla de la cadena.
- Actualización de pesos y sesgos mediante descenso de gradiente.
- Uso de funciones de activación Sigmoide y ReLU.
- Evaluación del efecto de diferentes valores de learning rate.
- Comparación entre una arquitectura con una capa oculta y otra con dos capas ocultas.

Para comprobar el funcionamiento de la implementación se utilizó el problema clásico XOR, debido a que sus datos no son linealmente separables y, por lo tanto, requieren una arquitectura con al menos una capa oculta y una función de activación no lineal.

---

## Estructura del Proyecto

```
sis070-lab02-mlp-leva-delgado/
│
├── src/
│   └── mlp_implementation.py    # Implementación completa del MLP
│
├── README.md                    # Documentación y análisis
│
└── requirements.txt             # Dependencias del proyecto
└── img1.png                     # Resultados
└── img2.png                     # Resultados
└── img3.png                     # Resultados
```

---

## Requisitos e Instalación

### Requisitos

- Python 3.8 o superior.
- NumPy.

### Instalación

Primero se instalan las dependencias:

```bash
pip install -r requirements.txt
```

Finalmente, se ejecuta el programa:

```bash
python src/mlp_implementation.py
```

---

## Explicación de la Implementación

### 1. Funciones de activación

En la implementación se utilizaron dos funciones de activación: ReLU y Sigmoide. Cada una cuenta con su respectiva derivada, esto es necesario durante el proceso de retropropagación.

#### ReLU

La función ReLU está definida como:

```
ReLU(z) = max(0, z)
```

Su derivada es:

```
ReLU'(z) = 1, si z > 0
           0, si z ≤ 0
```

Esta función permite introducir no linealidad en la red y para valores positivos mantiene un gradiente constante.

#### Sigmoide

La función sigmoide está definida como:

```
σ(z) = 1 / (1 + e^(-z))
```

Su derivada puede calcularse mediante:

```
σ'(z) = σ(z) * (1 - σ(z))
```

La sigmoide produce valores entre 0 y 1, por lo que resulta apropiada para la capa de salida en este problema de clasificación binaria.

---

### 2. Clase SimpleMLP

La primera implementación corresponde a una red neuronal con la siguiente arquitectura:

```
Entrada (2) → Capa oculta (4) → Salida (1)
```

La red recibe las dos variables de entrada del problema XOR y utiliza cuatro neuronas en la capa oculta.

#### Método forward(X)

El método realiza la **propagación hacia adelante**, es decir, el proceso mediante el cual los datos pasan desde la entrada hasta la salida de la red neuronal.


Primero se calcula:
Donde:

X: contiene los datos de entrada de la red.

W1: contiene los pesos que conectan la capa de entrada con la capa oculta.

b1: contiene los sesgos de las neuronas de la capa oculta.

Z1: representa la combinación lineal obtenida antes de aplicar la función

```
Z1 = X · W1 + b1
```

Después se aplica la función de activación para obtener:
Donde:

Z1: es el resultado de la combinación lineal anterior.

A1: representa las activaciones de las neuronas de la capa oculta después de aplicar ReLU o Sigmoide.

```
A1 = activación(Z1)
```

Posteriormente, la información de la capa oculta se utiliza para calcular la salida:
Donde:

A1: contiene las salidas de las neuronas de la capa oculta.

W2: contiene los pesos que conectan la capa oculta con la capa de salida.

b2: contiene el sesgo de la neurona de salida.

Z2: representa el valor obtenido en la capa de salida antes de aplicar la función de activación.

```
Z2 = A1 · W2 + b2
```

Finalmente:
Donde:

Z2: es el resultado de la combinación lineal de la capa de salida.

A2: es el resultado final de la red después de aplicar la función Sigmoide. En este caso, representa la predicción de la red, con un valor entre 0 y 1.

```
A2 = sigmoid(Z2)
```

#### Método compute_loss(y, ŷ)

Este método se utiliza para medir qué tan diferentes son las predicciones realizadas por la red respecto a los valores reales.

En este laboratorio se utilizó el **Error Cuadrático Medio (MSE, Mean Squared Error)**, cuya fórmula es:

```
MSE = (1/n) Σ(y - ŷ)²
```
Donde:

y: representa los valores reales o esperados.

ŷ: representa las predicciones realizadas por la red.

n: representa el número de ejemplos utilizados.

y - ŷ: representa la diferencia entre el valor real y la predicción.

(y - ŷ)²: eleva la diferencia al cuadrado para evitar que los errores positivos y negativos se cancelen.

Σ: indica que se suman los errores de todos los ejemplos.

1/n: permite obtener el promedio de los errores.

Una pérdida menor indica que las predicciones de la red están más próximas a los valores esperados.


#### Método backward(X, y, lr)

Este método implementa la retropropagación del error (*Backpropagation*). 
Su objetivo es determinar cuánto contribuyó cada peso y cada sesgo al error obtenido en la predicción, para posteriormente realizar los ajustes necesarios.


Durante este proceso se calculan los gradientes correspondientes a:

- dW1
- db1
- dW2
- db2

Donde:
dW2: indica cómo influye cada peso de la conexión entre la capa oculta y la capa de salida en el error.

db2: indica cómo influye el sesgo de la capa de salida en el error.

dW1: indica cómo influyen los pesos de la conexión entre la capa de entrada y la capa oculta en el error.

db1: indica cómo influyen los sesgos de la capa oculta en el erro

Estos gradientes indican cómo deben modificarse los parámetros de la red para reducir el error.

La actualización se realiza mediante descenso de gradiente:

```
W = W - lr · dW
b = b - lr · db
```

Donde: 
W: representa los pesos de la red.

b: representa los sesgos.

dW: representa el gradiente de los pesos.

db: representa el gradiente de los sesgos.

lr: representa la tasa de aprendizaje (learning rate).

#### Método train(X, y, lr, epochs)

Este método ejecuta el proceso completo de entrenamiento durante un número determinado de épocas.

En cada época se realizan las siguientes operaciones:

1. Se ejecuta el forward pass.
2. Se calcula la pérdida.
3. Se ejecuta el backward pass.
4. Se actualizan los pesos y sesgos.

De esta manera, la red modifica progresivamente sus parámetros para aproximarse a las salidas esperadas.

---

### 3. Clase DeepMLP

Como segunda implementación se agregó una capa oculta adicional.

La arquitectura utilizada es:

```
Entrada (2) → Oculta 1 (4) → Oculta 2 (4) → Salida (1)
```

En este caso se utilizan tres conjuntos de pesos y sesgos:

- W1, b1
- W2, b2
- W3, b3

La principal diferencia con SimpleMLP es que el error debe propagarse a través de una capa adicional.

Esto permite analizar cómo cambia el comportamiento de la red cuando aumenta su profundidad.

---

### 4. Flujo del Entrenamiento

El entrenamiento de ambas redes sigue el siguiente proceso:

```
Datos de entrada
       ↓
Forward propagation
       ↓
Predicción
       ↓
Cálculo de la pérdida
       ↓
Backpropagation
       ↓
Cálculo de gradientes
       ↓
Actualización de pesos y sesgos
       ↓
Siguiente época
```

Este proceso se repite hasta alcanzar el número de épocas establecido.

---

## Resultados de las Actividades Prácticas

### Actividad 1

![r1](/img1.png)

### Actividad 2

![r2](/img2.png)

### Actividad 3

![r3](/img3.png)

### Actividad 1: Modificación del learning rate

Para esta actividad se mantuvo la arquitectura base:

```
2 → 4 → 1
```

utilizando la función sigmoide, cuatro neuronas en la capa oculta y 5000 épocas de entrenamiento.

Los resultados obtenidos fueron:

| Learning Rate | Pérdida final | Predicciones XOR | Resultado |
|---------------|---------------|------------------|-----------|
| 0.9           | 0.003554      | [0, 1, 1, 0]     | Resuelve XOR |
| 0.1           | 0.249976      | [1, 1, 0, 0]     | No resuelve XOR |
| 0.0001        | 0.254985      | [0, 0, 0, 0]     | No resuelve XOR |

#### Análisis

Los resultados muestran que el valor del learning rate influye directamente en la velocidad y en el comportamiento del entrenamiento.

Con un learning rate de 0.9, la red logró reducir considerablemente la pérdida y obtuvo las cuatro predicciones correctas del problema XOR. En este experimento, la convergencia se produjo aproximadamente después de 3000 épocas.

Con 0.0001, las actualizaciones de los parámetros fueron muy pequeñas. Por esta razón, después de 5000 épocas la pérdida continuó cercana a su valor inicial y la red no consiguió aprender correctamente el patrón XOR.

En el caso de 0.1, la red alcanzó una pérdida cercana a 0.25, pero no logró encontrar una configuración de pesos que permitiera clasificar correctamente las cuatro entradas. Este comportamiento también depende de la inicialización de los pesos y de la dinámica del entrenamiento.

#### Conclusión de la actividad

Este experimento permite observar que el learning rate debe elegirse de acuerdo con las características del problema y de la arquitectura utilizada.

Un valor demasiado pequeño puede hacer que el aprendizaje sea muy lento, mientras que un valor demasiado grande puede producir actualizaciones excesivas y dificultar la convergencia. En esta configuración específica, el valor 0.9 presentó el mejor comportamiento entre los valores evaluados.

---

### Actividad 2: Comparación entre ReLU y Sigmoide

En esta actividad se modificó la función de activación de la capa oculta para comparar el comportamiento de ReLU y Sigmoide.

Los resultados fueron:

| Activación | Pérdida final | Predicciones XOR | Resultado | Épocas aprox. |
|------------|---------------|------------------|-----------|----------------|
| ReLU       | 0.166712      | [1, 1, 1, 0]     | No resuelve XOR | No converge correctamente |
| Sigmoide   | 0.099799      | [0, 1, 1, 0]     | Resuelve XOR | ~4500 |

#### Análisis

Con ReLU, algunas neuronas de la capa oculta dejaron de producir activaciones útiles para determinados ejemplos. Debido a que la derivada de ReLU es cero cuando su entrada es negativa, una neurona puede dejar de recibir actualizaciones efectivas si permanece en esa región.

Este comportamiento se conoce como "neurona muerta" y puede ser especialmente relevante cuando se trabaja con una red pequeña, como la utilizada en este experimento.

En cambio, utilizando Sigmoide, la red consiguió modificar progresivamente sus parámetros y finalmente obtuvo las cuatro predicciones correctas. Aunque el aprendizaje fue más lento, la configuración utilizada consiguió reducir el error hasta obtener una clasificación correcta de XOR.

#### Conclusión de la actividad

La comparación demuestra que la elección de la función de activación depende de la arquitectura y del problema.

En esta implementación específica, la Sigmoide obtuvo un mejor resultado que ReLU, debido a que permitió que la red encontrara una solución para XOR con la configuración utilizada.

Esto no significa que ReLU sea inferior en todos los casos. Su comportamiento depende de factores como la inicialización, la arquitectura, la tasa de aprendizaje y los datos utilizados.

---

### Actividad 3: Ampliación de la Arquitectura

En esta actividad se agregó una segunda capa oculta para comparar ambas arquitecturas:

MLP original:

```
2 → 4 → 1
```

MLP con mayor profundidad:

```
2 → 4 → 4 → 1
```

Los resultados obtenidos fueron:

| Arquitectura   | Pérdida final | Predicciones XOR | Resultado |
|----------------|---------------|------------------|-----------|
| 2 → 4 → 1      | 0.099799      | [0, 1, 1, 0]     | Resuelve XOR |
| 2 → 4 → 4 → 1  | 0.244155      | [0, 1, 1, 1]     | No resuelve completamente XOR |

#### Análisis

La arquitectura con una capa oculta consiguió resolver correctamente el problema XOR.

Al agregar una segunda capa oculta, la red obtuvo una mayor profundidad, pero esto no produjo una mejora en los resultados para esta configuración. Después de 5000 épocas, la pérdida solamente disminuyó aproximadamente de 0.2539 a 0.2441 y la última predicción continuó siendo incorrecta.

Una posible explicación es el desvanecimiento del gradiente (vanishing gradient). Al utilizar funciones sigmoides en varias capas, los gradientes pueden hacerse progresivamente pequeños durante la retropropagación. Como consecuencia, las primeras capas reciben actualizaciones menores y el aprendizaje puede volverse más lento.

#### Conclusión de la actividad

El experimento demuestra que incrementar el número de capas no garantiza un mejor resultado.

Para el problema XOR y la configuración utilizada, una arquitectura de:

```
2 → 4 → 1
```

fue suficiente para encontrar una solución.

La arquitectura:

```
2 → 4 → 4 → 1
```

introdujo una mayor profundidad, pero también hizo que el proceso de entrenamiento fuera más complejo. Para entrenar redes más profundas de manera eficiente pueden utilizarse estrategias como mejores métodos de inicialización, funciones de activación apropiadas, normalización, ajustes del learning rate y un mayor número de épocas.

---

## Conclusiones Generales

A partir de los experimentos realizados se obtuvieron las siguientes conclusiones:

1. **El MLP permite resolver el problema XOR.**
   La utilización de una capa oculta y funciones de activación no lineales permite que la red aprenda una relación que no puede ser representada mediante un modelo lineal simple.

2. **El learning rate influye significativamente en el entrenamiento.**
   En los experimentos realizados, un valor de 0.9 permitió obtener una reducción considerable de la pérdida y clasificar correctamente los cuatro casos de XOR. Los valores 0.1 y 0.0001 no consiguieron alcanzar el mismo resultado con las condiciones utilizadas.

3. **La función de activación afecta el aprendizaje.**
   En la configuración evaluada, Sigmoide permitió obtener una solución correcta para XOR, mientras que ReLU presentó problemas relacionados con la activación de algunas neuronas.

4. **Una mayor profundidad no garantiza mejores resultados.**
   La red de arquitectura 2 → 4 → 1 consiguió resolver XOR, mientras que la arquitectura 2 → 4 → 4 → 1 tuvo mayores dificultades para entrenarse con la configuración utilizada.

5. **La inicialización de los pesos influye en el resultado.**
   Al utilizar una semilla fija (seed = 42), los experimentos pueden reproducirse bajo las mismas condiciones. Sin embargo, diferentes inicializaciones pueden producir trayectorias de entrenamiento y resultados distintos.

---
