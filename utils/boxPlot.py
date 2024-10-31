# Parece que el estado se ha reiniciado, volveré a generar la gráfica desde el principio.

import matplotlib.pyplot as plt

# Datos del ejemplo
grupo_a = [25, 26, 24, 25, 26, 24, 25, 24]
grupo_b = [30, 35, 33, 32, 34, 35, 33, 36]

# Crear el diagrama de caja para el ejemplo
plt.figure(figsize=(8, 6))
plt.boxplot([grupo_a, grupo_b], labels=['Grupo A (3 veces/sem)', 'Grupo B (1 vez/sem)'])

# Título y etiquetas
plt.title('Comparación de tiempos de carrera entre dos grupos')
plt.ylabel('Tiempo (minutos)')
plt.grid(True)

# Mostrar gráfico
plt.show()
