# Entrenamiento y validación de k-NN en dataset pequeño

## Descripción del respositorio
El objetivo es el reporte en sí, en él se comenta sobre el código fuente de los scripts de Python. A continuación describo la información mínima para generar el reporte, el cual debería verse como `reporte_precompilado.pdf`.

## Setup
Se debe de tener una instalación de python3 con pip.

```bash
virtualenv env
. env/bin/activate
pip install numpy matplotlib pandas scikit-learn pandoc pandoc-include
```

## Compilar reporte
Es necesario también una distribución de Latex. Correr `main.py` puede demorar unos minutos dependiendo del equipo.

```bash
python main.py
pandoc -i reporte.md --filter pandoc-include --lua-filter=subfigs.lua -s -o reporte.pdf --pdf-engine=pdflatex
```

## Posible adición
Estimar un intervalo de confianza por métodos estadísticos.

## Nota
`subfigs.lua` es tomado de la repo [rnwst/pandoc-subfigs](https://github.com/rnwst/pandoc-subfigs/).
