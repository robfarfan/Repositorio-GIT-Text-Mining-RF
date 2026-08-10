# Laboratorio No. 3

La entrega principal está en `Laboratorio_No_3.ipynb`. El notebook ya contiene todas las celdas ejecutadas.
Se recomienda leer tambien `Reporte_Laboratorio_No_3.docx`. Como analisis final comparando imagen contra medidas reales.

## Reproducción

1. Mantener `moneda en papa.jpg` en este directorio.
2. Instalar las dependencias: `python -m pip install -r requirements.txt`.
3. Abrir `Laboratorio_No_3.ipynb` en Visual Studio Code y ejecutar todas las celdas.

Para regenerar también el notebook ejecutado, las imágenes y el reporte Word se puede ejecutar:

```powershell
python generar_entregables.py
```

Para verificar estructura, salidas y coherencia numérica:

```powershell
python validar_entregables.py
```

El resultado principal calculado es **8.65%** de área de moneda respecto del corte de papa. Es una aproximación bidimensional basada en bordes y contornos.
