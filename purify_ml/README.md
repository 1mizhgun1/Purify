# purify_ml

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

ML ядро для обработки текстовой информации для плагина Purify

#### 1. Baseline модель для классификации текстов

* Классификация текстов по тональности: ```нейтральный (0)``` и ```агрессивный (1).```

* Классический ```ML-анализ```$^{*}$

\* Возможно применение и алгоритмов глубокого обучения, например ```BERT.```

## Структура проекта

```
├── LICENSE            
├── Makefile          
├── README.md          
├── data
│   ├── external      
│   ├── interim        
│   ├── processed     
│   └── raw                        
│
├── models             
│
├── notebooks          
│                                              
├── pyproject.toml    
│                        
├── references        
│
├── reports            
│   └── figures       
│
├── requirements.txt   
│                        
├── setup.cfg        
│
└── purify_ml  
    │
    ├── __init__.py            
    │
    ├── config.py             
    │
    ├── dataset.py            
    │
    ├── features.py            
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py              
    │   └── train.py          
    │
    └── plots.py             
```

--------

